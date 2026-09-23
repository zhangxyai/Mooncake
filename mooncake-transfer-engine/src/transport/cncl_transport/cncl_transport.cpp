// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/cncl_transport/cncl_transport.h"

#include <cncl.h>
#include <cnrt.h>
#include <glog/logging.h>
#if __has_include(<jsoncpp/json/json.h>)
#include <jsoncpp/json/json.h>
#else
#include <json/json.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "config.h"
#include "error.h"
#include "transfer_metadata.h"

#if CNCL_VERSION < 10300
#error "Mooncake CNCL transport requires CNCL 1.30 or newer"
#endif

namespace mooncake {
namespace {

constexpr char kHandshakeProtocol[] = "cncl";
constexpr int kSessionTimeoutSeconds = 60;

struct BufferInfo {
    uint64_t addr = 0;
    uint64_t length = 0;
    int device_id = -1;
};

bool containsRange(const BufferInfo& buffer, uint64_t addr, size_t length) {
    if (length == 0 || addr < buffer.addr || length > buffer.length) {
        return false;
    }
    return addr - buffer.addr <= buffer.length - length;
}

std::string cnclError(cnclResult_t result, const char* operation) {
    std::ostringstream out;
    out << operation << " failed: " << cnclGetErrorStr(result);
    return out.str();
}

std::string cnrtError(cnrtRet_t result, const char* operation) {
    std::ostringstream out;
    out << operation << " failed: " << cnrtGetErrorStr(result);
    return out.str();
}

std::string encodeJson(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    return Json::writeString(builder, value);
}

bool decodeJson(const std::string& encoded, Json::Value* value,
                std::string* error) {
    if (!value) return false;
    Json::CharReaderBuilder builder;
    builder["allowComments"] = false;
    builder["allowTrailingCommas"] = false;
    builder["failIfExtra"] = true;
    builder["rejectDupKeys"] = true;
    builder["strictRoot"] = true;
    std::istringstream input(encoded);
    try {
        if (!Json::parseFromStream(builder, input, value, error)) return false;
    } catch (const Json::Exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    if (!value->isObject()) {
        if (error) *error = "CNCL handshake payload is not an object";
        return false;
    }
    return true;
}

bool hasStringField(const Json::Value& value, const char* name) {
    return value.isObject() && value.isMember(name) && value[name].isString();
}

bool hasIntField(const Json::Value& value, const char* name) {
    return value.isObject() && value.isMember(name) && value[name].isInt();
}

bool hasUInt64Field(const Json::Value& value, const char* name) {
    return value.isObject() && value.isMember(name) && value[name].isUInt64();
}

std::string encodeCliqueId(const cnclCliqueId& id) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&id);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < sizeof(id); ++i) {
        out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return out.str();
}

int decodeHexNibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

bool decodeCliqueId(const std::string& encoded, cnclCliqueId* id) {
    if (!id || encoded.size() != 2 * sizeof(*id)) return false;
    auto* bytes = reinterpret_cast<unsigned char*>(id);
    for (size_t i = 0; i < sizeof(*id); ++i) {
        const int high = decodeHexNibble(encoded[2 * i]);
        const int low = decodeHexNibble(encoded[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        bytes[i] = static_cast<unsigned char>((high << 4) | low);
    }
    return true;
}

int getPointerDevice(const void* ptr) {
    cnrtPointerAttributes_t attributes;
    cnrtRet_t result = cnrtPointerGetAttributes(&attributes, ptr);
    if (result != cnrtSuccess) {
        cnrtGetLastError();
        return -1;
    }
    if (attributes.type != cnrtMemTypeDevice) return -1;
    return attributes.device;
}

bool endpointLess(const std::string& local_name, int local_device,
                  const std::string& peer_name, int peer_device) {
    return std::tie(local_name, local_device) <
           std::tie(peer_name, peer_device);
}

std::string makeSessionKey(const std::string& local_name, int local_device,
                           const std::string& peer_name, int peer_device) {
    std::ostringstream out;
    if (endpointLess(local_name, local_device, peer_name, peer_device)) {
        out << local_name << '#' << local_device << '|' << peer_name << '#'
            << peer_device;
    } else {
        out << peer_name << '#' << peer_device << '|' << local_name << '#'
            << local_device;
    }
    return out.str();
}

// Completion for the CNCL transport is tracked per submission group rather
// than per descriptor. CNCL is two-sided: it has no shared completion ring, so
// every descriptor in flight needs its own cnrtNotifier_t, and that object is
// scarce on the device (cnrtNotifierCreate fails with
// CN_OPS_ERROR_OUT_OF_RESOURCES once the pool is exhausted). A group is a run
// of cnclSend calls issued on one session's queue, followed by a single
// cnrtPlaceNotifier on that same queue. CNCL executes a queue in FIFO order,
// so the tail notifier completes only after every send of the group has
// completed: one notifier replaces one per descriptor, and the in-flight
// notifier count is bounded by the number of concurrent groups instead of by
// the number of descriptors.
struct CnclCompletionGroup {
    cnrtNotifier_t notifier = nullptr;
    int device_id = -1;
    int64_t start_nano = 0;
    // Deadline for the group to resolve, so a stuck send fails the batch
    // instead of hanging it. 0 disables the check.
    int64_t deadline_nano = 0;
    // Slices still referencing this group. The last one out deletes it.
    std::atomic<int64_t> slice_refs{0};
    // The notifier has been placed on the session queue.
    std::atomic<bool> armed{false};
    std::atomic<bool> resolved{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> notifier_freed{false};
    // Serializes notifier query and destroy, so each happens at most once.
    std::mutex resolve_mutex;
};

void destroyNotifier(cnrtNotifier_t notifier, int device_id) {
    int saved_device = -1;
    cnrtGetDevice(&saved_device);
    if (device_id >= 0) cnrtSetDevice(device_id);
    cnrtRet_t result = cnrtNotifierDestroy(notifier);
    if (saved_device >= 0) cnrtSetDevice(saved_device);
    if (result != cnrtSuccess) {
        cnrtGetLastError();
        LOG(ERROR) << "[CNCL] cnrtNotifierDestroy failed device=" << device_id
                   << ": " << cnrtGetErrorStr(result);
    }
}

// Releases the group's notifier exactly once, whichever gets there first: the
// poller that resolves the group, or the slice-cache cleanup of the last slice
// still holding a reference.
void freeGroupNotifier(CnclCompletionGroup* group) {
    if (!group) return;
    std::lock_guard<std::mutex> lock(group->resolve_mutex);
    if (group->notifier_freed.exchange(true)) return;
    cnrtNotifier_t notifier = group->notifier;
    group->notifier = nullptr;
    if (!notifier) return;
    destroyNotifier(notifier, group->device_id);
}

// Slice cleanup callback. Every slice of a group holds one reference; the
// group object is freed by whichever slice is released last.
void releaseCnclSliceResources(Transport::Slice* slice) {
    if (!slice) return;
    auto* group = static_cast<CnclCompletionGroup*>(slice->cncl.group);
    slice->cncl.group = nullptr;
    slice->cncl.device_id = -1;
    if (!group) return;
    if (group->slice_refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    // Reachable when the batch is torn down before the group resolved. A
    // resolved group has already returned its notifier to the device.
    freeGroupNotifier(group);
    delete group;
}

// Bounded retry around cnrtNotifierCreate. The device notifier pool is shared
// by every context in the process, so a burst of concurrent groups can
// exhaust it; retrying gives already-resolved groups time to return their
// notifiers instead of failing the submission outright. The budget also bounds
// the cost of a genuinely stuck pool, after which the group fails cleanly
// rather than hanging.
cnrtRet_t createNotifierWithRetry(int device_id, cnrtNotifier_t* notifier) {
    constexpr int64_t kRetryBudgetNano = 1000LL * 1000 * 1000;
    constexpr int64_t kMaxBackoffNano = 64LL * 1000 * 1000;
    const int64_t deadline_nano = getCurrentTimeInNano() + kRetryBudgetNano;
    int64_t backoff_nano = 1000 * 1000;
    for (;;) {
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtSetDevice(device_id);
        cnrtRet_t result = cnrtNotifierCreate(notifier);
        if (saved_device >= 0) cnrtSetDevice(saved_device);
        if (result == cnrtSuccess) return result;
        if (getCurrentTimeInNano() + backoff_nano >= deadline_nano) {
            return result;
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(backoff_nano));
        backoff_nano = std::min(backoff_nano * 2, kMaxBackoffNano);
    }
}

// Queries a group's tail notifier and, once the group is done, releases the
// notifier immediately so the device pool gets it back without waiting for the
// batch to be freed.
void resolveGroup(CnclCompletionGroup* group) {
    if (!group) return;
    {
        std::lock_guard<std::mutex> lock(group->resolve_mutex);
        if (group->resolved.load(std::memory_order_relaxed)) return;
        // Not armed yet: the notifier is not on the queue, so there is
        // nothing to query. A later poll picks it up.
        if (!group->armed.load(std::memory_order_relaxed)) return;

        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtSetDevice(group->device_id);
        cnrtRet_t result = cnrtQueryNotifier(group->notifier);
        if (saved_device >= 0) cnrtSetDevice(saved_device);

        if (result == cnrtSuccess) {
            group->resolved.store(true, std::memory_order_release);
        } else if (result != cnrtErrorNotReady) {
            cnrtGetLastError();
            LOG(ERROR) << "[CNCL] transfer failed: "
                       << cnrtError(result, "cnrtQueryNotifier");
            group->failed.store(true, std::memory_order_release);
            group->resolved.store(true, std::memory_order_release);
        } else if (group->deadline_nano > 0 &&
                   getCurrentTimeInNano() > group->deadline_nano) {
            LOG(ERROR) << "[CNCL] completion notifier unresolved after "
                       << (getCurrentTimeInNano() - group->start_nano) / 1e9
                       << "s (device=" << group->device_id
                       << "); failing the group";
            group->failed.store(true, std::memory_order_release);
            group->resolved.store(true, std::memory_order_release);
        }
    }
    if (group->resolved.load(std::memory_order_relaxed)) {
        freeGroupNotifier(group);
    }
}

// Places the group's tail notifier on the session queue. Must run after every
// send of the group has been enqueued: the queue is FIFO, so the notifier
// completes only once all of them have. The caller passes the session's submit
// lock so the place cannot interleave with a send that is still being issued
// on the same session.
int armGroup(std::mutex& submit_mutex, cnrtQueue_t queue,
             CnclCompletionGroup* group, bool has_sends, std::string* error) {
    if (!has_sends) {
        // Nothing reached the queue, so there is nothing to wait for.
        group->resolved.store(true, std::memory_order_release);
        freeGroupNotifier(group);
        return 0;
    }
    cnrtRet_t result = cnrtSuccess;
    {
        std::lock_guard<std::mutex> lock(submit_mutex);
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtSetDevice(group->device_id);
        result = cnrtPlaceNotifier(group->notifier, queue);
        if (result != cnrtSuccess) {
            // The sends are already enqueued. Do not let the caller reuse or
            // free their source buffers until the queue has drained.
            cnrtQueueSync(queue);
        }
        if (saved_device >= 0) cnrtSetDevice(saved_device);
    }
    if (result != cnrtSuccess) {
        if (error) *error = cnrtError(result, "cnrtPlaceNotifier");
        group->failed.store(true, std::memory_order_release);
        group->resolved.store(true, std::memory_order_release);
        freeGroupNotifier(group);
        return -1;
    }
    group->armed.store(true, std::memory_order_relaxed);
    return 0;
}

// CNCL refuses to initialize the same clique id twice within one process
// (CNCL_RET_ERR_REINIT / CLE070001), which is exactly what happens when both
// endpoints of a session live in the same process. Track the clique ids this
// process has already handed to cnclInitComms so that misconfiguration
// surfaces as a clear session failure instead of an opaque CNCL argument
// error. Ids are never released: cnclGetCliqueId never repeats an id, so the
// registry stays bounded by the session count.
class CliqueIdRegistry {
   public:
    // Returns false when the id was already reserved by another session.
    bool reserve(const std::string& clique_id, const std::string& session_key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return reserved_.emplace(clique_id, session_key).second;
    }

    std::string owner(const std::string& clique_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = reserved_.find(clique_id);
        return it == reserved_.end() ? std::string() : it->second;
    }

   private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::string> reserved_;
};

CliqueIdRegistry& cliqueIdRegistry() {
    static CliqueIdRegistry registry;
    return registry;
}

class CnclSession {
   public:
    CnclSession(std::string key, std::string peer_name, int local_device,
                int peer_device, int rank, cnclCliqueId clique_id)
        : key_(std::move(key)),
          peer_name_(std::move(peer_name)),
          local_device_(local_device),
          peer_device_(peer_device),
          rank_(rank),
          clique_id_(clique_id),
          clique_id_string_(encodeCliqueId(clique_id)) {}

    ~CnclSession() {
        if (init_thread_.joinable()) init_thread_.join();
        cleanup();
    }

    const std::string& cliqueIdString() const { return clique_id_string_; }
    const std::string& key() const { return key_; }
    int rank() const { return rank_; }
    int peerRank() const { return 1 - rank_; }
    int localDevice() const { return local_device_; }
    cnrtQueue_t queue() const { return queue_; }

    void start() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (started_) return;
        started_ = true;
        init_thread_ = std::thread([this] { initialize(); });
    }

    bool waitReady(std::string* error) {
        std::unique_lock<std::mutex> lock(state_mutex_);
        const bool finished = state_cv_.wait_for(
            lock, std::chrono::seconds(kSessionTimeoutSeconds),
            [this] { return ready_ || failed_; });
        if (!finished) {
            if (error) *error = "Timed out initializing CNCL session";
            return false;
        }
        if (failed_) {
            if (error) *error = error_;
            return false;
        }
        return true;
    }

    // Responder side of a WRITE: enqueue the cnclRecv that matches the
    // writer's cnclSend. Called from the peer's handshake daemon while it
    // holds the response, so a successful return guarantees the recv is
    // already queued before the writer can enqueue its send.
    int enqueueRecv(void* dest, size_t length, std::string* error) {
        if (!waitReady(error)) return -1;
        std::lock_guard<std::mutex> lock(enqueue_mutex_);
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtRet_t cnrt_result = cnrtSetDevice(local_device_);
        if (cnrt_result != cnrtSuccess) {
            if (error) *error = cnrtError(cnrt_result, "cnrtSetDevice");
            return -1;
        }
        cnclResult_t result =
            cnclRecv(dest, length, cnclUint8, peerRank(), comm_, queue_);
        if (saved_device >= 0) cnrtSetDevice(saved_device);
        if (result != CNCL_RET_SUCCESS) {
            if (error) *error = cnclError(result, "cnclRecv");
            return -1;
        }
        return 0;
    }

    // Writer side of a WRITE. The caller must hold submit_mutex() across the
    // reservation RPC and this call (see submitWrite below) so the order in
    // which sends and matching recvs are issued is identical on both sides.
    // Completion is not tracked here: the caller places one notifier after the
    // whole group of sends (see armGroup).
    int enqueueSend(const void* source, size_t length, std::string* error) {
        if (!waitReady(error)) return -1;
        std::lock_guard<std::mutex> lock(enqueue_mutex_);
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtRet_t cnrt_result = cnrtSetDevice(local_device_);
        if (cnrt_result != cnrtSuccess) {
            if (error) *error = cnrtError(cnrt_result, "cnrtSetDevice");
            return -1;
        }

        cnclResult_t result = cnclSend(const_cast<void*>(source), length,
                                       cnclUint8, peerRank(), comm_, queue_);
        if (saved_device >= 0) cnrtSetDevice(saved_device);

        if (result != CNCL_RET_SUCCESS) {
            if (error) *error = cnclError(result, "cnclSend");
            return -1;
        }
        return 0;
    }

    // Serializes [reservation RPC -> cnclSend] on the writer side. CNCL
    // matches cnclSend/cnclRecv per rank pair in issue order, and the RPC
    // round trip makes the peer enqueue its matching cnclRecv first; holding
    // this lock across both steps keeps every writer on the session issuing
    // operations in the same order the peer queues their recvs.
    std::mutex& submitMutex() { return submit_mutex_; }

   private:
    void setFailure(const std::string& error) {
        LOG(ERROR) << "[CNCL] session " << key_ << ": " << error;
        std::lock_guard<std::mutex> lock(state_mutex_);
        error_ = error;
        failed_ = true;
        state_cv_.notify_all();
    }

    void initialize() {
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtRet_t cnrt_result = cnrtSetDevice(local_device_);
        if (cnrt_result != cnrtSuccess) {
            setFailure(cnrtError(cnrt_result, "cnrtSetDevice"));
            return;
        }

        if (!cliqueIdRegistry().reserve(clique_id_string_, key_)) {
            setFailure(
                "CNCL clique id is already initialized in this process by "
                "session " +
                cliqueIdRegistry().owner(clique_id_string_) +
                "; a CNCL session cannot have both endpoints in one process");
            if (saved_device >= 0) cnrtSetDevice(saved_device);
            return;
        }

        cnclComm_t comm = nullptr;
        const int dev_list[1] = {local_device_};
        const int rank_list[1] = {rank_};
        cnclResult_t result =
            cnclInitComms(&comm, 1, dev_list, rank_list, 2, &clique_id_);
        if (result != CNCL_RET_SUCCESS || !comm) {
            if (result == CNCL_RET_SUCCESS) result = CNCL_RET_ERR_INTERNAL;
            setFailure(cnclError(result, "cnclInitComms"));
            if (saved_device >= 0) cnrtSetDevice(saved_device);
            return;
        }
        comm_ = comm;

        cnrtQueue_t queue = nullptr;
        cnrt_result = cnrtQueueCreate(&queue);
        if (cnrt_result != cnrtSuccess) {
            setFailure(cnrtError(cnrt_result, "cnrtQueueCreate"));
            cleanup();
            if (saved_device >= 0) cnrtSetDevice(saved_device);
            return;
        }
        queue_ = queue;

        if (saved_device >= 0) cnrtSetDevice(saved_device);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            ready_ = true;
        }
        state_cv_.notify_all();
        LOG(INFO) << "[CNCL] session ready peer=" << peer_name_
                  << " rank=" << rank_ << " local_device=" << local_device_
                  << " peer_device=" << peer_device_;
    }

    void cleanup() {
        if (!comm_) {
            if (queue_) {
                cnrtQueueDestroy(queue_);
                queue_ = nullptr;
            }
            return;
        }
        int saved_device = -1;
        cnrtGetDevice(&saved_device);
        cnrtSetDevice(local_device_);
        if (queue_) cnrtQueueSync(queue_);
        // cnclFreeComm waits for the communicator's running tasks. The
        // peer should quiesce its session as well: CNCL requires clique
        // communicators to be released together.
        cnclFreeComm(comm_);
        comm_ = nullptr;
        if (queue_) {
            cnrtQueueDestroy(queue_);
            queue_ = nullptr;
        }
        if (saved_device >= 0) cnrtSetDevice(saved_device);
    }

    std::string key_;
    std::string peer_name_;
    int local_device_;
    int peer_device_;
    int rank_;
    cnclCliqueId clique_id_{};
    std::string clique_id_string_;

    std::thread init_thread_;
    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    bool started_ = false;
    bool ready_ = false;
    bool failed_ = false;
    std::string error_;

    cnclComm_t comm_ = nullptr;
    cnrtQueue_t queue_ = nullptr;
    std::mutex enqueue_mutex_;
    std::mutex submit_mutex_;
};

}  // namespace

class CnclTransport::Impl {
   public:
    explicit Impl(CnclTransport* owner) : owner_(owner) {}

    ~Impl() {
        std::unordered_map<std::string, std::shared_ptr<CnclSession>> sessions;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions.swap(sessions_);
            clique_ids_.clear();
        }
        sessions.clear();
    }

    int install(const std::string& local_server_name,
                std::shared_ptr<TransferMetadata> metadata) {
        local_server_name_ = local_server_name;
        metadata_ = std::move(metadata);

        auto segment = std::make_shared<SegmentDesc>();
        segment->name = local_server_name_;
        segment->protocol = kHandshakeProtocol;
        int result = metadata_->addLocalSegment(
            LOCAL_SEGMENT_ID, local_server_name_, std::move(segment));
        if (result != 0) return result;

        result = metadata_->startHandshakeDaemon(
            [this](const HandShakeDesc& peer, HandShakeDesc& local) {
                return onHandshake(peer, local);
            },
            metadata_->localRpcMeta().rpc_port,
            metadata_->localRpcMeta().sockfd);
        if (result != 0) return result;
        return metadata_->updateLocalSegmentDesc();
    }

    int registerMemory(void* addr, size_t length, const std::string& location,
                       bool remote_accessible, bool update_metadata) {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        return registerMemoryLocked(addr, length, location, remote_accessible,
                                    update_metadata);
    }

    int unregisterMemory(void* addr, bool update_metadata) {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        return unregisterMemoryLocked(addr, update_metadata);
    }

    int registerMemoryBatch(const std::vector<BufferEntry>& buffer_list,
                            const std::string& location) {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        std::vector<void*> registered;
        registered.reserve(buffer_list.size());
        for (const auto& buffer : buffer_list) {
            int result = registerMemoryLocked(buffer.addr, buffer.length,
                                              location, true, false);
            if (result != 0) {
                for (auto it = registered.rbegin(); it != registered.rend();
                     ++it) {
                    unregisterMemoryLocked(*it, false);
                }
                return result;
            }
            registered.push_back(buffer.addr);
        }
        int result = metadata_->updateLocalSegmentDesc();
        if (result != 0) {
            for (auto it = registered.rbegin(); it != registered.rend(); ++it) {
                int rollback_result = unregisterMemoryLocked(*it, false);
                if (rollback_result != 0) {
                    LOG(ERROR) << "[CNCL] failed to roll back buffer " << *it
                               << " after metadata publication failure: "
                               << rollback_result;
                }
            }
        }
        return result;
    }

    int unregisterMemoryBatch(const std::vector<void*>& addr_list) {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        std::vector<BufferDesc> metadata_buffers;
        metadata_buffers.reserve(addr_list.size());
        for (size_t index = 0; index < addr_list.size(); ++index) {
            void* addr = addr_list[index];
            if (std::find(addr_list.begin(), addr_list.begin() + index, addr) !=
                addr_list.begin() + index) {
                return ERR_INVALID_ARGUMENT;
            }
            auto it = std::find_if(local_buffers_.begin(), local_buffers_.end(),
                                   [addr](const BufferInfo& buffer) {
                                       return buffer.addr ==
                                              reinterpret_cast<uint64_t>(addr);
                                   });
            if (it == local_buffers_.end()) return ERR_ADDRESS_NOT_REGISTERED;
            BufferDesc metadata_buffer;
            if (!findMetadataBuffer(addr, &metadata_buffer)) {
                return ERR_ADDRESS_NOT_REGISTERED;
            }
            metadata_buffers.push_back(std::move(metadata_buffer));
        }
        size_t removed = 0;
        for (; removed < addr_list.size(); ++removed) {
            void* addr = addr_list[removed];
            int result = metadata_->removeLocalMemoryBuffer(addr, false);
            if (result != 0) {
                restoreMetadataBuffers(metadata_buffers, removed);
                return result;
            }
        }
        int result = metadata_->updateLocalSegmentDesc();
        if (result != 0) {
            restoreMetadataBuffers(metadata_buffers, metadata_buffers.size());
            return result;
        }
        for (void* addr : addr_list) {
            releaseCnclBufferCache(addr);
            local_buffers_.erase(
                std::remove_if(local_buffers_.begin(), local_buffers_.end(),
                               [addr](const BufferInfo& buffer) {
                                   return buffer.addr ==
                                          reinterpret_cast<uint64_t>(addr);
                               }),
                local_buffers_.end());
        }
        return 0;
    }

    Status submitTasks(const std::vector<TransferTask*>& task_list) {
        Status overall = Status::OK();
        // Phase 1 resolves each descriptor's metadata and session without
        // touching the device, so a request that cannot be sent never costs a
        // notifier.
        std::vector<PendingSend> pending;
        for (TransferTask* task : task_list) {
            if (!task || !task->request) {
                overall = Status::InvalidArgument("Missing CNCL request");
                continue;
            }
            const TransferRequest& request = *task->request;
            task->total_bytes = request.length;

            Slice* slice = owner_->getSliceCache().allocate();
            slice->source_addr = request.source;
            slice->length = request.length;
            slice->opcode = request.opcode;
            slice->target_id = request.target_id;
            slice->task = task;
            slice->status = Slice::PENDING;
            slice->ts = getCurrentTimeInNano();
            slice->cncl.group = nullptr;
            slice->cncl.device_id = -1;
            slice->cleanup_callback = releaseCnclSliceResources;
            task->slice_list.push_back(slice);
            __sync_fetch_and_add(&task->slice_count, 1);

            std::string error;
            if (request.opcode != TransferRequest::WRITE) {
                error =
                    "CNCL transport supports WRITE only; a READ would have to "
                    "enqueue the peer's cnclSend before the local matching "
                    "cnclRecv, which cannot be ordered safely against "
                    "concurrent WRITEs on the same session";
                slice->markFailed();
                if (overall.ok()) {
                    overall = Status::NotSupportedTransport(error);
                }
                LOG(ERROR) << "[CNCL] submit failed: " << error;
                continue;
            }

            auto target = metadata_->getSegmentDescByID(request.target_id);
            if (!target) {
                error = "Target segment metadata is unavailable";
            } else if (request.target_id == LOCAL_SEGMENT_ID) {
                error = "CNCL transport requires a remote target";
            }

            BufferInfo remote_buffer;
            if (error.empty() &&
                !findRemoteBuffer(*target, request.target_offset,
                                  request.length, &remote_buffer)) {
                error = "Target address is not in a registered CNCL buffer";
            }

            int local_device = -1;
            if (error.empty()) {
                local_device = getPointerDevice(request.source);
                if (local_device < 0) {
                    error = "CNCL local buffer must be MLU device memory";
                } else if (!findLocalBuffer(
                               reinterpret_cast<uint64_t>(request.source),
                               request.length, local_device)) {
                    error = "CNCL local buffer is not registered";
                }
            }

            std::shared_ptr<CnclSession> session;
            if (error.empty()) {
                session = getOrCreateSession(target->name, local_device,
                                             remote_buffer.device_id, &error);
            }

            if (!error.empty()) {
                LOG(ERROR) << "[CNCL] submit failed: " << error;
                slice->markFailed();
                if (overall.ok()) overall = Status::Context(error);
                continue;
            }

            pending.push_back(PendingSend{slice, std::move(session),
                                          request.source, request.length,
                                          request.target_offset, local_device,
                                          remote_buffer.device_id,
                                          target->name});
        }

        // Phase 2 groups the pending sends by session and creates one notifier
        // per group instead of one per descriptor.
        std::unordered_map<CnclSession*, size_t> group_index;
        std::vector<std::vector<size_t>> group_members;
        for (size_t index = 0; index < pending.size(); ++index) {
            CnclSession* key = pending[index].session.get();
            auto inserted = group_index.emplace(key, group_members.size());
            if (inserted.second) group_members.emplace_back();
            group_members[inserted.first->second].push_back(index);
        }

        for (const auto& members : group_members) {
            const std::shared_ptr<CnclSession>& session =
                pending[members.front()].session;
            const int device_id = session->localDevice();

            cnrtNotifier_t notifier = nullptr;
            cnrtRet_t cnrt_result =
                createNotifierWithRetry(device_id, &notifier);
            if (cnrt_result != cnrtSuccess) {
                const std::string error =
                    cnrtError(cnrt_result, "cnrtNotifierCreate");
                // One log line per group: the pool is shared, so a burst would
                // otherwise emit one line per descriptor.
                LOG(ERROR) << "[CNCL] submit failed for " << members.size()
                           << " request(s) on session " << session->key()
                           << ": " << error;
                for (size_t index : members) {
                    pending[index].slice->markFailed();
                }
                if (overall.ok()) overall = Status::Context(error);
                continue;
            }

            auto* group = new CnclCompletionGroup();
            group->notifier = notifier;
            group->device_id = device_id;
            group->start_nano = getCurrentTimeInNano();
            const int64_t timeout_nano =
                globalConfig().cncl_group_timeout * 1000LL * 1000 * 1000;
            group->deadline_nano =
                timeout_nano > 0 ? group->start_nano + timeout_nano : 0;

            bool has_sends = false;
            std::string group_error;
            for (size_t index : members) {
                PendingSend& send = pending[index];
                // The slice holds a reference from the moment the group
                // exists, so a failure below still leaves the group reachable
                // for cleanup.
                send.slice->cncl.group = group;
                send.slice->cncl.device_id = device_id;
                group->slice_refs.fetch_add(1, std::memory_order_relaxed);

                std::string error;
                if (submitWrite(send.session, send.peer_name,
                                send.local_device, send.peer_device,
                                send.dest_addr, send.source, send.length,
                                &error) == 0) {
                    send.slice->status = Slice::POSTED;
                    has_sends = true;
                } else {
                    LOG(ERROR) << "[CNCL] submit failed: " << error;
                    send.slice->markFailed();
                    if (group_error.empty()) group_error = error;
                }
            }
            if (!group_error.empty() && overall.ok()) {
                overall = Status::Context(group_error);
            }

            // Phase 3 arms the group once all of its sends are on the queue.
            std::string arm_error;
            if (armGroup(session->submitMutex(), session->queue(), group,
                         has_sends, &arm_error) != 0) {
                LOG(ERROR) << "[CNCL] submit failed: " << arm_error;
                for (size_t index : members) {
                    // The group is already resolved as failed, so a
                    // concurrent poller may mark these slices first.
                    pending[index].slice->tryMarkFailed();
                }
                if (overall.ok()) overall = Status::Context(arm_error);
            }
        }
        return overall;
    }

    Status poll(BatchID batch_id, size_t task_id, TransferStatus& status) {
        auto& batch = Transport::toBatchDesc(batch_id);
        if (task_id >= batch.task_list.size()) {
            return Status::InvalidArgument("CNCL task ID out of range");
        }
        auto& task = batch.task_list[task_id];
        // A group covers every slice submitted on its session in one batch, so
        // query each distinct group once instead of once per slice.
        std::vector<CnclCompletionGroup*> queried;
        for (Slice* slice : task.slice_list) {
            if (!slice || __atomic_load_n(&slice->status, __ATOMIC_ACQUIRE) !=
                              Slice::POSTED) {
                continue;
            }
            auto* group = static_cast<CnclCompletionGroup*>(slice->cncl.group);
            if (!group) continue;
            if (std::find(queried.begin(), queried.end(), group) !=
                queried.end()) {
                continue;
            }
            queried.push_back(group);
            resolveGroup(group);
        }
        for (Slice* slice : task.slice_list) {
            if (!slice || __atomic_load_n(&slice->status, __ATOMIC_ACQUIRE) !=
                              Slice::POSTED) {
                continue;
            }
            auto* group = static_cast<CnclCompletionGroup*>(slice->cncl.group);
            if (!group || !group->resolved.load(std::memory_order_acquire)) {
                continue;
            }
            // tryMark*: a concurrent poller of the same batch (or abortBatch)
            // may resolve the same slice; only the first transition counts.
            if (group->failed.load(std::memory_order_acquire)) {
                slice->tryMarkFailed();
            } else {
                slice->tryMarkSuccess();
            }
        }

        uint64_t success_slice_count =
            __atomic_load_n(&task.success_slice_count, __ATOMIC_ACQUIRE);
        uint64_t failed_slice_count =
            __atomic_load_n(&task.failed_slice_count, __ATOMIC_ACQUIRE);
        // Completion counters publish the preceding byte updates.
        status.transferred_bytes =
            __atomic_load_n(&task.transferred_bytes, __ATOMIC_RELAXED);
        if (success_slice_count + failed_slice_count == task.slice_count) {
            status.s = failed_slice_count ? TransferStatusEnum::FAILED
                                          : TransferStatusEnum::COMPLETED;
            task.is_finished = true;
        } else {
            status.s = TransferStatusEnum::WAITING;
        }
        return Status::OK();
    }

   private:
    // A descriptor that passed validation and is waiting to be enqueued.
    struct PendingSend {
        Slice* slice;
        std::shared_ptr<CnclSession> session;
        const void* source;
        size_t length;
        uint64_t dest_addr;
        int local_device;
        int peer_device;
        std::string peer_name;
    };

    bool findMetadataBuffer(void* addr, BufferDesc* result) const {
        if (!result) return false;
        auto segment = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
        if (!segment) return false;
        auto it = std::find_if(segment->buffers.begin(), segment->buffers.end(),
                               [addr](const BufferDesc& buffer) {
                                   return buffer.addr ==
                                          reinterpret_cast<uint64_t>(addr);
                               });
        if (it == segment->buffers.end()) return false;
        *result = *it;
        return true;
    }

    void restoreMetadataBuffers(const std::vector<BufferDesc>& buffers,
                                size_t count) {
        for (size_t index = 0; index < count; ++index) {
            int result = metadata_->addLocalMemoryBuffer(buffers[index], false);
            if (result != 0) {
                LOG(ERROR) << "[CNCL] failed to restore metadata buffer "
                           << reinterpret_cast<void*>(buffers[index].addr)
                           << ": " << result;
            }
        }
    }

    // CNCL caches buffer information inside every communicator; dropping the
    // registration also drops that cache so a future allocation reusing the
    // address cannot be misclassified.
    void releaseCnclBufferCache(void* addr) {
        cnclResult_t result = cnclFreeBufferCache(addr);
        if (result != CNCL_RET_SUCCESS) {
            LOG(WARNING) << "[CNCL] cnclFreeBufferCache(" << addr
                         << ") failed: " << cnclGetErrorStr(result);
        }
    }

    int registerMemoryLocked(void* addr, size_t length,
                             const std::string& location,
                             bool remote_accessible, bool update_metadata) {
        (void)remote_accessible;
        if (!addr || length == 0) return ERR_INVALID_ARGUMENT;
        int device_id = getPointerDevice(addr);
        if (device_id < 0) {
            LOG(ERROR) << "[CNCL] only MLU device memory can be registered";
            return ERR_INVALID_ARGUMENT;
        }

        const uint64_t address = reinterpret_cast<uint64_t>(addr);
        if (length > std::numeric_limits<uint64_t>::max() - address) {
            LOG(ERROR) << "[CNCL] memory registration range overflows";
            return ERR_INVALID_ARGUMENT;
        }

        BufferInfo info{address, length, device_id};
        for (const auto& buffer : local_buffers_) {
            const uint64_t lhs_end = info.addr + info.length;
            const uint64_t rhs_end = buffer.addr + buffer.length;
            if (info.addr < rhs_end && buffer.addr < lhs_end) {
                LOG(ERROR) << "[CNCL] overlapping memory registration";
                return ERR_ADDRESS_OVERLAPPED;
            }
        }
        local_buffers_.push_back(info);

        BufferDesc desc;
        desc.name = location;
        desc.addr = info.addr;
        desc.length = info.length;
        desc.device_id = info.device_id;
        int result = metadata_->addLocalMemoryBuffer(desc, update_metadata);
        if (result != 0) {
            int rollback_result =
                metadata_->removeLocalMemoryBuffer(addr, false);
            if (rollback_result != 0 &&
                rollback_result != ERR_ADDRESS_NOT_REGISTERED) {
                LOG(ERROR) << "[CNCL] failed to roll back metadata for " << addr
                           << ": " << rollback_result;
            }
            local_buffers_.erase(
                std::remove_if(local_buffers_.begin(), local_buffers_.end(),
                               [addr](const BufferInfo& buffer) {
                                   return buffer.addr ==
                                          reinterpret_cast<uint64_t>(addr);
                               }),
                local_buffers_.end());
        }
        return result;
    }

    int unregisterMemoryLocked(void* addr, bool update_metadata) {
        auto it = std::find_if(local_buffers_.begin(), local_buffers_.end(),
                               [addr](const BufferInfo& buffer) {
                                   return buffer.addr ==
                                          reinterpret_cast<uint64_t>(addr);
                               });
        if (it == local_buffers_.end()) return ERR_ADDRESS_NOT_REGISTERED;

        BufferDesc metadata_buffer;
        if (!findMetadataBuffer(addr, &metadata_buffer)) {
            return ERR_ADDRESS_NOT_REGISTERED;
        }
        int result = metadata_->removeLocalMemoryBuffer(addr, update_metadata);
        if (result != 0) {
            if (update_metadata) {
                restoreMetadataBuffers({metadata_buffer}, 1);
            }
            return result;
        }
        releaseCnclBufferCache(addr);
        local_buffers_.erase(it);
        return 0;
    }

    bool findLocalBuffer(uint64_t addr, size_t length, int device_id) const {
        std::lock_guard<std::mutex> lock(buffers_mutex_);
        for (const auto& buffer : local_buffers_) {
            if (buffer.device_id == device_id &&
                containsRange(buffer, addr, length)) {
                return true;
            }
        }
        return false;
    }

    bool findRemoteBuffer(const SegmentDesc& segment, uint64_t addr,
                          size_t length, BufferInfo* result) const {
        for (const auto& buffer : segment.buffers) {
            BufferInfo info{buffer.addr, buffer.length, buffer.device_id};
            if (info.device_id >= 0 && containsRange(info, addr, length)) {
                if (result) *result = info;
                return true;
            }
        }
        return false;
    }

    bool selectCliqueId(const std::string& key, const std::string& proposed_id,
                        bool may_create, cnclCliqueId* clique_id,
                        std::string* encoded_id, std::string* error) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto it = clique_ids_.find(key);
        if (it != clique_ids_.end()) {
            if (!proposed_id.empty() && proposed_id != it->second) {
                if (error) *error = "Concurrent CNCL bootstrap ID mismatch";
                return false;
            }
            *encoded_id = it->second;
            if (!decodeCliqueId(*encoded_id, clique_id)) {
                if (error) *error = "Stored CNCL clique ID is invalid";
                return false;
            }
            return true;
        }

        if (!proposed_id.empty()) {
            *encoded_id = proposed_id;
            if (!decodeCliqueId(*encoded_id, clique_id)) {
                if (error) *error = "Invalid CNCL clique ID in bootstrap";
                return false;
            }
        } else {
            if (!may_create) {
                if (error) *error = "CNCL rank 1 cannot create a clique ID";
                return false;
            }
            cnclResult_t result = cnclGetCliqueId(clique_id);
            if (result != CNCL_RET_SUCCESS) {
                if (error) *error = cnclError(result, "cnclGetCliqueId");
                return false;
            }
            *encoded_id = encodeCliqueId(*clique_id);
        }
        clique_ids_.emplace(key, *encoded_id);
        return true;
    }

    std::shared_ptr<CnclSession> getOrCreateSession(
        const std::string& peer_name, int local_device, int peer_device,
        std::string* error) {
        const std::string key = makeSessionKey(local_server_name_, local_device,
                                               peer_name, peer_device);
        std::shared_ptr<CnclSession> existing;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            auto it = sessions_.find(key);
            if (it != sessions_.end()) {
                existing = it->second;
            }
        }
        if (existing) {
            // CNCL communicator initialization is collective. Retain a
            // terminally failed session rather than letting one endpoint retry
            // a new collective generation without coordinating its peer.
            if (!existing->waitReady(error)) return nullptr;
            return existing;
        }

        const int local_rank = endpointLess(local_server_name_, local_device,
                                            peer_name, peer_device)
                                   ? 0
                                   : 1;
        cnclCliqueId clique_id{};
        std::string clique_id_string;
        if (local_rank == 0) {
            int saved_device = -1;
            cnrtGetDevice(&saved_device);
            cnrtSetDevice(local_device);
            const bool ok = selectCliqueId(key, "", true, &clique_id,
                                           &clique_id_string, error);
            if (saved_device >= 0) cnrtSetDevice(saved_device);
            if (!ok) return nullptr;
        }

        Json::Value request;
        request["op"] = "bootstrap";
        request["peer_name"] = local_server_name_;
        request["local_device"] = local_device;
        request["remote_device"] = peer_device;
        request["unique_id"] = clique_id_string;

        HandShakeDesc local_desc;
        local_desc.payload = encodeJson(request);
        HandShakeDesc peer_desc;
        int result = metadata_->sendHandshake(peer_name, local_desc, peer_desc);
        if (result != 0) {
            if (error) *error = "CNCL bootstrap handshake failed";
            return nullptr;
        }

        Json::Value response;
        std::string parse_error;
        if (!decodeJson(peer_desc.payload, &response, &parse_error)) {
            if (error)
                *error = "Invalid CNCL bootstrap response: " + parse_error;
            return nullptr;
        }
        if (!hasStringField(response, "unique_id")) {
            if (error) *error = "Malformed CNCL bootstrap response";
            return nullptr;
        }
        const std::string response_id = response["unique_id"].asString();
        if (!selectCliqueId(key, response_id, local_rank == 0, &clique_id,
                            &clique_id_string, error)) {
            return nullptr;
        }

        std::shared_ptr<CnclSession> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            auto it = sessions_.find(key);
            if (it != sessions_.end()) {
                session = it->second;
                if (session->cliqueIdString() != clique_id_string) {
                    if (error) *error = "Concurrent CNCL bootstrap conflict";
                    return nullptr;
                }
            } else {
                session = std::make_shared<CnclSession>(
                    key, peer_name, local_device, peer_device, local_rank,
                    clique_id);
                sessions_.emplace(key, session);
                session->start();
            }
        }
        if (!session->waitReady(error)) return nullptr;
        return session;
    }

    // Reserve the peer's cnclRecv and enqueue the local cnclSend as one
    // session-ordered step. The submit lock must cover the RPC round trip:
    // the reply is only sent after the peer queued the matching cnclRecv, so
    // sends and recvs are issued in the same order on both endpoints even
    // when several writer threads share the session. Completion is tracked by
    // the caller's group notifier, not per send.
    int submitWrite(const std::shared_ptr<CnclSession>& session,
                    const std::string& peer_name, int local_device,
                    int peer_device, uint64_t dest_addr, const void* source,
                    size_t length, std::string* error) {
        std::lock_guard<std::mutex> lock(session->submitMutex());

        Json::Value request;
        request["op"] = "write";
        request["peer_name"] = local_server_name_;
        request["writer_device"] = local_device;
        request["target_device"] = peer_device;
        request["dest_addr"] = static_cast<Json::UInt64>(dest_addr);
        request["length"] = static_cast<Json::UInt64>(length);

        HandShakeDesc local_desc;
        local_desc.payload = encodeJson(request);
        HandShakeDesc peer_desc;
        int result = metadata_->sendHandshake(peer_name, local_desc, peer_desc);
        if (result != 0) {
            if (error) *error = "CNCL write handshake failed";
            return -1;
        }
        return session->enqueueSend(source, length, error);
    }

    int onHandshake(const HandShakeDesc& peer_desc, HandShakeDesc& local_desc) {
        Json::Value request;
        std::string error;
        if (!decodeJson(peer_desc.payload, &request, &error)) {
            local_desc.reply_msg = "Invalid CNCL handshake payload: " + error;
            return 0;
        }
        if (!hasStringField(request, "op")) {
            local_desc.reply_msg =
                "Missing or invalid 'op' field in CNCL handshake request";
            return 0;
        }

        const std::string op = request["op"].asString();
        Json::Value response;
        try {
            if (op == "bootstrap") {
                if (handleBootstrap(request, &response, &error) != 0) {
                    local_desc.reply_msg = error;
                }
            } else if (op == "write") {
                if (handleWrite(request, &response, &error) != 0) {
                    local_desc.reply_msg = error;
                }
            } else {
                local_desc.reply_msg = "Unknown CNCL handshake operation";
            }
        } catch (const Json::Exception& exception) {
            local_desc.reply_msg = "Malformed CNCL handshake request: " +
                                   std::string(exception.what());
        }
        local_desc.payload = encodeJson(response);
        return 0;
    }

    int handleBootstrap(const Json::Value& request, Json::Value* response,
                        std::string* error) {
        if (!response || !hasStringField(request, "peer_name") ||
            !hasIntField(request, "local_device") ||
            !hasIntField(request, "remote_device") ||
            !hasStringField(request, "unique_id")) {
            if (error) *error = "Malformed CNCL bootstrap request";
            return -1;
        }
        const std::string peer_name = request["peer_name"].asString();
        const int peer_device = request["local_device"].asInt();
        const int local_device = request["remote_device"].asInt();
        if (peer_name.empty() || peer_name == local_server_name_ ||
            local_device < 0 || peer_device < 0) {
            if (error) *error = "Invalid CNCL bootstrap endpoint";
            return -1;
        }

        const int local_rank = endpointLess(local_server_name_, local_device,
                                            peer_name, peer_device)
                                   ? 0
                                   : 1;
        const std::string key = makeSessionKey(local_server_name_, local_device,
                                               peer_name, peer_device);
        const std::string proposed_id = request["unique_id"].asString();
        cnclCliqueId proposed_clique_id{};
        if ((!proposed_id.empty() &&
             !decodeCliqueId(proposed_id, &proposed_clique_id)) ||
            (proposed_id.empty() && local_rank != 0)) {
            if (error) *error = "Invalid CNCL clique ID in bootstrap";
            return -1;
        }

        cnclCliqueId clique_id{};
        std::string clique_id_string;
        if (local_rank == 0) {
            int saved_device = -1;
            cnrtGetDevice(&saved_device);
            cnrtSetDevice(local_device);
            const bool ok = selectCliqueId(key, proposed_id, true, &clique_id,
                                           &clique_id_string, error);
            if (saved_device >= 0) cnrtSetDevice(saved_device);
            if (!ok) return -1;
        } else {
            if (!selectCliqueId(key, proposed_id, false, &clique_id,
                                &clique_id_string, error)) {
                return -1;
            }
        }

        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            auto it = sessions_.find(key);
            if (it != sessions_.end()) {
                if (it->second->cliqueIdString() != clique_id_string) {
                    if (error) *error = "Concurrent CNCL bootstrap conflict";
                    return -1;
                }
            } else {
                auto session = std::make_shared<CnclSession>(
                    key, peer_name, local_device, peer_device, local_rank,
                    clique_id);
                sessions_.emplace(key, session);
                session->start();
            }
        }

        (*response)["unique_id"] = clique_id_string;
        return 0;
    }

    int handleWrite(const Json::Value& request, Json::Value* response,
                    std::string* error) {
        if (!response || !hasStringField(request, "peer_name") ||
            !hasIntField(request, "writer_device") ||
            !hasIntField(request, "target_device") ||
            !hasUInt64Field(request, "dest_addr") ||
            !hasUInt64Field(request, "length")) {
            if (error) *error = "Malformed CNCL write request";
            return -1;
        }
        const std::string peer_name = request["peer_name"].asString();
        const int peer_device = request["writer_device"].asInt();
        const int local_device = request["target_device"].asInt();
        const uint64_t dest_addr = request["dest_addr"].asUInt64();
        const size_t length = static_cast<size_t>(request["length"].asUInt64());
        if (peer_name.empty() || peer_name == local_server_name_ ||
            local_device < 0 || peer_device < 0 || length == 0) {
            if (error) *error = "Invalid CNCL write endpoint";
            return -1;
        }

        const std::string key = makeSessionKey(local_server_name_, local_device,
                                               peer_name, peer_device);
        std::shared_ptr<CnclSession> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            auto it = sessions_.find(key);
            if (it != sessions_.end()) session = it->second;
        }
        if (!session) {
            if (error)
                *error = "No CNCL session for peer " + peer_name +
                         "; the writer must bootstrap the session first";
            return -1;
        }

        // The destination must be inside a buffer this endpoint registered on
        // the targeted device before the recv is queued.
        {
            std::lock_guard<std::mutex> lock(buffers_mutex_);
            bool found = false;
            for (const auto& buffer : local_buffers_) {
                if (buffer.device_id == local_device &&
                    containsRange(buffer, dest_addr, length)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (error)
                    *error =
                        "CNCL target address is not in a registered "
                        "buffer on device " +
                        std::to_string(local_device);
                return -1;
            }
        }

        return session->enqueueRecv(reinterpret_cast<void*>(dest_addr), length,
                                    error);
    }

    CnclTransport* owner_;
    std::string local_server_name_;
    std::shared_ptr<TransferMetadata> metadata_;

    mutable std::mutex buffers_mutex_;
    std::vector<BufferInfo> local_buffers_;
    std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<CnclSession>> sessions_;
    std::unordered_map<std::string, std::string> clique_ids_;
};

CnclTransport::CnclTransport() : impl_(std::make_unique<Impl>(this)) {}

CnclTransport::~CnclTransport() = default;

int CnclTransport::install(std::string& local_server_name,
                           std::shared_ptr<TransferMetadata> metadata,
                           std::shared_ptr<Topology> topology) {
    (void)topology;
    local_server_name_ = local_server_name;
    metadata_ = metadata;
    return impl_->install(local_server_name, std::move(metadata));
}

Status CnclTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {
    auto& batch = toBatchDesc(batch_id);
    if (batch.task_list.size() + entries.size() > batch.batch_size) {
        return Status::TooManyRequests("CNCL batch capacity exceeded");
    }
    const size_t first = batch.task_list.size();
    batch.task_list.resize(first + entries.size());
    std::vector<TransferTask*> tasks;
    tasks.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        auto& task = batch.task_list[first + i];
        task.batch_id = batch_id;
        task.transport_ = this;
        task.request = &entries[i];
        tasks.push_back(&task);
    }
    return impl_->submitTasks(tasks);
}

Status CnclTransport::submitTransferTask(
    const std::vector<TransferTask*>& task_list) {
    return impl_->submitTasks(task_list);
}

Status CnclTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                        TransferStatus& status) {
    return impl_->poll(batch_id, task_id, status);
}

int CnclTransport::registerLocalMemory(void* addr, size_t length,
                                       const std::string& location,
                                       bool remote_accessible,
                                       bool update_metadata) {
    return impl_->registerMemory(addr, length, location, remote_accessible,
                                 update_metadata);
}

int CnclTransport::unregisterLocalMemory(void* addr, bool update_metadata) {
    return impl_->unregisterMemory(addr, update_metadata);
}

int CnclTransport::registerLocalMemoryBatch(
    const std::vector<BufferEntry>& buffer_list, const std::string& location) {
    return impl_->registerMemoryBatch(buffer_list, location);
}

int CnclTransport::unregisterLocalMemoryBatch(
    const std::vector<void*>& addr_list) {
    return impl_->unregisterMemoryBatch(addr_list);
}

}  // namespace mooncake
