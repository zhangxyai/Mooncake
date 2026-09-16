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

// CNCL transport validation example.
//
// Unlike the NCCL host example, which hosts both endpoints in one process,
// this example must run each endpoint in its own process: CNCL refuses to
// initialize the same clique id twice within a process, so a CNCL session
// can never have both ends in one address space. Run without arguments, the
// binary re-executes itself twice with --rank 0/--rank 1 and a shared
// workdir used to exchange engine names and step barriers.

#include <cnrt.h>
#include <glog/logging.h>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "transfer_engine.h"

namespace {

using mooncake::BatchID;
using mooncake::SegmentHandle;
using mooncake::TransferEngine;
using mooncake::TransferRequest;
using mooncake::TransferStatus;
using mooncake::TransferStatusEnum;

constexpr size_t kBufferBytes = 16 << 20;     // 16 MiB
constexpr size_t kTransferBytes = 256 << 10;  // 256 KiB
constexpr size_t kOffsetBytes = 4096;
constexpr size_t kLargeBytes = 8 << 20;  // 8 MiB
constexpr size_t kLargeOffset = 4 << 20;
constexpr size_t kConcurrentOffset = 1 << 20;
constexpr size_t kBidirectionalOffset = 14 << 20;

bool checkCnrt(cnrtRet_t result, const char* operation) {
    if (result == cnrtSuccess) return true;
    LOG(ERROR) << operation << " failed: " << cnrtGetErrorStr(result);
    return false;
}

// --- workdir helpers -------------------------------------------------------

std::string filePath(const std::string& workdir, const std::string& name) {
    return workdir + "/" + name;
}

bool writeFile(const std::string& path, const std::string& content) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        if (!out.is_open()) return false;
        out << content;
        if (!out.good()) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool readFile(const std::string& path, std::string* content) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    *content = ss.str();
    return !content->empty();
}

bool waitForFile(const std::string& path, std::string* content) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        if (readFile(path, content)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    LOG(ERROR) << "Timed out waiting for " << path;
    return false;
}

// Both ranks rendezvous on a per-step file pair.
bool barrier(const std::string& workdir, int rank, const std::string& step) {
    const std::string mine =
        filePath(workdir, step + "." + std::to_string(rank));
    if (!writeFile(mine, "ok")) {
        LOG(ERROR) << "Failed to write barrier file " << mine;
        return false;
    }
    const std::string peer =
        filePath(workdir, step + "." + std::to_string(1 - rank));
    std::string ignored;
    return waitForFile(peer, &ignored);
}

// --- transfer helpers ------------------------------------------------------

bool waitForTransfer(TransferEngine* engine, BatchID batch) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
        TransferStatus status{};
        auto result = engine->getTransferStatus(batch, 0, status);
        if (!result.ok()) {
            LOG(ERROR) << result.ToString();
            return false;
        }
        if (status.s == TransferStatusEnum::COMPLETED) return true;
        if (status.s == TransferStatusEnum::FAILED ||
            status.s == TransferStatusEnum::TIMEOUT) {
            LOG(ERROR) << "Transfer failed with status " << status.s;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG(ERROR) << "Timed out waiting for CNCL transfer";
    return false;
}

bool submitWrite(TransferEngine* engine, SegmentHandle target, void* local,
                 uint64_t remote, size_t length) {
    BatchID batch = engine->allocateBatchID(1);
    if (batch == mooncake::INVALID_BATCH_ID) return false;

    TransferRequest request{};
    request.opcode = TransferRequest::WRITE;
    request.source = local;
    request.target_id = target;
    request.target_offset = remote;
    request.length = length;
    auto result = engine->submitTransfer(batch, {request});
    if (!result.ok()) {
        LOG(ERROR) << result.ToString();
        return false;
    }
    if (!waitForTransfer(engine, batch)) return false;
    result = engine->freeBatchID(batch);
    if (!result.ok()) {
        LOG(ERROR) << result.ToString();
        return false;
    }
    return true;
}

// A rejected request must both return a submit error and drive its task to
// FAILED so the batch cannot report success.
bool expectSubmitRejected(TransferEngine* engine, SegmentHandle target,
                          void* local, uint64_t remote,
                          TransferRequest::OpCode opcode, size_t length,
                          bool expect_not_supported) {
    BatchID batch = engine->allocateBatchID(1);
    if (batch == mooncake::INVALID_BATCH_ID) return false;

    TransferRequest request{};
    request.opcode = opcode;
    request.source = local;
    request.target_id = target;
    request.target_offset = remote;
    request.length = length;
    auto result = engine->submitTransfer(batch, {request});
    const bool rejected = !result.ok();
    if (!rejected) {
        LOG(ERROR) << "Request unexpectedly accepted: " << result.ToString();
    } else if (expect_not_supported && !result.IsNotSupportedTransport()) {
        LOG(ERROR) << "Expected NotSupportedTransport, got "
                   << result.ToString();
        return false;
    }

    TransferStatus status{};
    auto status_result = engine->getTransferStatus(batch, 0, status);
    const bool failed =
        status_result.ok() && status.s == TransferStatusEnum::FAILED;
    if (!failed) {
        LOG(ERROR) << "Rejected request did not reach FAILED status";
    }

    auto free_result = engine->freeBatchID(batch);
    if (!free_result.ok()) LOG(ERROR) << free_result.ToString();
    return rejected && failed && free_result.ok();
}

// --- verification ----------------------------------------------------------

// The writer's completion notifier only proves the send left the local
// device, so the receiver polls its MLU memory until the pattern lands (or
// times out).
bool verifyRegion(void* device_ptr, int device, size_t offset, size_t length,
                  uint8_t expected) {
    if (!checkCnrt(cnrtSetDevice(device), "cnrtSetDevice")) return false;
    std::vector<uint8_t> host(length);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!checkCnrt(
                cnrtMemcpy(host.data(), static_cast<char*>(device_ptr) + offset,
                           length, cnrtMemcpyDevToHost),
                "cnrtMemcpy device-to-host")) {
            return false;
        }
        bool match = true;
        size_t first_mismatch = 0;
        for (size_t i = 0; i < length; ++i) {
            if (host[i] != expected) {
                match = false;
                first_mismatch = i;
                break;
            }
        }
        if (match) return true;
        LOG(INFO) << "Region not settled yet (byte " << first_mismatch
                  << " expected " << static_cast<int>(expected) << ", got "
                  << static_cast<int>(host[first_mismatch]) << "), retrying";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    LOG(ERROR) << "Timed out verifying device region at offset " << offset;
    return false;
}

bool verifyFullImage(void* device_ptr, int device,
                     const std::vector<uint8_t>& expected) {
    if (!checkCnrt(cnrtSetDevice(device), "cnrtSetDevice")) return false;
    std::vector<uint8_t> host(expected.size());
    if (!checkCnrt(cnrtMemcpy(host.data(), device_ptr, host.size(),
                              cnrtMemcpyDevToHost),
                   "cnrtMemcpy device-to-host")) {
        return false;
    }
    for (size_t i = 0; i < host.size(); ++i) {
        if (host[i] != expected[i]) {
            LOG(ERROR) << "Full image mismatch at byte " << i << ": expected "
                       << static_cast<int>(expected[i]) << ", got "
                       << static_cast<int>(host[i]);
            return false;
        }
    }
    return true;
}

void fillRegion(std::vector<uint8_t>* image, size_t offset, size_t length,
                uint8_t value) {
    std::memset(image->data() + offset, value, length);
}

// --- rank body -------------------------------------------------------------

int runRank(int rank, const std::string& workdir) {
    if (!checkCnrt(cnrtSetDevice(rank), "cnrtSetDevice")) return 1;
    void* buffer = nullptr;
    if (!checkCnrt(cnrtMalloc(&buffer, kBufferBytes), "cnrtMalloc") ||
        !checkCnrt(cnrtMemset(buffer, 0, kBufferBytes), "cnrtMemset")) {
        return 1;
    }
    void* scratch = nullptr;  // unregistered-source probe
    if (!checkCnrt(cnrtMalloc(&scratch, kTransferBytes), "cnrtMalloc") ||
        !checkCnrt(cnrtMemset(scratch, 0, kTransferBytes), "cnrtMemset")) {
        return 1;
    }

    auto engine = std::make_unique<TransferEngine>(false);
    if (engine->init(P2PHANDSHAKE, "127.0.0.1:0", "127.0.0.1", 0) != 0 ||
        engine->installTransport("cncl", nullptr) == nullptr ||
        engine->registerLocalMemory(buffer, kBufferBytes,
                                    "mlu:" + std::to_string(rank)) != 0) {
        LOG(ERROR) << "Failed to initialize CNCL TE rank " << rank;
        return 1;
    }
    const std::string name = engine->getLocalIpAndPort();
    if (!writeFile(filePath(workdir, "name." + std::to_string(rank)), name)) {
        LOG(ERROR) << "Failed to publish engine name";
        return 1;
    }
    std::string peer_name;
    if (!waitForFile(filePath(workdir, "name." + std::to_string(1 - rank)),
                     &peer_name)) {
        return 1;
    }
    SegmentHandle peer = engine->openSegment(peer_name);
    if (peer == static_cast<SegmentHandle>(-1)) {
        LOG(ERROR) << "Failed to open peer segment";
        return 1;
    }
    // The remote target address is the peer's device buffer address, taken
    // from the segment metadata exchanged over the handshake.
    auto peer_desc = engine->getMetadata()->getSegmentDescByID(peer);
    if (!peer_desc || peer_desc->buffers.size() != 1 ||
        peer_desc->buffers[0].length != kBufferBytes ||
        peer_desc->buffers[0].device_id != (1 - rank)) {
        LOG(ERROR) << "Peer segment metadata is unexpected";
        return 1;
    }
    const uint64_t peer_buffer = peer_desc->buffers[0].addr;

    // Each rank tracks what its own buffer should contain: the regions it
    // staged locally as send sources plus the regions it received.
    std::vector<uint8_t> local_image(kBufferBytes, 0);

    if (rank == 0) {
        // (1) READ must be rejected without submitting any CNCL op.
        if (!expectSubmitRejected(engine.get(), peer, buffer, peer_buffer,
                                  TransferRequest::READ, kTransferBytes,
                                  /*expect_not_supported=*/true)) {
            return 1;
        }
        LOG(INFO) << "CNCL READ rejection validation passed";
    }
    if (!barrier(workdir, rank, "step1")) return 1;

    if (rank == 0) {
        // (2) WRITE at a non-zero offset; peers must see untouched guard
        // bytes on both sides of the landing zone.
        if (!checkCnrt(cnrtMemset(buffer, 0x5a, kTransferBytes),
                       "cnrtMemset write source") ||
            !submitWrite(engine.get(), peer, buffer, peer_buffer + kOffsetBytes,
                         kTransferBytes)) {
            return 1;
        }
        fillRegion(&local_image, 0, kTransferBytes, 0x5a);
    } else {
        if (!verifyRegion(buffer, rank, kOffsetBytes, kTransferBytes, 0x5a) ||
            !verifyRegion(buffer, rank, 0, kOffsetBytes, 0) ||
            !verifyRegion(buffer, rank, kOffsetBytes + kTransferBytes, 4096,
                          0)) {
            return 1;
        }
        LOG(INFO) << "CNCL offset WRITE validation passed";
        fillRegion(&local_image, kOffsetBytes, kTransferBytes, 0x5a);
    }
    if (!barrier(workdir, rank, "step2")) return 1;

    if (rank == 0) {
        // (3) Four concurrent writers targeting disjoint regions with
        // distinct patterns: a send/recv mismatch would land the wrong
        // pattern in the wrong region.
        const uint8_t patterns[4] = {0x10, 0x20, 0x30, 0x40};
        std::atomic<bool> writes_ok{true};
        std::vector<std::thread> writers;
        for (int i = 0; i < 4; ++i) {
            const size_t offset = kConcurrentOffset + i * kTransferBytes;
            writers.emplace_back([&, i, offset] {
                std::vector<uint8_t> source(kTransferBytes, patterns[i]);
                if (!checkCnrt(cnrtMemcpy(static_cast<char*>(buffer) + offset,
                                          source.data(), kTransferBytes,
                                          cnrtMemcpyHostToDev),
                               "cnrtMemcpy host-to-device") ||
                    !submitWrite(engine.get(), peer,
                                 static_cast<char*>(buffer) + offset,
                                 peer_buffer + offset, kTransferBytes)) {
                    writes_ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& writer : writers) writer.join();
        if (!writes_ok.load(std::memory_order_relaxed)) return 1;
        for (int i = 0; i < 4; ++i) {
            fillRegion(&local_image, kConcurrentOffset + i * kTransferBytes,
                       kTransferBytes, patterns[i]);
        }
    } else {
        const uint8_t patterns[4] = {0x10, 0x20, 0x30, 0x40};
        for (int i = 0; i < 4; ++i) {
            if (!verifyRegion(buffer, rank,
                              kConcurrentOffset + i * kTransferBytes,
                              kTransferBytes, patterns[i])) {
                return 1;
            }
        }
        if (!verifyRegion(buffer, rank, kConcurrentOffset + 4 * kTransferBytes,
                          4096, 0)) {
            return 1;
        }
        LOG(INFO) << "CNCL concurrent WRITE validation passed";
        for (int i = 0; i < 4; ++i) {
            fillRegion(&local_image, kConcurrentOffset + i * kTransferBytes,
                       kTransferBytes, patterns[i]);
        }
    }
    if (!barrier(workdir, rank, "step3")) return 1;

    if (rank == 1) {
        // (4) Reverse WRITE: rank 1 writes into rank 0's buffer. Note this
        // overwrites rank 0's 0x5a staging region from step (2).
        if (!checkCnrt(cnrtMemset(buffer, 0xa5, kTransferBytes),
                       "cnrtMemset reverse write source") ||
            !submitWrite(engine.get(), peer, buffer, peer_buffer,
                         kTransferBytes)) {
            return 1;
        }
        fillRegion(&local_image, 0, kTransferBytes, 0xa5);
    } else {
        if (!verifyRegion(buffer, rank, 0, kTransferBytes, 0xa5)) {
            return 1;
        }
        LOG(INFO) << "CNCL reverse WRITE validation passed";
        fillRegion(&local_image, 0, kTransferBytes, 0xa5);
    }
    if (!barrier(workdir, rank, "step4")) return 1;

    {
        // (5) Bidirectional concurrent writes over one session.
        const uint8_t patterns[2] = {0x3c, 0xc3};
        std::atomic<bool> writes_ok{true};
        std::vector<std::thread> writers;
        for (int i = 0; i < 2; ++i) {
            const size_t offset = kBidirectionalOffset + i * kTransferBytes;
            writers.emplace_back([&, i, offset] {
                std::vector<uint8_t> source(kTransferBytes, patterns[i]);
                if (!checkCnrt(cnrtMemcpy(static_cast<char*>(buffer) + offset,
                                          source.data(), kTransferBytes,
                                          cnrtMemcpyHostToDev),
                               "cnrtMemcpy host-to-device") ||
                    !submitWrite(engine.get(), peer,
                                 static_cast<char*>(buffer) + offset,
                                 peer_buffer + offset, kTransferBytes)) {
                    writes_ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& writer : writers) writer.join();
        if (!writes_ok.load(std::memory_order_relaxed)) return 1;
        // Both the locally staged sources and the received regions carry the
        // same patterns at the same offsets.
        for (int i = 0; i < 2; ++i) {
            fillRegion(&local_image, kBidirectionalOffset + i * kTransferBytes,
                       kTransferBytes, patterns[i]);
        }
    }
    if (!barrier(workdir, rank, "step5")) return 1;
    {
        const uint8_t patterns[2] = {0x3c, 0xc3};
        for (int i = 0; i < 2; ++i) {
            if (!verifyRegion(buffer, rank,
                              kBidirectionalOffset + i * kTransferBytes,
                              kTransferBytes, patterns[i])) {
                return 1;
            }
        }
        LOG(INFO) << "CNCL bidirectional WRITE validation passed";
    }

    if (rank == 0) {
        // (6) Large transfer (8 MiB) straddling most of the buffer.
        std::vector<uint8_t> source(kLargeBytes, 0x77);
        if (!checkCnrt(
                cnrtMemcpy(static_cast<char*>(buffer) + kLargeOffset,
                           source.data(), kLargeBytes, cnrtMemcpyHostToDev),
                "cnrtMemcpy host-to-device") ||
            !submitWrite(engine.get(), peer,
                         static_cast<char*>(buffer) + kLargeOffset,
                         peer_buffer + kLargeOffset, kLargeBytes)) {
            return 1;
        }
        fillRegion(&local_image, kLargeOffset, kLargeBytes, 0x77);
    } else {
        if (!verifyRegion(buffer, rank, kLargeOffset, kLargeBytes, 0x77)) {
            return 1;
        }
        LOG(INFO) << "CNCL large WRITE validation passed";
        fillRegion(&local_image, kLargeOffset, kLargeBytes, 0x77);
    }
    if (!barrier(workdir, rank, "step6")) return 1;

    if (rank == 0) {
        // (7) Unregistered source: still MLU memory, but not registered.
        if (engine->registerLocalMemory(scratch, kTransferBytes,
                                        "mlu:" + std::to_string(rank)) != 0 ||
            engine->unregisterLocalMemory(scratch) != 0) {
            LOG(ERROR) << "Failed to cycle scratch registration";
            return 1;
        }
        if (!expectSubmitRejected(engine.get(), peer, scratch, peer_buffer,
                                  TransferRequest::WRITE, kTransferBytes,
                                  false)) {
            return 1;
        }
        LOG(INFO) << "CNCL unregistered source rejection passed";

        // (8) Target outside every registered remote buffer.
        if (!expectSubmitRejected(
                engine.get(), peer, buffer, peer_buffer + kBufferBytes,
                TransferRequest::WRITE, kTransferBytes, false)) {
            return 1;
        }
        LOG(INFO) << "CNCL unregistered target rejection passed";
    }
    if (!barrier(workdir, rank, "step7")) return 1;

    // Final full-image check on both ranks.
    if (!verifyFullImage(buffer, rank, local_image)) {
        return 1;
    }
    LOG(INFO) << "CNCL full image validation passed on rank " << rank;

    if (engine->unregisterLocalMemory(buffer) != 0) {
        LOG(ERROR) << "Failed to unregister buffer";
        return 1;
    }
    engine->freeEngine();
    engine.reset();
    if (!checkCnrt(cnrtFree(buffer), "cnrtFree") ||
        !checkCnrt(cnrtFree(scratch), "cnrtFree")) {
        return 1;
    }
    LOG(INFO) << "CNCL transport example rank " << rank << " PASSED";
    return 0;
}

// --- launcher --------------------------------------------------------------

int spawnChild(int rank, const std::string& workdir) {
    const pid_t pid = fork();
    if (pid < 0) {
        PLOG(ERROR) << "fork";
        return -1;
    }
    if (pid == 0) {
        const std::string rank_arg = "--rank=" + std::to_string(rank);
        const std::string workdir_arg = "--workdir=" + workdir;
        execl("/proc/self/exe", "cncl_transport_example", rank_arg.c_str(),
              workdir_arg.c_str(), nullptr);
        PLOG(FATAL) << "execl";
    }
    return pid;
}

int runLauncher() {
    char template_dir[] = "/tmp/cncl_example_XXXXXX";
    const char* workdir = mkdtemp(template_dir);
    if (!workdir) {
        PLOG(ERROR) << "mkdtemp";
        return 1;
    }
    const pid_t rank0 = spawnChild(0, workdir);
    const pid_t rank1 = spawnChild(1, workdir);
    if (rank0 < 0 || rank1 < 0) {
        if (rank0 > 0) kill(rank0, SIGKILL);
        if (rank1 > 0) kill(rank1, SIGKILL);
        return 1;
    }
    int failed = 0;
    for (int i = 0; i < 2; ++i) {
        int status = 0;
        const pid_t done = wait(&status);
        if (done != rank0 && done != rank1) {
            LOG(ERROR) << "wait returned unexpected pid " << done;
            failed = 1;
            continue;
        }
        if (WIFSIGNALED(status)) {
            LOG(ERROR) << "child died from signal " << WTERMSIG(status);
            failed = 1;
        } else if (WEXITSTATUS(status) != 0) {
            LOG(ERROR) << "child exited with " << WEXITSTATUS(status);
            failed = 1;
        }
    }
    if (failed == 0) {
        LOG(INFO) << "CNCL transport example PASSED";
    } else {
        LOG(ERROR) << "CNCL transport example FAILED";
    }
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    unsigned int device_count = 0;
    if (!checkCnrt(cnrtGetDeviceCount(&device_count), "cnrtGetDeviceCount") ||
        device_count < 2) {
        LOG(ERROR) << "The CNCL example requires two MLU devices";
        return 1;
    }

    std::string workdir;
    int rank = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--rank=", 0) == 0) {
            rank = std::stoi(arg.substr(7));
        } else if (arg.rfind("--workdir=", 0) == 0) {
            workdir = arg.substr(10);
        } else {
            LOG(ERROR) << "Unknown argument " << arg;
            return 1;
        }
    }

    if (rank < 0) return runLauncher();
    if (rank > 1 || workdir.empty()) {
        LOG(ERROR) << "Invalid --rank/--workdir combination";
        return 1;
    }
    return runRank(rank, workdir);
}
