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

// CNCL notifier-exhaustion stress repro.
//
// Self-contained reproduction of the production incident in which the CNCL
// transport's per-descriptor cnrtNotifier_t allocation exhausts the device
// notifier pool (cnrtNotifierCreate -> CN_OPS_ERROR_OUT_OF_RESOURCES, 100053)
// and the engine then goes silent. Only this repository and the Neuware
// toolkit are needed; no vLLM, no second machine.
//
// Shape of the repro (mirrors the vLLM PD-disaggregation load):
//   - rank 0 (sender) runs --threads worker threads; every thread loops
//     allocateBatchID(N) -> submitTransfer(N WRITE requests) -> busy-poll
//     getBatchTransferStatus -> freeBatchID, exactly like
//     TransferEnginePy::batchTransferSync.
//   - rank 1 (receiver) only runs its engine; the transport's handshake
//     daemon enqueues the matching cnclRecv per request.
//   - N defaults to 2048 (~2.1-2.5k descriptors per production transfer) and
//     the thread count to 128 (production num_workers).
//
// The binary must run each endpoint in its own process (CNCL refuses to
// initialize the same clique id twice within a process), so run without
// --rank it re-executes itself twice with --rank 0 / --rank 1, redirecting
// each child's log to <workdir>/rank<N>.log.
//
// Phases:
//   A) integrity lockstep: a few writer threads submit batches whose pattern
//      is unique per round; the receiver reads each region back over D2H as
//      soon as the writer reports COMPLETED (file handshake), tail slice
//      first, so a completion signal that fires before the data has landed
//      is caught. Failure here means the completion semantics are broken
//      (silent-corruption risk) and the run aborts before the stress phase.
//   B) stress: full-concurrency loop for --duration seconds, then one final
//      "gold epoch" batch per thread with a reserved pattern. The receiver
//      afterwards verifies every byte of every thread's region against the
//      gold pattern: any lost, duplicated or misrouted send shows up.
//
// Exit codes (rank 0 unless noted): 0 clean pass; 1 setup failure; 2 rank 1
// timed out waiting for rank 0 (wedge signature); 3 engine wedge detected by
// the watchdog; 4 phase-A integrity failure; 5 stress-phase batch failures;
// 6 final verification mismatch. The launcher aggregates and reports.
//
// --kill-peer-at=S: fault-injection mode. S seconds after phase B starts the
// sender SIGKILLs the receiver mid-stream, so every in-flight and subsequent
// submit fails (handshake RPC unreachable, session quarantined) and each
// failed batch is freed immediately. Expected outcome: rank 0 exits 5, the
// launcher reports rank1=137, and rank0.log contains no "freeBatchID failed"
// line and no wedge exit (2/3) — the abortBatch hook settles the abandoned
// groups so the frees succeed and the engine stays responsive. S must be
// smaller than --duration; final verification is skipped (the peer is dead).

#include <cnrt.h>
#include <glog/logging.h>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "transfer_engine.h"

namespace {

using mooncake::BatchID;
using mooncake::getCurrentTimeInNano;
using mooncake::SegmentHandle;
using mooncake::TransferEngine;
using mooncake::TransferRequest;
using mooncake::TransferStatus;
using mooncake::TransferStatusEnum;

// ---------------------------------------------------------------------------
// Flags (simple --key=value parsing; everything has a default).
// ---------------------------------------------------------------------------

struct Flags {
    int rank = -1;
    std::string workdir;
    int device0 = 0;  // sender device
    int device1 = 1;  // receiver device

    // Phase B (stress) shape: defaults mirror the production incident
    // (num_workers=128, ~2.1-2.5k descriptors per transfer).
    int threads = 128;
    int slices = 2048;
    size_t slice_bytes = 64 << 10;
    int duration_sec = 300;
    int variants = 4;          // distinct source patterns cycled per iteration
    int batch_timeout_sec = 180;  // per-batch submit+poll budget
    int stall_warn_sec = 60;      // per-thread no-progress warning
    int wedge_window_sec = 120;   // global no-resolution window => wedge
    int kill_peer_at_sec = -1;    // SIGKILL the receiver this far into phase B

    // Phase A (integrity lockstep).
    int phase_a_rounds = 12;
    int phase_a_threads = 2;
    int phase_a_slices = 64;
    size_t phase_a_slice_bytes = 8 << 20;
};

bool parseFlags(int argc, char** argv, Flags* f) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto eq = arg.find('=');
        if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
            LOG(ERROR) << "Unknown argument " << arg;
            return false;
        }
        const std::string key = arg.substr(2, eq - 2);
        const std::string val = arg.substr(eq + 1);
        auto num = [&]() { return std::stoll(val); };
        if (key == "rank") f->rank = std::stoi(val);
        else if (key == "workdir") f->workdir = val;
        else if (key == "device0") f->device0 = std::stoi(val);
        else if (key == "device1") f->device1 = std::stoi(val);
        else if (key == "threads") f->threads = std::stoi(val);
        else if (key == "slices") f->slices = std::stoi(val);
        else if (key == "slice-bytes") f->slice_bytes = num();
        else if (key == "duration") f->duration_sec = std::stoi(val);
        else if (key == "variants") f->variants = std::stoi(val);
        else if (key == "batch-timeout") f->batch_timeout_sec = std::stoi(val);
        else if (key == "stall-warn") f->stall_warn_sec = std::stoi(val);
        else if (key == "wedge-window") f->wedge_window_sec = std::stoi(val);
        else if (key == "kill-peer-at") f->kill_peer_at_sec = std::stoi(val);
        else if (key == "phase-a-rounds") f->phase_a_rounds = std::stoi(val);
        else if (key == "phase-a-threads") f->phase_a_threads = std::stoi(val);
        else if (key == "phase-a-slices") f->phase_a_slices = std::stoi(val);
        else if (key == "phase-a-slice-bytes") f->phase_a_slice_bytes = num();
        else {
            LOG(ERROR) << "Unknown flag --" << key;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pattern: deterministic per (tag, slice, byte). Both ranks compute it
// independently; the receiver never needs the sender's data to verify.
// ---------------------------------------------------------------------------

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

constexpr uint64_t kSliceSalt = 0xD1B54A32D192ED03ULL;

// 8-byte word `word` of slice `slice` for pattern tag `tag`.
uint64_t patternWord(uint64_t tag, uint64_t slice, uint64_t word) {
    return splitmix64(tag ^ slice * kSliceSalt ^ word * 0x2545F4914F6CDD1DULL);
}

void fillPattern(uint8_t* dst, uint64_t tag, uint64_t slice, size_t bytes) {
    auto* out = reinterpret_cast<uint64_t*>(dst);
    const size_t words = bytes / sizeof(uint64_t);
    for (size_t w = 0; w < words; ++w) out[w] = patternWord(tag, slice, w);
    for (size_t b = words * sizeof(uint64_t); b < bytes; ++b) {
        dst[b] = static_cast<uint8_t>(patternWord(tag, slice, b / 8) >>
                                      ((b % 8) * 8));
    }
}

// Returns the first mismatching byte index, or bytes when fully equal.
size_t comparePattern(const uint8_t* got, uint64_t tag, uint64_t slice,
                      size_t bytes) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(got);
    const size_t nwords = bytes / sizeof(uint64_t);
    for (size_t w = 0; w < nwords; ++w) {
        const uint64_t want = patternWord(tag, slice, w);
        if (words[w] != want) {
            for (size_t b = w * 8; b < w * 8 + 8; ++b) {
                if (got[b] != static_cast<uint8_t>(want >> ((b % 8) * 8))) {
                    return b;
                }
            }
            return w * 8;
        }
    }
    for (size_t b = nwords * 8; b < bytes; ++b) {
        const uint8_t want =
            static_cast<uint8_t>(patternWord(tag, slice, b / 8) >> ((b % 8) * 8));
        if (got[b] != want) return b;
    }
    return bytes;
}

// Pattern tags.
uint64_t variantTag(int variant) { return 0x1000ULL + variant; }
constexpr uint64_t kGoldTag = 0x600DULL;
uint64_t phaseATag(int thread, int round) {
    return 0xA000ULL + (uint64_t)thread * 1000 + round;
}

// ---------------------------------------------------------------------------
// Workdir helpers (same rendezvous scheme as cncl_transport_example).
// ---------------------------------------------------------------------------

std::string filePath(const std::string& workdir, const std::string& name) {
    return workdir + "/" + name;
}

bool writeFile(const std::string& path, const std::string& content) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
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

bool waitForFile(const std::string& path, std::string* content,
                 int timeout_sec) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        if (readFile(path, content)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    LOG(ERROR) << "Timed out waiting for " << path;
    return false;
}

bool fileExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool barrier(const std::string& workdir, int rank, const std::string& step,
             int timeout_sec = 600) {
    if (!writeFile(filePath(workdir, step + "." + std::to_string(rank)),
                   "ok")) {
        LOG(ERROR) << "Failed to write barrier file for " << step;
        return false;
    }
    std::string ignored;
    return waitForFile(
        filePath(workdir, step + "." + std::to_string(1 - rank)), &ignored,
        timeout_sec);
}

// ---------------------------------------------------------------------------
// CNRT helpers.
// ---------------------------------------------------------------------------

bool checkCnrt(cnrtRet_t result, const char* operation) {
    if (result == cnrtSuccess) return true;
    LOG(ERROR) << operation << " failed: " << cnrtGetErrorStr(result);
    return false;
}

bool copyToDevice(void* dst, const void* src, size_t bytes) {
    return checkCnrt(cnrtMemcpy(dst, const_cast<void*>(src), bytes,
                                cnrtMemcpyHostToDev),
                     "cnrtMemcpy host-to-device");
}

bool copyToHost(void* dst, const void* src, size_t bytes) {
    return checkCnrt(cnrtMemcpy(dst, const_cast<void*>(src), bytes,
                                cnrtMemcpyDevToHost),
                     "cnrtMemcpy device-to-host");
}

// ---------------------------------------------------------------------------
// Engine setup shared by both ranks. Returns nullptr on failure.
// ---------------------------------------------------------------------------

struct EngineCtx {
    std::unique_ptr<TransferEngine> engine;
    void* buffer = nullptr;
    size_t buffer_bytes = 0;
    SegmentHandle peer = 0;
    uint64_t peer_buffer = 0;
};

std::unique_ptr<EngineCtx> setupEngine(int rank, int device,
                                       size_t buffer_bytes,
                                       const std::string& workdir) {
    auto ctx = std::make_unique<EngineCtx>();
    ctx->buffer_bytes = buffer_bytes;
    if (!checkCnrt(cnrtSetDevice(device), "cnrtSetDevice") ||
        !checkCnrt(cnrtMalloc(&ctx->buffer, buffer_bytes), "cnrtMalloc") ||
        !checkCnrt(cnrtMemset(ctx->buffer, 0, buffer_bytes), "cnrtMemset")) {
        return nullptr;
    }

    ctx->engine = std::make_unique<TransferEngine>(false);
    if (ctx->engine->init(P2PHANDSHAKE, "127.0.0.1:0", "127.0.0.1", 0) != 0 ||
        ctx->engine->installTransport("cncl", nullptr) == nullptr ||
        ctx->engine->registerLocalMemory(ctx->buffer, buffer_bytes,
                                         "mlu:" + std::to_string(device)) !=
            0) {
        LOG(ERROR) << "Failed to initialize CNCL engine on rank " << rank;
        return nullptr;
    }

    const std::string name = ctx->engine->getLocalIpAndPort();
    if (!writeFile(filePath(workdir, "name." + std::to_string(rank)), name)) {
        LOG(ERROR) << "Failed to publish engine name";
        return nullptr;
    }
    std::string peer_name;
    if (!waitForFile(filePath(workdir, "name." + std::to_string(1 - rank)),
                     &peer_name, 120)) {
        return nullptr;
    }
    ctx->peer = ctx->engine->openSegment(peer_name);
    if (ctx->peer == static_cast<SegmentHandle>(-1)) {
        LOG(ERROR) << "Failed to open peer segment " << peer_name;
        return nullptr;
    }
    auto peer_desc = ctx->engine->getMetadata()->getSegmentDescByID(ctx->peer);
    if (!peer_desc || peer_desc->buffers.size() != 1 ||
        peer_desc->buffers[0].length != buffer_bytes ||
        peer_desc->buffers[0].device_id != (rank == 0 ? 1 : 0)) {
        LOG(ERROR) << "Peer segment metadata is unexpected";
        return nullptr;
    }
    ctx->peer_buffer = peer_desc->buffers[0].addr;
    LOG(INFO) << "[stress] rank " << rank << " engine up, peer buffer 0x"
              << std::hex << ctx->peer_buffer << std::dec << " bytes "
              << buffer_bytes;
    return ctx;
}

void teardownEngine(EngineCtx* ctx) {
    if (!ctx) return;
    if (ctx->engine) {
        if (ctx->buffer &&
            ctx->engine->unregisterLocalMemory(ctx->buffer) != 0) {
            LOG(ERROR) << "Failed to unregister buffer";
        }
        ctx->engine->freeEngine();
        ctx->engine.reset();
    }
    if (ctx->buffer) {
        if (!checkCnrt(cnrtFree(ctx->buffer), "cnrtFree")) {
            // Keep going; process is about to exit anyway.
        }
        ctx->buffer = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Transfer driver: one batch of `slices` WRITE requests, vLLM-shaped.
// ---------------------------------------------------------------------------

enum class BatchResult { kOk, kSubmitFailed, kFailed, kTimeout };

BatchResult runBatch(TransferEngine* engine, SegmentHandle peer,
                     const std::vector<TransferRequest>& requests,
                     int timeout_sec, std::string* detail) {
    BatchID batch = engine->allocateBatchID(requests.size());
    if (batch == mooncake::INVALID_BATCH_ID) {
        *detail = "allocateBatchID failed";
        return BatchResult::kSubmitFailed;
    }
    auto result = engine->submitTransfer(batch, requests);
    if (!result.ok()) {
        *detail = "submitTransfer: " + result.ToString();
        auto free_status = engine->freeBatchID(batch);
        if (!free_status.ok()) {
            LOG(ERROR) << "[stress] freeBatchID after submit failure: "
                       << free_status.ToString();
        }
        return BatchResult::kSubmitFailed;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    BatchResult outcome = BatchResult::kOk;
    // Busy poll with no sleep, mirroring batchTransferSync's wait loop.
    for (;;) {
        TransferStatus status{};
        auto status_result = engine->getBatchTransferStatus(batch, status);
        if (!status_result.ok()) {
            *detail = "getBatchTransferStatus: " + status_result.ToString();
            outcome = BatchResult::kFailed;
            break;
        }
        if (status.s == TransferStatusEnum::COMPLETED) break;
        if (status.s == TransferStatusEnum::FAILED ||
            status.s == TransferStatusEnum::TIMEOUT) {
            *detail = "batch reached status " + std::to_string(status.s);
            outcome = BatchResult::kFailed;
            break;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            *detail = "batch did not complete within " +
                      std::to_string(timeout_sec) + "s";
            outcome = BatchResult::kTimeout;
            break;
        }
    }
    auto free_status = engine->freeBatchID(batch);
    if (!free_status.ok()) {
        // BatchBusy here is the known abandoned-batch leak path.
        LOG(ERROR) << "[stress] freeBatchID failed: "
                   << free_status.ToString();
    }
    return outcome;
}

std::vector<TransferRequest> buildRequests(uint64_t src_base,
                                           uint64_t dst_base, int slices,
                                           size_t slice_bytes,
                                           SegmentHandle peer) {
    std::vector<TransferRequest> requests(slices);
    for (int i = 0; i < slices; ++i) {
        requests[i].opcode = TransferRequest::WRITE;
        requests[i].source =
            reinterpret_cast<void*>(src_base + (size_t)i * slice_bytes);
        requests[i].target_id = peer;
        requests[i].target_offset = dst_base + (size_t)i * slice_bytes;
        requests[i].length = slice_bytes;
    }
    return requests;
}

// ---------------------------------------------------------------------------
// Layout: rank 0 owns the source buffer, rank 1 the target buffer.
//
//   [ phase-B variant 0 .. V-1 | gold variant | phase-A per-thread sources ]
//   [ phase-B per-thread targets | phase-A per-thread targets             ]
// ---------------------------------------------------------------------------

struct Layout {
    size_t phase_b_region;     // per-thread target bytes
    size_t variant_base;       // offset of variant v in source buffer
    size_t gold_base;          // offset of the gold source pattern
    size_t phase_a_src_base;   // offset of thread t's phase-A source
    size_t phase_a_dst_base;   // offset of thread t's phase-A target
    size_t total_bytes;
};

Layout computeLayout(const Flags& f) {
    Layout l{};
    const size_t batch_bytes = (size_t)f.slices * f.slice_bytes;
    l.phase_b_region = batch_bytes;
    l.variant_base = 0;
    l.gold_base = (size_t)f.variants * batch_bytes;
    l.phase_a_src_base = (size_t)(f.variants + 1) * batch_bytes;
    l.phase_a_dst_base = (size_t)f.threads * batch_bytes;
    const size_t phase_a_region =
        (size_t)f.phase_a_threads * f.phase_a_slices * f.phase_a_slice_bytes;
    const size_t src_bytes = l.phase_a_src_base + phase_a_region;
    const size_t dst_bytes = l.phase_a_dst_base + phase_a_region;
    l.total_bytes = std::max(src_bytes, dst_bytes);
    return l;
}

// ---------------------------------------------------------------------------
// Rank 0: phase A (integrity lockstep).
// ---------------------------------------------------------------------------

bool phaseAWriter(const Flags& f, const Layout& l, EngineCtx* ctx, int t,
                  std::string* error) {
    const size_t region_bytes =
        (size_t)f.phase_a_slices * f.phase_a_slice_bytes;
    const uint64_t src =
        (uint64_t)(uintptr_t)ctx->buffer + l.phase_a_src_base +
        (size_t)t * region_bytes;
    const uint64_t dst =
        ctx->peer_buffer + l.phase_a_dst_base + (size_t)t * region_bytes;
    std::vector<uint8_t> staging(f.phase_a_slice_bytes);

    for (int round = 0; round < f.phase_a_rounds; ++round) {
        const uint64_t tag = phaseATag(t, round);
        // Stage the whole batch on device: slice i carries pattern(tag, i).
        for (int i = 0; i < f.phase_a_slices; ++i) {
            fillPattern(staging.data(), tag, i, f.phase_a_slice_bytes);
            if (!copyToDevice(
                    reinterpret_cast<void*>(src + (size_t)i * f.phase_a_slice_bytes),
                    staging.data(), f.phase_a_slice_bytes)) {
                *error = "H2D staging failed";
                return false;
            }
        }
        auto requests = buildRequests(src, dst, f.phase_a_slices,
                                      f.phase_a_slice_bytes, ctx->peer);
        std::string detail;
        const BatchResult result =
            runBatch(ctx->engine.get(), ctx->peer, requests, 300, &detail);
        if (result != BatchResult::kOk) {
            *error = "phase A batch failed: " + detail;
            return false;
        }
        // The batch reports COMPLETED; the receiver now reads the region
        // back. Any completion-before-data shows up as a mismatch.
        const std::string stem =
            "va." + std::to_string(t) + "." + std::to_string(round);
        if (!writeFile(filePath(f.workdir, stem), "go")) {
            *error = "failed to post verify request";
            return false;
        }
        std::string verdict;
        if (!waitForFile(filePath(f.workdir, "vr." + std::to_string(t) + "." +
                                              std::to_string(round)),
                         &verdict, 300)) {
            *error = "verify response timed out";
            return false;
        }
        if (verdict != "ok") {
            *error = "receiver verification: " + verdict;
            return false;
        }
        LOG(INFO) << "[stress] phase A thread " << t << " round " << round
                  << " verified";
    }
    return true;
}

// Rank 1 side of phase A: watch for va.t.r files, read the region back tail
// slice first (the tail lands last under FIFO, so it is the most sensitive
// to a completion signal that fires early), and answer vr.t.r.
void phaseAVerifier(const Flags& f, const Layout& l, EngineCtx* ctx,
                    std::atomic<bool>* stop, std::atomic<int>* mismatches) {
    const size_t region_bytes =
        (size_t)f.phase_a_slices * f.phase_a_slice_bytes;
    std::vector<uint8_t> host(f.phase_a_slice_bytes);
    std::vector<std::vector<bool>> done(
        f.phase_a_threads,
        std::vector<bool>(f.phase_a_rounds, false));
    size_t remaining = (size_t)f.phase_a_threads * f.phase_a_rounds;

    while (!stop->load(std::memory_order_relaxed) && remaining > 0) {
        bool progressed = false;
        for (int t = 0; t < f.phase_a_threads; ++t) {
            for (int r = 0; r < f.phase_a_rounds; ++r) {
                if (done[t][r]) continue;
                const std::string stem =
                    std::to_string(t) + "." + std::to_string(r);
                if (!fileExists(filePath(f.workdir, "va." + stem))) continue;
                done[t][r] = true;
                --remaining;
                progressed = true;
                const uint64_t tag = phaseATag(t, r);
                const uint64_t base =
                    (uint64_t)(uintptr_t)ctx->buffer + l.phase_a_dst_base +
                    (size_t)t * region_bytes;
                std::string verdict = "ok";
                for (int i = f.phase_a_slices - 1; i >= 0; --i) {
                    if (!copyToHost(
                            host.data(),
                            reinterpret_cast<const void*>(
                                base + (size_t)i * f.phase_a_slice_bytes),
                            f.phase_a_slice_bytes)) {
                        verdict = "D2H copy failed";
                        break;
                    }
                    const size_t bad = comparePattern(
                        host.data(), tag, i, f.phase_a_slice_bytes);
                    if (bad != f.phase_a_slice_bytes) {
                        std::ostringstream out;
                        out << "MISMATCH thread " << t << " round " << r
                            << " slice " << i << " byte " << bad << " got 0x"
                            << std::hex << std::setw(2) << std::setfill('0')
                            << (unsigned)host[bad] << " want 0x" << std::setw(2)
                            << (unsigned)(patternWord(tag, i, bad / 8) >>
                                          ((bad % 8) * 8) & 0xff);
                        verdict = out.str();
                        break;
                    }
                }
                if (verdict != "ok") {
                    LOG(ERROR) << "[stress] phase A verify: " << verdict;
                    mismatches->fetch_add(1, std::memory_order_relaxed);
                }
                if (!writeFile(filePath(f.workdir, "vr." + stem), verdict)) {
                    LOG(ERROR) << "[stress] failed to answer verify " << stem;
                }
            }
        }
        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ---------------------------------------------------------------------------
// Rank 0: phase B (stress).
// ---------------------------------------------------------------------------

struct WorkerState {
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> submit_failed{0};
    std::atomic<int64_t> failed{0};
    std::atomic<int64_t> timed_out{0};
    std::atomic<int64_t> last_done_ns{0};
    std::atomic<int> state{0};  // 0 idle, 1 submitting, 2 polling, 3 gold
    std::atomic<bool> gold_ok{false};
};

const char* stateName(int state) {
    switch (state) {
        case 1: return "submitting";
        case 2: return "polling";
        case 3: return "gold";
        default: return "idle";
    }
}

void stressWorker(const Flags& f, const Layout& l, EngineCtx* ctx, int t,
                  WorkerState* state,
                  const std::chrono::steady_clock::time_point& end_time) {
    const uint64_t src_base = (uint64_t)(uintptr_t)ctx->buffer;
    const uint64_t dst_base = ctx->peer_buffer + (size_t)t * l.phase_b_region;
    state->last_done_ns.store(getCurrentTimeInNano(),
                              std::memory_order_relaxed);
    int64_t iter = 0;
    while (std::chrono::steady_clock::now() < end_time) {
        const uint64_t src =
            src_base + (size_t)(iter % f.variants) * l.phase_b_region;
        auto requests = buildRequests(src, dst_base, f.slices, f.slice_bytes,
                                      ctx->peer);
        state->state.store(1, std::memory_order_relaxed);
        std::string detail;
        const BatchResult result =
            runBatch(ctx->engine.get(), ctx->peer, requests,
                     f.batch_timeout_sec, &detail);
        state->state.store(0, std::memory_order_relaxed);
        switch (result) {
            case BatchResult::kOk:
                state->done.fetch_add(1, std::memory_order_relaxed);
                break;
            case BatchResult::kSubmitFailed:
                state->submit_failed.fetch_add(1, std::memory_order_relaxed);
                LOG(ERROR) << "[stress] thread " << t
                           << " submit failed: " << detail;
                break;
            case BatchResult::kFailed:
                state->failed.fetch_add(1, std::memory_order_relaxed);
                LOG(ERROR) << "[stress] thread " << t
                           << " batch failed: " << detail;
                break;
            case BatchResult::kTimeout:
                state->timed_out.fetch_add(1, std::memory_order_relaxed);
                LOG(ERROR) << "[stress] thread " << t
                           << " batch timeout: " << detail;
                break;
        }
        state->last_done_ns.store(getCurrentTimeInNano(),
                                  std::memory_order_relaxed);
        ++iter;
    }

    // Gold epoch: one final batch with a pattern used nowhere else. The
    // receiver verifies the whole region against it afterwards.
    state->state.store(3, std::memory_order_relaxed);
    auto requests = buildRequests(src_base + l.gold_base, dst_base, f.slices,
                                  f.slice_bytes, ctx->peer);
    std::string detail;
    const BatchResult result = runBatch(ctx->engine.get(), ctx->peer, requests,
                                        f.batch_timeout_sec, &detail);
    if (result == BatchResult::kOk) {
        state->gold_ok.store(true, std::memory_order_relaxed);
    } else {
        LOG(ERROR) << "[stress] thread " << t << " gold batch failed: "
                   << detail;
    }
    state->state.store(0, std::memory_order_relaxed);
    state->last_done_ns.store(getCurrentTimeInNano(),
                              std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Rank 0 entry.
// ---------------------------------------------------------------------------

int runSender(const Flags& f, const Layout& l) {
    auto ctx = setupEngine(0, f.device0, l.total_bytes, f.workdir);
    if (!ctx) return 1;

    // Pre-stage phase-B source variants and the gold pattern on device.
    {
        std::vector<uint8_t> staging(f.slice_bytes);
        const uint64_t src_base = (uint64_t)(uintptr_t)ctx->buffer;
        for (int v = 0; v < f.variants; ++v) {
            for (int i = 0; i < f.slices; ++i) {
                fillPattern(staging.data(), variantTag(v), i, f.slice_bytes);
                if (!copyToDevice(
                        reinterpret_cast<void*>(
                            src_base + (size_t)v * l.phase_b_region +
                            (size_t)i * f.slice_bytes),
                        staging.data(), f.slice_bytes)) {
                    return 1;
                }
            }
            LOG(INFO) << "[stress] variant " << v << " staged";
        }
        for (int i = 0; i < f.slices; ++i) {
            fillPattern(staging.data(), kGoldTag, i, f.slice_bytes);
            if (!copyToDevice(
                    reinterpret_cast<void*>(src_base + l.gold_base +
                                          (size_t)i * f.slice_bytes),
                    staging.data(), f.slice_bytes)) {
                return 1;
            }
        }
        LOG(INFO) << "[stress] gold variant staged";
    }

    // ---- Phase A ----
    if (f.phase_a_rounds > 0) {
        std::vector<std::thread> writers;
        std::vector<std::string> errors(f.phase_a_threads);
        std::atomic<int> bad{0};
        for (int t = 0; t < f.phase_a_threads; ++t) {
            writers.emplace_back([&, t] {
                if (!phaseAWriter(f, l, ctx.get(), t, &errors[t])) {
                    bad.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& w : writers) w.join();
        // Tell the receiver no more verify requests are coming, then sync.
        writeFile(filePath(f.workdir, "stopVerify"), "ok");
        if (bad.load() != 0) {
            LOG(ERROR) << "[stress] phase A FAILED: " << errors[0];
            writeFile(filePath(f.workdir, "stressDone.0"), "aborted");
            barrier(f.workdir, 0, "teardown", 120);
            teardownEngine(ctx.get());
            return 4;
        }
        LOG(INFO) << "[stress] phase A PASSED (" << f.phase_a_threads
                  << " threads x " << f.phase_a_rounds << " rounds x "
                  << f.phase_a_slices << " slices x "
                  << f.phase_a_slice_bytes << " B)";
        if (!barrier(f.workdir, 0, "phaseA", 600)) return 1;
    } else {
        writeFile(filePath(f.workdir, "stopVerify"), "ok");
        if (!barrier(f.workdir, 0, "phaseA", 600)) return 1;
    }

    // ---- Phase B ----
    LOG(INFO) << "[stress] phase B starting: " << f.threads << " threads x "
              << f.slices << " slices x " << f.slice_bytes << " B for "
              << f.duration_sec << "s";
    if (f.kill_peer_at_sec >= f.duration_sec && f.kill_peer_at_sec >= 0) {
        LOG(ERROR) << "[stress] --kill-peer-at must be smaller than --duration";
        return 1;
    }
    std::vector<std::unique_ptr<WorkerState>> states(f.threads);
    for (auto& s : states) s = std::make_unique<WorkerState>();
    std::atomic<bool> monitor_stop{false};
    const auto start = std::chrono::steady_clock::now();
    const auto end_time = start + std::chrono::seconds(f.duration_sec);

    std::thread monitor([&] {
        int64_t last_done = 0;
        int64_t last_resolved = 0;
        auto last_progress = std::chrono::steady_clock::now();
        while (!monitor_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            int64_t total = 0, submit_failed = 0, failed = 0, timed_out = 0,
                    gold = 0;
            int stalled = 0;
            int worst_thread = -1;
            double worst_idle = 0;
            const int64_t now_ns = getCurrentTimeInNano();
            for (int t = 0; t < f.threads; ++t) {
                auto& s = states[t];
                total += s->done.load(std::memory_order_relaxed);
                submit_failed += s->submit_failed.load(std::memory_order_relaxed);
                failed += s->failed.load(std::memory_order_relaxed);
                timed_out += s->timed_out.load(std::memory_order_relaxed);
                gold += s->gold_ok.load() ? 1 : 0;
                const int state = s->state.load(std::memory_order_relaxed);
                if (state == 0) continue;
                const double idle =
                    (now_ns - s->last_done_ns.load(std::memory_order_relaxed)) /
                    1e9;
                if (idle > f.stall_warn_sec) ++stalled;
                if (idle > worst_idle) {
                    worst_idle = idle;
                    worst_thread = t;
                }
            }
            // Wedge = no batch *resolves* anywhere (workers stuck inside the
            // engine). Counting fast failures as resolutions keeps the
            // detector silent when the engine is responsive but rejecting,
            // e.g. after --kill-peer-at takes the receiver down.
            const int64_t resolved =
                total + submit_failed + failed + timed_out;
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              start)
                    .count();
            LOG(INFO) << "[stress][monitor] t=" << elapsed << "s done=" << total
                      << " (+" << (total - last_done) << "/10s)"
                      << " submit_failed=" << submit_failed
                      << " failed=" << failed << " timeout=" << timed_out
                      << " gold=" << gold << " stalled_threads=" << stalled
                      << " worst=" << worst_idle << "s";
            last_done = total;
            if (worst_idle > f.stall_warn_sec && worst_thread >= 0) {
                auto& s = states[worst_thread];
                LOG(WARNING) << "[stress][monitor] STALL thread "
                             << worst_thread << " state="
                             << stateName(s->state.load()) << " no batch for "
                             << worst_idle << "s";
            }
            if (resolved > last_resolved) {
                last_resolved = resolved;
                last_progress = std::chrono::steady_clock::now();
            } else if (std::chrono::steady_clock::now() - last_progress >
                           std::chrono::seconds(f.wedge_window_sec) &&
                       std::chrono::steady_clock::now() - start >
                           std::chrono::seconds(f.wedge_window_sec)) {
                // The incident signature: no batch completes anywhere while
                // workers are stuck inside the engine. Give the transport's
                // own stall reports a grace period to fire, then get out.
                LOG(ERROR) << "[stress][monitor] ENGINE WEDGE: no batch "
                              "completed for "
                           << f.wedge_window_sec << "s; workers stuck in "
                              "submit/poll. Dumping stall summary and exiting.";
                for (int t = 0; t < f.threads; ++t) {
                    auto& s = states[t];
                    const int st = s->state.load();
                    if (st == 0) continue;
                    const double idle =
                        (now_ns -
                         s->last_done_ns.load(std::memory_order_relaxed)) /
                        1e9;
                    LOG(ERROR) << "[stress][monitor] stuck thread " << t
                               << " state=" << stateName(st) << " idle="
                               << idle << "s done=" << s->done.load();
                }
                std::this_thread::sleep_for(std::chrono::seconds(65));
                LOG(ERROR) << "[stress][monitor] exiting with wedge code";
                _exit(3);
            }
        }
    });

    std::vector<std::thread> workers;
    workers.reserve(f.threads);
    for (int t = 0; t < f.threads; ++t) {
        workers.emplace_back([&, t] {
            stressWorker(f, l, ctx.get(), t, states[t].get(), end_time);
        });
    }

    // Fault injection: take the receiver down mid-phase-B. Every in-flight
    // and later submit then fails (handshake RPC unreachable and the session
    // quarantined fail-loud), which is exactly the path that used to leak
    // batches on freeBatchID.
    std::thread killer;
    if (f.kill_peer_at_sec >= 0) {
        killer = std::thread([&] {
            std::this_thread::sleep_for(
                std::chrono::seconds(f.kill_peer_at_sec));
            std::string pid_str;
            if (!waitForFile(filePath(f.workdir, "pid.1"), &pid_str, 60)) {
                LOG(ERROR) << "[stress] kill-peer: no receiver pid file";
                return;
            }
            const pid_t peer = std::stol(pid_str);
            LOG(ERROR) << "[stress] kill-peer: SIGKILL receiver pid " << peer
                       << " at t=" << f.kill_peer_at_sec << "s";
            kill(peer, SIGKILL);
        });
    }

    for (auto& w : workers) w.join();
    monitor_stop.store(true, std::memory_order_relaxed);
    monitor.join();
    if (killer.joinable()) killer.join();

    int64_t done = 0, submit_failed = 0, failed = 0, timed_out = 0, gold = 0;
    for (auto& s : states) {
        done += s->done.load();
        submit_failed += s->submit_failed.load();
        failed += s->failed.load();
        timed_out += s->timed_out.load();
        gold += s->gold_ok.load() ? 1 : 0;
    }
    LOG(INFO) << "[stress] phase B finished: done=" << done
              << " submit_failed=" << submit_failed << " failed=" << failed
              << " timeout=" << timed_out << " gold_ok=" << gold << "/"
              << f.threads;

    // Hand over to the receiver for the final verification.
    if (!writeFile(filePath(f.workdir, "stressDone.0"),
                   "done=" + std::to_string(done))) {
        return 1;
    }

    int rc = 0;
    if (submit_failed + failed + timed_out > 0) {
        rc = 5;
    } else if (gold != f.threads) {
        rc = 5;
    }

    if (f.kill_peer_at_sec >= 0) {
        // The receiver is dead: final verification is impossible, the
        // teardown barrier would never be answered, and engine teardown
        // would hang in cnclFreeComm on the dead peer. The assertions for
        // this mode live in the log: no "freeBatchID failed", no wedge.
        LOG(INFO) << "[stress] kill-peer mode: skipping final verification "
                     "and teardown, rc="
                  << rc;
        _exit(rc);
    }

    std::string verdict;
    if (!waitForFile(filePath(f.workdir, "verifyDone.1"), &verdict, 1800)) {
        return 1;
    }
    if (!barrier(f.workdir, 0, "teardown", 120)) return 1;
    teardownEngine(ctx.get());

    if (verdict != "ok") {
        LOG(ERROR) << "[stress] final verification FAILED: " << verdict;
        rc = 6;
    }
    LOG(INFO) << "[stress] rank 0 done, rc=" << rc;
    return rc;
}

// ---------------------------------------------------------------------------
// Rank 1 entry.
// ---------------------------------------------------------------------------

int runReceiver(const Flags& f, const Layout& l) {
    // Publish our pid first: --kill-peer-at makes the sender SIGKILL us.
    if (!writeFile(filePath(f.workdir, "pid.1"), std::to_string(getpid()))) {
        LOG(ERROR) << "Failed to publish receiver pid";
        return 1;
    }
    auto ctx = setupEngine(1, f.device1, l.total_bytes, f.workdir);
    if (!ctx) return 1;

    std::atomic<bool> stop_verify{false};
    std::atomic<int> mismatches{0};
    std::thread verifier(phaseAVerifier, std::cref(f), std::cref(l),
                         ctx.get(), &stop_verify, &mismatches);

    // Sync with the sender at the end of phase A. The sender only arrives
    // after every verify request has been answered, so arriving early is
    // fine.
    if (!barrier(f.workdir, 1, "phaseA", 1800)) {
        stop_verify.store(true, std::memory_order_relaxed);
        verifier.join();
        _exit(1);
    }

    // Stress phase: the engine's handshake daemon does all the work. Keep a
    // heartbeat so a silent process is distinguishable from a live one.
    std::atomic<bool> heartbeat_stop{false};
    std::thread heartbeat([&] {
        int beat = 0;
        while (!heartbeat_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            LOG(INFO) << "[stress][receiver] alive beat=" << ++beat;
        }
    });

    std::string done_info;
    // duration + phase A headroom + slack; a missing stressDone means the
    // sender wedged or died. The launcher also kills us once the sender
    // reports a wedge, so this timeout is only a backstop.
    const int wait_sec = f.duration_sec + 900;
    const bool got_done = waitForFile(filePath(f.workdir, "stressDone.0"),
                                      &done_info, wait_sec);

    stop_verify.store(true, std::memory_order_relaxed);
    verifier.join();
    heartbeat_stop.store(true, std::memory_order_relaxed);
    heartbeat.join();

    if (!got_done) {
        LOG(ERROR) << "[stress] sender never finished (wedge/crash); "
                      "exiting with code 2";
        // Do not tear the engine down: a wedged peer makes cnclFreeComm
        // hang. Exit is cleaner and the launcher reaps us.
        _exit(2);
    }
    LOG(INFO) << "[stress] sender finished (" << done_info
              << "); verifying regions";
    if (mismatches.load() != 0) {
        writeFile(filePath(f.workdir, "verifyDone.1"),
                  "phase-A mismatches=" + std::to_string(mismatches.load()));
        barrier(f.workdir, 1, "teardown", 120);
        teardownEngine(ctx.get());
        return 6;
    }

    // Final verification: every byte of every sender thread's region must
    // equal the gold pattern of its slice.
    const uint64_t dst_base = (uint64_t)(uintptr_t)ctx->buffer;
    std::atomic<int64_t> bad_slots{0};
    std::atomic<int> next_thread{0};
    std::vector<std::thread> checkers;
    const int check_threads = 8;
    const size_t chunk = 16 << 20;  // D2H granularity
    std::mutex log_mutex;
    for (int c = 0; c < check_threads; ++c) {
        checkers.emplace_back([&, c] {
            std::vector<uint8_t> host(chunk);
            for (;;) {
                const int t = next_thread.fetch_add(1);
                if (t >= f.threads) break;
                const uint64_t region =
                    dst_base + (size_t)t * l.phase_b_region;
                for (int i = 0; i < f.slices; ++i) {
                    const uint64_t addr = region + (size_t)i * f.slice_bytes;
                    for (size_t off = 0; off < f.slice_bytes; off += chunk) {
                        const size_t bytes =
                            std::min(chunk, f.slice_bytes - off);
                        if (!copyToHost(host.data(),
                                        reinterpret_cast<const void*>(addr +
                                                                      off),
                                        bytes)) {
                            bad_slots.fetch_add(1);
                            continue;
                        }
                        // comparePattern works on whole slices; for chunked
                        // reads compare word-by-word with an offset.
                        const uint64_t* words =
                            reinterpret_cast<const uint64_t*>(host.data());
                        const size_t nwords = bytes / 8;
                        for (size_t w = 0; w < nwords; ++w) {
                            const uint64_t want = patternWord(
                                kGoldTag, i, (off / 8) + w);
                            if (words[w] == want) continue;
                            bad_slots.fetch_add(1);
                            std::lock_guard<std::mutex> g(log_mutex);
                            LOG(ERROR)
                                << "[stress] verify mismatch thread " << t
                                << " slice " << i << " word " << ((off / 8) + w)
                                << " got 0x" << std::hex << words[w]
                                << " want 0x" << want << std::dec;
                            break;  // one report per chunk is enough
                        }
                    }
                }
                if ((t + 1) % 16 == 0) {
                    LOG(INFO) << "[stress] verify progress " << (t + 1) << "/"
                              << f.threads;
                }
            }
        });
    }
    for (auto& c : checkers) c.join();

    int rc = 0;
    if (bad_slots.load() == 0) {
        writeFile(filePath(f.workdir, "verifyDone.1"), "ok");
        LOG(INFO) << "[stress] final verification PASSED";
    } else {
        writeFile(filePath(f.workdir, "verifyDone.1"),
                  "bad_slots=" + std::to_string(bad_slots.load()));
        LOG(ERROR) << "[stress] final verification FAILED, bad_slots="
                   << bad_slots.load();
        rc = 6;
    }
    if (!barrier(f.workdir, 1, "teardown", 120)) {
        // The sender is gone; don't hang trying to sync with it.
        _exit(rc ? rc : 1);
    }
    teardownEngine(ctx.get());
    return rc;
}

// ---------------------------------------------------------------------------
// Launcher: fork rank 0 / rank 1 children, each logging to its own file.
// ---------------------------------------------------------------------------

int spawnChild(int rank, const Flags& f) {
    const pid_t pid = fork();
    if (pid < 0) {
        PLOG(ERROR) << "fork";
        return -1;
    }
    if (pid == 0) {
        const std::string log_path = filePath(
            f.workdir, "rank" + std::to_string(rank) + ".log");
        const int fd =
            open(log_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            if (fd > STDERR_FILENO) close(fd);
        }
        auto str = [](const char* k, const std::string& v) {
            return std::string("--") + k + "=" + v;
        };
        auto num = [](const char* k, int64_t v) {
            return std::string("--") + k + "=" + std::to_string(v);
        };
        const std::string rank_arg = num("rank", rank);
        std::vector<std::string> args = {
            rank_arg,
            str("workdir", f.workdir),
            num("device0", f.device0),
            num("device1", f.device1),
            num("threads", f.threads),
            num("slices", f.slices),
            num("slice-bytes", (int64_t)f.slice_bytes),
            num("duration", f.duration_sec),
            num("variants", f.variants),
            num("batch-timeout", f.batch_timeout_sec),
            num("stall-warn", f.stall_warn_sec),
            num("wedge-window", f.wedge_window_sec),
            num("kill-peer-at", f.kill_peer_at_sec),
            num("phase-a-rounds", f.phase_a_rounds),
            num("phase-a-threads", f.phase_a_threads),
            num("phase-a-slices", f.phase_a_slices),
            num("phase-a-slice-bytes", (int64_t)f.phase_a_slice_bytes),
        };
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>("cncl_notifier_stress"));
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execv("/proc/self/exe", argv.data());
        PLOG(FATAL) << "execv";
    }
    return pid;
}

int runLauncher(const Flags& f) {
    std::string workdir = f.workdir;
    if (workdir.empty()) {
        char tmpl[] = "/tmp/cncl_stress_XXXXXX";
        const char* made = mkdtemp(tmpl);
        if (!made) {
            PLOG(ERROR) << "mkdtemp";
            return 1;
        }
        workdir = made;
    }
    Flags child = f;
    child.workdir = workdir;
    LOG(INFO) << "[stress] workdir " << workdir;

    const pid_t rank0 = spawnChild(0, child);
    const pid_t rank1 = spawnChild(1, child);
    if (rank0 < 0 || rank1 < 0) {
        if (rank0 > 0) kill(rank0, SIGKILL);
        if (rank1 > 0) kill(rank1, SIGKILL);
        return 1;
    }

    // Watchdog: phase A worst case + stress + verify + slack.
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(f.phase_a_rounds * 120 + f.duration_sec + 2400);
    int remaining = 2;
    int rc0 = -1, rc1 = -1;
    bool timed_out = false;
    auto kill_after_grace = [&](pid_t who, int grace_sec) {
        // One child reported a wedge-class exit; give the peer a short grace
        // period to flush its own logs, then stop it so the run stays short.
        std::this_thread::sleep_for(std::chrono::seconds(grace_sec));
        kill(who, SIGKILL);
    };
    while (remaining > 0) {
        int status = 0;
        const pid_t done = waitpid(-1, &status, WNOHANG);
        if (done > 0) {
            --remaining;
            int code = -1;
            if (WIFSIGNALED(status)) {
                code = 128 + WTERMSIG(status);
            } else if (WIFEXITED(status)) {
                code = WEXITSTATUS(status);
            }
            if (done == rank0) rc0 = code;
            if (done == rank1) rc1 = code;
            const int wedge = (code == 2 || code == 3);
            if (wedge && remaining > 0) {
                kill_after_grace(done == rank0 ? rank1 : rank0, 30);
                int kstatus;
                while (waitpid(-1, &kstatus, WNOHANG) > 0) --remaining;
                remaining = 0;
            }
            continue;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            timed_out = true;
            kill(rank0, SIGKILL);
            kill(rank1, SIGKILL);
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // Reap anything left.
    int status;
    while (waitpid(-1, &status, WNOHANG) > 0) {}

    std::ostringstream summary;
    summary << "[stress] RESULT rank0=" << rc0 << " rank1=" << rc1
            << (timed_out ? " (launcher watchdog fired)" : "") << " workdir="
            << workdir;
    if (rc0 == 0 && rc1 == 0 && !timed_out) {
        LOG(INFO) << summary.str() << " => PASSED";
        return 0;
    }
    LOG(ERROR) << summary.str() << " => FAILED";
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    Flags f;
    if (!parseFlags(argc, argv, &f)) return 1;

    unsigned int device_count = 0;
    if (!checkCnrt(cnrtGetDeviceCount(&device_count), "cnrtGetDeviceCount") ||
        device_count < 2) {
        LOG(ERROR) << "The CNCL stress test requires two MLU devices";
        return 1;
    }

    if (f.rank < 0) return runLauncher(f);
    if (f.rank > 1 || f.workdir.empty()) {
        LOG(ERROR) << "Invalid --rank/--workdir combination";
        return 1;
    }
    const Layout layout = computeLayout(f);
    LOG(INFO) << "[stress] rank " << f.rank << " buffer bytes "
              << layout.total_bytes;
    if (f.rank == 0) return runSender(f, layout);
    return runReceiver(f, layout);
}
