// MLU transport benchmark: CNCL vs RDMA over identical MLU buffers.
//
// Two-process design (same as cncl_transport_example.cpp): the launcher
// re-executes itself as --rank 0 / --rank 1 children sharing a workdir,
// each pinned to one MLU device with its own TransferEngine instance.
// CNCL requires the two endpoints in separate processes; RDMA runs the
// same harness so both transports move the same bytes between the same
// buffers.
//
// Options are --key=value pairs (parsed manually, like
// cncl_transport_example.cpp):
//   protocol=cncl|rdma          transport under test
//   mode=all|integrity|latency|bandwidth
//   sizes=<comma list>          transfer sizes in bytes
//   latency_iters=N             iterations for the smallest latency size
//   latency_min_iters=N         iteration floor for larger sizes
//   latency_bytes_budget=B      per-size byte budget for latency iterations
//   latency_warmup=N            untimed warmup cap per size (scaled down for
//                               sizes above the byte budget)
//   bw_total_bytes=B            bytes moved per bandwidth size
//   bw_min_ops=K                minimum ops per bandwidth size (keeps the
//                               pipeline full when size exceeds the budget)
//   bw_outstanding=K            pipelined in-flight transfers
//   bw_max_ops=K                op cap per bandwidth size (bounds small-size
//                               run time)
//
// Phases:
//   integrity  one patterned WRITE per direction + D2H verification
//   latency    sequential sync WRITEs per size, reports min/avg/p50/p99
//   bandwidth  pipelined WRITEs (K outstanding) per size, reports GB/s
//
// Results are printed as CSV lines prefixed with "RESULT," so they can be
// collected from the interleaved glog output of both ranks.

#include <glog/logging.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "cnrt.h"
#include "transfer_engine.h"

namespace {

constexpr size_t kBufferBytes = 9ull << 30;  // per-rank MLU buffer

using mooncake::BatchID;
using mooncake::SegmentHandle;
using mooncake::TransferEngine;
using mooncake::TransferRequest;
using mooncake::TransferStatus;
using mooncake::TransferStatusEnum;

const SegmentHandle kInvalidSegment = static_cast<SegmentHandle>(-1);

struct Options {
    std::string protocol = "cncl";
    std::string mode = "all";
    // 4 KiB * 2^n ladder up to 8 GiB.
    std::string sizes =
        "4096,8192,16384,32768,65536,131072,262144,524288,1048576,"
        "2097152,4194304,8388608,16777216,33554432,67108864,"
        "134217728,268435456,536870912,1073741824,2147483648,"
        "4294967296,8589934592";
    int latency_iters = 200;
    int latency_min_iters = 20;
    uint64_t latency_bytes_budget = 64ull << 20;
    int latency_warmup = 10;
    uint64_t bw_total_bytes = 512ull << 20;
    int bw_min_ops = 8;
    int bw_outstanding = 8;
    int bw_max_ops = 8192;
    int rank = -1;
    std::string workdir;
};

Options parseOptions(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) continue;
        const auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = arg.substr(2, eq - 2);
        const std::string value = arg.substr(eq + 1);
        if (key == "protocol") {
            opts.protocol = value;
        } else if (key == "mode") {
            opts.mode = value;
        } else if (key == "sizes") {
            opts.sizes = value;
        } else if (key == "latency_iters") {
            opts.latency_iters = atoi(value.c_str());
        } else if (key == "latency_min_iters") {
            opts.latency_min_iters = atoi(value.c_str());
        } else if (key == "latency_bytes_budget") {
            opts.latency_bytes_budget = strtoull(value.c_str(), nullptr, 10);
        } else if (key == "latency_warmup") {
            opts.latency_warmup = atoi(value.c_str());
        } else if (key == "bw_total_bytes") {
            opts.bw_total_bytes = strtoull(value.c_str(), nullptr, 10);
        } else if (key == "bw_min_ops") {
            opts.bw_min_ops = atoi(value.c_str());
        } else if (key == "bw_outstanding") {
            opts.bw_outstanding = atoi(value.c_str());
        } else if (key == "bw_max_ops") {
            opts.bw_max_ops = atoi(value.c_str());
        } else if (key == "rank") {
            opts.rank = atoi(value.c_str());
        } else if (key == "workdir") {
            opts.workdir = value;
        }
    }
    return opts;
}

Options g_opts;

// --- small file/barrier helpers (standalone, as in the example) -----------

std::string filePath(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

bool writeFile(const std::string& path, const std::string& content) {
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    const size_t n = fwrite(content.data(), 1, content.size(), f);
    fclose(f);
    if (n != content.size()) return false;
    return rename(tmp.c_str(), path.c_str()) == 0;
}

bool readFile(const std::string& path, std::string* content) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char buf[256];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (n == 0) return false;
    content->assign(buf, n);
    return true;
}

bool waitForFile(const std::string& path, std::string* content) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
        if (readFile(path, content)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    LOG(ERROR) << "Timed out waiting for " << path;
    return false;
}

bool barrier(const std::string& workdir, int rank, const std::string& step) {
    if (!writeFile(filePath(workdir, step + "." + std::to_string(rank)),
                   "ok")) {
        return false;
    }
    std::string ignored;
    return waitForFile(filePath(workdir, step + "." + std::to_string(1 - rank)),
                       &ignored);
}

// --- cnrt helpers ----------------------------------------------------------

bool checkCnrt(cnrtRet_t ret, const char* what) {
    if (ret == cnrtSuccess) return true;
    LOG(ERROR) << what << " failed: " << cnrtGetErrorStr(ret);
    return false;
}

// Position-dependent pattern so misrouted regions cannot verify.
inline uint8_t patternByte(int src_rank, uint64_t offset) {
    return static_cast<uint8_t>((offset * 2654435761ull + src_rank * 97) >> 24);
}

void fillHostPattern(std::vector<uint8_t>* host, int src_rank,
                     uint64_t offset) {
    for (size_t i = 0; i < host->size(); ++i) {
        (*host)[i] = patternByte(src_rank, offset + i);
    }
}

bool verifyRegion(void* dev, uint64_t offset, size_t length, int src_rank) {
    std::vector<uint8_t> host(length);
    if (!checkCnrt(cnrtMemcpy(host.data(), static_cast<char*>(dev) + offset,
                              length, cnrtMemcpyDevToHost),
                   "cnrtMemcpy D2H")) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        if (host[i] != patternByte(src_rank, offset + i)) {
            LOG(ERROR) << "Verification failed at offset " << offset + i
                       << ": got " << (int)host[i] << " want "
                       << (int)patternByte(src_rank, offset + i);
            return false;
        }
    }
    return true;
}

// --- transfer helpers ------------------------------------------------------

bool submitWriteAsync(TransferEngine* engine, SegmentHandle target, void* local,
                      uint64_t remote, size_t length, BatchID batch) {
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
    return true;
}

// Polls like the Python binding's transferSync: tight getTransferStatus
// loop without sleep, so the measured latency is not inflated by poll
// granularity.
bool waitBatch(TransferEngine* engine, BatchID batch, int timeout_seconds) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(timeout_seconds);
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
    }
    LOG(ERROR) << "Timed out waiting for transfer";
    return false;
}

bool submitWriteSync(TransferEngine* engine, SegmentHandle target, void* local,
                     uint64_t remote, size_t length) {
    BatchID batch = engine->allocateBatchID(1);
    if (batch == mooncake::INVALID_BATCH_ID) return false;
    if (!submitWriteAsync(engine, target, local, remote, length, batch) ||
        !waitBatch(engine, batch, 300)) {
        engine->freeBatchID(batch);
        return false;
    }
    auto result = engine->freeBatchID(batch);
    if (!result.ok()) {
        LOG(ERROR) << result.ToString();
        return false;
    }
    return true;
}

std::vector<size_t> parseSizes(const std::string& spec) {
    std::vector<size_t> sizes;
    size_t start = 0;
    while (start <= spec.size()) {
        size_t comma = spec.find(',', start);
        if (comma == std::string::npos) comma = spec.size();
        if (comma > start) {
            sizes.push_back(strtoull(spec.substr(start, comma - start).c_str(),
                                     nullptr, 10));
        }
        start = comma + 1;
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

// --- benchmark phases ------------------------------------------------------

struct RankCtx {
    int rank = 0;
    std::string workdir;
    void* buffer = nullptr;
    std::unique_ptr<TransferEngine> engine;
    SegmentHandle peer_segment = kInvalidSegment;
    std::string peer_name;
    uint64_t peer_buffer = 0;
    // The bandwidth pattern is position-dependent and never changes, so the
    // writer fills its full buffer only once per run instead of per size.
    bool bw_pattern_filled = false;
};

// Step names embed the writer rank (and size, where applicable) so every
// barrier invocation in a run is unique: barrier files are never deleted,
// so a repeated name would let a later phase sync on stale files and race
// ahead of its writer.
bool integrityPhase(RankCtx* ctx, int writer_rank) {
    const size_t len = 4ull << 20;
    const uint64_t offset = 8ull << 20;
    const int reader_rank = 1 - writer_rank;
    const std::string step = "integrity-w" + std::to_string(writer_rank);
    if (!barrier(ctx->workdir, ctx->rank, step)) return false;
    if (ctx->rank == writer_rank) {
        std::vector<uint8_t> host(len);
        fillHostPattern(&host, writer_rank, offset);
        if (!checkCnrt(cnrtMemcpy(static_cast<char*>(ctx->buffer) + offset,
                                  host.data(), len, cnrtMemcpyHostToDev),
                       "cnrtMemcpy H2D")) {
            return false;
        }
        if (!submitWriteSync(ctx->engine.get(), ctx->peer_segment,
                             static_cast<char*>(ctx->buffer) + offset,
                             ctx->peer_buffer + offset, len)) {
            return false;
        }
    }
    if (!barrier(ctx->workdir, ctx->rank, step + "-done")) return false;
    if (ctx->rank == reader_rank) {
        if (!verifyRegion(ctx->buffer, offset, len, writer_rank)) {
            printf("RESULT,integrity,protocol=%s,writer=%d,ok=0\n",
                   g_opts.protocol.c_str(), writer_rank);
            return false;
        }
        printf("RESULT,integrity,protocol=%s,writer=%d,ok=1\n",
               g_opts.protocol.c_str(), writer_rank);
    }
    return true;
}

// One direction only; the writer measures submit->complete per iteration.
bool latencyPhase(RankCtx* ctx, int writer_rank, size_t size) {
    const std::string step =
        "latency-w" + std::to_string(writer_rank) + "-s" + std::to_string(size);
    if (!barrier(ctx->workdir, ctx->rank, step)) return false;

    if (ctx->rank != writer_rank) {
        return barrier(ctx->workdir, ctx->rank, step + "-done");
    }

    int iters = static_cast<int>(std::min<uint64_t>(
        g_opts.latency_bytes_budget / size, g_opts.latency_iters));
    iters = std::max(iters, g_opts.latency_min_iters);
    // Warmup matters mainly for sizes within the byte budget; scale it down
    // so huge sizes do not spend more time warming up than being measured.
    const int warmup = std::max(
        1, std::min<int>(g_opts.latency_warmup,
                         static_cast<int>(g_opts.latency_bytes_budget / size)));

    const uint64_t span = kBufferBytes - size;
    for (int i = 0; i < warmup; ++i) {
        const uint64_t off = (static_cast<uint64_t>(i) * size * 8) % span;
        if (!submitWriteSync(ctx->engine.get(), ctx->peer_segment,
                             static_cast<char*>(ctx->buffer) + off,
                             ctx->peer_buffer + off, size)) {
            return false;
        }
    }

    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        const uint64_t off = (static_cast<uint64_t>(i) * size * 8) % span;
        void* src = static_cast<char*>(ctx->buffer) + off;
        const auto t0 = std::chrono::steady_clock::now();
        if (!submitWriteSync(ctx->engine.get(), ctx->peer_segment, src,
                             ctx->peer_buffer + off, size)) {
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(us.begin(), us.end());
    const double avg = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
    printf(
        "RESULT,latency,protocol=%s,writer=%d,size=%zu,iters=%d,"
        "min_us=%.1f,p50_us=%.1f,avg_us=%.1f,p99_us=%.1f\n",
        g_opts.protocol.c_str(), writer_rank, size, iters, us.front(),
        us[us.size() / 2], avg, us[(us.size() * 99) / 100]);
    return barrier(ctx->workdir, ctx->rank, step + "-done");
}

// One direction; the writer keeps bw_outstanding batches in flight and
// the reader verifies a sample of the destination afterwards. Finished
// tasks stay in a batch until freeBatchID, so each op gets its own batch:
// a completed slot's batch is freed and a fresh one takes its place.
bool bandwidthPhase(RankCtx* ctx, int writer_rank, size_t size) {
    const std::string step =
        "bw-w" + std::to_string(writer_rank) + "-s" + std::to_string(size);
    // Small sizes are capped by op count, large sizes by a minimum op count
    // (so the pipeline stays full when one op already exceeds the byte
    // budget); both bounds keep the run bounded while measuring steady
    // state.
    const uint64_t ops = std::max<uint64_t>(
        1, std::min<uint64_t>(
               std::max<uint64_t>(g_opts.bw_total_bytes / size,
                                  static_cast<uint64_t>(g_opts.bw_min_ops)),
               static_cast<uint64_t>(g_opts.bw_max_ops)));
    const int k =
        static_cast<int>(std::min<uint64_t>(g_opts.bw_outstanding, ops));
    const uint64_t span = kBufferBytes - size;

    if (ctx->rank == writer_rank && !ctx->bw_pattern_filled) {
        // Make the source position-dependent so the reader can verify. The
        // pattern depends only on position, so one fill serves every size.
        const size_t fill_chunk = 1ull << 20;
        std::vector<uint8_t> host(fill_chunk);
        for (uint64_t off = 0; off < kBufferBytes; off += fill_chunk) {
            const size_t n = std::min<size_t>(fill_chunk, kBufferBytes - off);
            fillHostPattern(&host, writer_rank, off);
            if (!checkCnrt(cnrtMemcpy(static_cast<char*>(ctx->buffer) + off,
                                      host.data(), n, cnrtMemcpyHostToDev),
                           "cnrtMemcpy H2D")) {
                return false;
            }
        }
        ctx->bw_pattern_filled = true;
    }
    if (!barrier(ctx->workdir, ctx->rank, step + "-fill")) return false;

    if (ctx->rank != writer_rank) {
        if (!barrier(ctx->workdir, ctx->rank, step + "-done")) return false;
        // Reader verifies a sample of the transferred regions.
        const int checks = std::min<uint64_t>(32, ops);
        for (int i = 0; i < checks; ++i) {
            const uint64_t op = i * (ops - 1) / std::max(1, checks - 1);
            const uint64_t off = (op * size * 8) % span;
            if (!verifyRegion(ctx->buffer, off,
                              std::min<size_t>(size, 1ull << 20),
                              writer_rank)) {
                printf(
                    "RESULT,bandwidth,protocol=%s,writer=%d,size=%zu,"
                    "verify=0\n",
                    g_opts.protocol.c_str(), writer_rank, size);
                return false;
            }
        }
        return true;
    }

    std::vector<BatchID> batches(k, mooncake::INVALID_BATCH_ID);
    uint64_t submitted = 0, completed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (completed < ops) {
        while (submitted < ops &&
               submitted - completed < static_cast<uint64_t>(k)) {
            const uint64_t off = (submitted * size * 8) % span;
            const int slot = static_cast<int>(submitted % k);
            batches[slot] = ctx->engine->allocateBatchID(1);
            if (batches[slot] == mooncake::INVALID_BATCH_ID) return false;
            if (!submitWriteAsync(ctx->engine.get(), ctx->peer_segment,
                                  static_cast<char*>(ctx->buffer) + off,
                                  ctx->peer_buffer + off, size,
                                  batches[slot])) {
                return false;
            }
            ++submitted;
        }
        const int slot = static_cast<int>(completed % k);
        if (!waitBatch(ctx->engine.get(), batches[slot], 600)) return false;
        auto result = ctx->engine->freeBatchID(batches[slot]);
        if (!result.ok()) {
            LOG(ERROR) << result.ToString();
            return false;
        }
        batches[slot] = mooncake::INVALID_BATCH_ID;
        ++completed;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double gbps = static_cast<double>(ops) * size / secs / 1e9;
    printf(
        "RESULT,bandwidth,protocol=%s,writer=%d,size=%zu,ops=%llu,"
        "outstanding=%d,time_s=%.3f,gbps=%.2f,verify=1\n",
        g_opts.protocol.c_str(), writer_rank, size,
        static_cast<unsigned long long>(ops), k, secs, gbps);
    return barrier(ctx->workdir, ctx->rank, step + "-done");
}

// --- rank body -------------------------------------------------------------

bool setupEngine(RankCtx* ctx) {
    if (g_opts.protocol == "cncl") {
        // CNCL must be the engine's only transport: no auto-discovery.
        ctx->engine = std::make_unique<TransferEngine>(false);
        if (ctx->engine->init(P2PHANDSHAKE, "127.0.0.1:0", "127.0.0.1", 0) !=
            0) {
            return false;
        }
        if (ctx->engine->installTransport("cncl", nullptr) == nullptr) {
            return false;
        }
    } else if (g_opts.protocol == "rdma") {
        // Auto-discover installs the RDMA transport from topology.
        ctx->engine = std::make_unique<TransferEngine>(true);
        if (ctx->engine->init(P2PHANDSHAKE, "127.0.0.1:0", "127.0.0.1", 0) !=
            0) {
            return false;
        }
    } else {
        LOG(ERROR) << "Unknown protocol " << g_opts.protocol;
        return false;
    }

    if (ctx->engine->registerLocalMemory(ctx->buffer, kBufferBytes,
                                         "mlu:" + std::to_string(ctx->rank)) !=
        0) {
        return false;
    }

    if (!writeFile(filePath(ctx->workdir, "name." + std::to_string(ctx->rank)),
                   ctx->engine->getLocalIpAndPort()) ||
        !writeFile(filePath(ctx->workdir, "addr." + std::to_string(ctx->rank)),
                   std::to_string(reinterpret_cast<uintptr_t>(ctx->buffer)))) {
        return false;
    }
    std::string peer_addr;
    if (!waitForFile(
            filePath(ctx->workdir, "name." + std::to_string(1 - ctx->rank)),
            &ctx->peer_name) ||
        !waitForFile(
            filePath(ctx->workdir, "addr." + std::to_string(1 - ctx->rank)),
            &peer_addr)) {
        return false;
    }
    ctx->peer_buffer = strtoull(peer_addr.c_str(), nullptr, 10);

    ctx->peer_segment = ctx->engine->openSegment(ctx->peer_name);
    return ctx->peer_segment != kInvalidSegment;
}

int runRank(int rank, const std::string& workdir) {
    if (!checkCnrt(cnrtSetDevice(rank), "cnrtSetDevice")) return 1;
    RankCtx ctx;
    ctx.rank = rank;
    ctx.workdir = workdir;
    if (!checkCnrt(cnrtMalloc(&ctx.buffer, kBufferBytes), "cnrtMalloc") ||
        !checkCnrt(cnrtMemset(ctx.buffer, 0, kBufferBytes), "cnrtMemset")) {
        return 1;
    }
    if (!setupEngine(&ctx)) {
        LOG(ERROR) << "Engine setup failed on rank " << rank;
        return 1;
    }
    LOG(INFO) << "[" << g_opts.protocol << "] rank " << rank
              << " ready, peer buffer at 0x" << std::hex << ctx.peer_buffer
              << std::dec;

    const auto sizes = parseSizes(g_opts.sizes);
    bool ok = true;
    for (size_t size : sizes) {
        if (size == 0 || size > kBufferBytes) {
            LOG(ERROR) << "Size " << size << " is outside (0, " << kBufferBytes
                       << "]";
            ok = false;
        }
    }
    if (g_opts.mode == "all" || g_opts.mode == "integrity") {
        ok = ok && integrityPhase(&ctx, 0) && integrityPhase(&ctx, 1);
    }
    if (ok && (g_opts.mode == "all" || g_opts.mode == "latency")) {
        for (size_t size : sizes) ok = ok && latencyPhase(&ctx, 0, size);
        for (size_t size : sizes) ok = ok && latencyPhase(&ctx, 1, size);
    }
    if (ok && (g_opts.mode == "all" || g_opts.mode == "bandwidth")) {
        for (size_t size : sizes) ok = ok && bandwidthPhase(&ctx, 0, size);
        for (size_t size : sizes) ok = ok && bandwidthPhase(&ctx, 1, size);
    }

    if (ctx.engine) {
        ctx.engine->unregisterLocalMemory(ctx.buffer);
        ctx.engine->freeEngine();
    }
    checkCnrt(cnrtFree(ctx.buffer), "cnrtFree");
    if (!ok) {
        LOG(ERROR) << "[" << g_opts.protocol << "] rank " << rank << " FAILED";
        return 1;
    }
    LOG(INFO) << "[" << g_opts.protocol << "] rank " << rank << " PASSED";
    return 0;
}

// --- launcher --------------------------------------------------------------

int runLauncher() {
    char tmpl[] = "/tmp/mlu_bench_XXXXXX";
    const char* workdir = mkdtemp(tmpl);
    if (!workdir) {
        PLOG(ERROR) << "mkdtemp";
        return 1;
    }
    std::vector<pid_t> pids;
    for (int rank = 0; rank < 2; ++rank) {
        const pid_t pid = fork();
        if (pid == 0) {
            const std::string rank_arg = "--rank=" + std::to_string(rank);
            const std::string workdir_arg = std::string("--workdir=") + workdir;
            const std::string proto_arg = "--protocol=" + g_opts.protocol;
            const std::string mode_arg = "--mode=" + g_opts.mode;
            const std::string sizes_arg = "--sizes=" + g_opts.sizes;
            const std::string lat_iters_arg =
                "--latency_iters=" + std::to_string(g_opts.latency_iters);
            const std::string lat_min_arg =
                "--latency_min_iters=" +
                std::to_string(g_opts.latency_min_iters);
            const std::string lat_budget_arg =
                "--latency_bytes_budget=" +
                std::to_string(g_opts.latency_bytes_budget);
            const std::string lat_warm_arg =
                "--latency_warmup=" + std::to_string(g_opts.latency_warmup);
            const std::string bw_total_arg =
                "--bw_total_bytes=" + std::to_string(g_opts.bw_total_bytes);
            const std::string bw_min_ops_arg =
                "--bw_min_ops=" + std::to_string(g_opts.bw_min_ops);
            const std::string bw_out_arg =
                "--bw_outstanding=" + std::to_string(g_opts.bw_outstanding);
            const std::string bw_max_arg =
                "--bw_max_ops=" + std::to_string(g_opts.bw_max_ops);
            execl("/proc/self/exe", "mlu_transport_bench", rank_arg.c_str(),
                  workdir_arg.c_str(), proto_arg.c_str(), mode_arg.c_str(),
                  sizes_arg.c_str(), lat_iters_arg.c_str(), lat_min_arg.c_str(),
                  lat_budget_arg.c_str(), lat_warm_arg.c_str(),
                  bw_total_arg.c_str(), bw_min_ops_arg.c_str(),
                  bw_out_arg.c_str(), bw_max_arg.c_str(), nullptr);
            PLOG(FATAL) << "execl";
        }
        if (pid < 0) {
            PLOG(ERROR) << "fork";
            return 1;
        }
        pids.push_back(pid);
    }
    int rc = 0;
    for (pid_t pid : pids) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc = 1;
    }
    printf("RESULT,summary,protocol=%s,rc=%d\n", g_opts.protocol.c_str(), rc);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    g_opts = parseOptions(argc, argv);
    if (g_opts.rank >= 0) return runRank(g_opts.rank, g_opts.workdir);
    return runLauncher();
}
