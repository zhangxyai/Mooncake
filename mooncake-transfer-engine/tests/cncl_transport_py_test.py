#!/usr/bin/env python3
# Copyright 2024 KVCache.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Two-process functional test for the TransferEngine Python interface on
# Cambricon MLU. Covers the cncl transport (WRITE only, each session
# endpoint in its own process) and, as a reference, the rdma transport.
#
# The CNCL transport requires the two endpoints to live in separate
# processes (a clique id can be initialized only once per process), so the
# launcher re-executes itself as --rank 0 / --rank 1 children sharing a
# workdir. Both ranks allocate their buffer as a torch_mlu tensor and
# register its device address through the Python binding, so the whole
# data path is exercised from Python.
#
# Usage:
#   python3 cncl_transport_py_test.py --protocol cncl \
#       --engine-lib /path/to/build/mooncake-integration
#   python3 cncl_transport_py_test.py --protocol rdma \
#       --engine-lib /path/to/build/mooncake-integration
#
# The module directory can also be supplied via MOONCAKE_ENGINE_LIB.

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import torch_mlu  # noqa: F401  registers the "mlu" device backend

BUFFER_BYTES = 16 << 20  # 16 MiB per rank
TRANSFER_BYTES = 1 << 20  # 1 MiB per test transfer
OFFSET_BYTES = 4096


def log(msg):
    print(f"[py-test] {msg}", flush=True)


def file_path(workdir, name):
    return os.path.join(workdir, name)


def write_file(path, content):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.rename(tmp, path)


def wait_for_file(path, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


def barrier(workdir, rank, step):
    write_file(file_path(workdir, f"{step}.{rank}"), "ok")
    wait_for_file(file_path(workdir, f"{step}.{1 - rank}"))


def pattern(src_rank, offset, length):
    """Position-dependent pattern (mirrors the C++ example)."""
    vals = (np.arange(length, dtype=np.uint64) + offset) * np.uint64(
        2654435761
    ) + np.uint64(src_rank * 97)
    return (vals >> np.uint64(24)).astype(np.uint8)


def verify_region(buffer, src_rank, buf_offset, pattern_offset, length):
    """Check buffer[buf_offset:buf_offset+length] against the pattern that
    the writer staged at pattern_offset in its own buffer."""
    got = buffer[buf_offset : buf_offset + length].to("cpu").numpy()
    want = pattern(src_rank, pattern_offset, length)
    if not np.array_equal(got, want):
        bad = int(np.argmax(got != want))
        log(
            f"verify failed at offset {buf_offset + bad}: "
            f"got {got[bad]} want {want[bad]}"
        )
        return False
    return True


def run_rank(rank, workdir, protocol, engine_lib):
    import engine  # resolved via --engine-lib / MOONCAKE_ENGINE_LIB

    device = f"mlu:{rank}"
    buffer = torch.zeros(BUFFER_BYTES, dtype=torch.uint8, device=device)

    te = engine.TransferEngine()
    rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", protocol, "")
    assert rc == 0, f"initialize failed: {rc}"
    rc = te.register_memory(buffer.data_ptr(), BUFFER_BYTES, f"mlu:{rank}")
    assert rc == 0, f"register_memory failed: {rc}"

    name = f"127.0.0.1:{te.get_rpc_port()}"
    write_file(file_path(workdir, f"name.{rank}"), name)
    write_file(file_path(workdir, f"addr.{rank}"), str(buffer.data_ptr()))
    peer_name = wait_for_file(file_path(workdir, f"name.{1 - rank}"))
    peer_addr = int(wait_for_file(file_path(workdir, f"addr.{1 - rank}")))

    checks = []

    def check(name_, ok):
        checks.append(ok)
        log(f"rank {rank}: {name_}: " f"{'PASSED' if ok else 'FAILED'}")

    # 1. Offset sync WRITE rank0 -> rank1, verified on the reader.
    barrier(workdir, rank, "t1")
    if rank == 0:
        src = torch.from_numpy(pattern(0, OFFSET_BYTES, TRANSFER_BYTES)).to(device)
        buffer[OFFSET_BYTES : OFFSET_BYTES + TRANSFER_BYTES] = src
        rc = te.transfer_sync_write(
            peer_name,
            buffer.data_ptr() + OFFSET_BYTES,
            peer_addr + OFFSET_BYTES,
            TRANSFER_BYTES,
        )
        check("offset sync write submit", rc == 0)
    barrier(workdir, rank, "t1-done")
    if rank == 1:
        check(
            "offset sync write data",
            verify_region(buffer, 0, OFFSET_BYTES, OFFSET_BYTES, TRANSFER_BYTES),
        )

    # 2. Reverse sync WRITE rank1 -> rank0.
    barrier(workdir, rank, "t2")
    if rank == 1:
        src = torch.from_numpy(pattern(1, 2 * OFFSET_BYTES, TRANSFER_BYTES)).to(device)
        buffer[2 * OFFSET_BYTES : 2 * OFFSET_BYTES + TRANSFER_BYTES] = src
        rc = te.transfer_sync_write(
            peer_name,
            buffer.data_ptr() + 2 * OFFSET_BYTES,
            peer_addr + 2 * OFFSET_BYTES,
            TRANSFER_BYTES,
        )
        check("reverse sync write submit", rc == 0)
    barrier(workdir, rank, "t2-done")
    if rank == 0:
        check(
            "reverse sync write data",
            verify_region(
                buffer, 1, 2 * OFFSET_BYTES, 2 * OFFSET_BYTES, TRANSFER_BYTES
            ),
        )

    # 3. Async WRITE (transfer_submit_write + transfer_check_status).
    barrier(workdir, rank, "t3")
    if rank == 0:
        src = torch.from_numpy(pattern(0, 3 * OFFSET_BYTES, TRANSFER_BYTES)).to(device)
        buffer[3 * OFFSET_BYTES : 3 * OFFSET_BYTES + TRANSFER_BYTES] = src
        batch_id = te.transfer_submit_write(
            peer_name,
            buffer.data_ptr() + 3 * OFFSET_BYTES,
            peer_addr + 3 * OFFSET_BYTES,
            TRANSFER_BYTES,
        )
        check("async write submit", batch_id != 0)
        status = 0
        deadline = time.time() + 60
        while time.time() < deadline:
            status = te.transfer_check_status(batch_id)
            if status != 0:
                break
            time.sleep(0.001)
        check("async write completion", status == 1)
    barrier(workdir, rank, "t3-done")
    if rank == 1:
        check(
            "async write data",
            verify_region(
                buffer, 0, 3 * OFFSET_BYTES, 3 * OFFSET_BYTES, TRANSFER_BYTES
            ),
        )

    # 4. READ semantics: cncl rejects READ; rdma supports it. The read
    #    pulls the peer's region written in test 1 (pattern(0, OFFSET))
    #    into the local buffer at offset 0.
    barrier(workdir, rank, "t4")
    if rank == 0:
        rc = te.transfer_sync_read(
            peer_name, buffer.data_ptr(), peer_addr + OFFSET_BYTES, TRANSFER_BYTES
        )
        if protocol == "cncl":
            check("cncl read rejection", rc != 0)
        else:
            check("rdma read submit", rc == 0)
            check(
                "rdma read data",
                verify_region(buffer, 0, 0, OFFSET_BYTES, TRANSFER_BYTES),
            )
    barrier(workdir, rank, "t4-done")

    rc = te.unregister_memory(buffer.data_ptr())
    assert rc == 0, f"unregister_memory failed: {rc}"

    ok = all(checks)
    log(
        f"rank {rank} [{protocol}]: "
        f"{'PASSED' if ok else 'FAILED'} "
        f"({sum(checks)}/{len(checks)} checks)"
    )
    return 0 if ok else 1


def run_launcher(protocol, engine_lib):
    workdir = tempfile.mkdtemp(prefix="cncl_py_test_")
    procs = []
    for rank in (0, 1):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--rank",
            str(rank),
            "--workdir",
            workdir,
            "--protocol",
            protocol,
        ]
        if engine_lib:
            cmd += ["--engine-lib", engine_lib]
        procs.append(subprocess.Popen(cmd))
    rc = 0
    for p in procs:
        if p.wait() != 0:
            rc = 1
    log(f"python interface test [{protocol}]: " f"{'PASSED' if rc == 0 else 'FAILED'}")
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="cncl", choices=["cncl", "rdma"])
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--workdir", default="")
    parser.add_argument(
        "--engine-lib", default=os.environ.get("MOONCAKE_ENGINE_LIB", "")
    )
    args = parser.parse_args()

    if args.engine_lib:
        sys.path.insert(0, args.engine_lib)

    if args.rank >= 0:
        sys.exit(run_rank(args.rank, args.workdir, args.protocol, args.engine_lib))
    sys.exit(run_launcher(args.protocol, args.engine_lib))


if __name__ == "__main__":
    main()
