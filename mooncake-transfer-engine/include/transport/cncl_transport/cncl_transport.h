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

#ifndef CNCL_TRANSPORT_H_
#define CNCL_TRANSPORT_H_

#include <memory>

#include "transport/transport.h"

namespace mooncake {

// CNCL (Cambricon Communication Library) transport for the classic Transfer
// Engine API, mirroring the NCCL host transport. CNCL 1.30 has no host-side
// one-sided RMA primitives, so WRITE is implemented over the two-sided
// cnclSend/cnclRecv pair: the writer reserves the peer's matching cnclRecv
// through a per-transfer handshake RPC, and only then enqueues its cnclSend.
// Native CNCL communicators are intentionally private.
//
// Install the CNCL transport before registering its buffers.
//
// Each session is a two-rank CNCL communicator scoped to one
// (peer name, local device, peer device) triple and is bootstrapped lazily by
// the first transfer toward that endpoint. Session initialization is attempted
// once per endpoint/device pair. A terminal session failure remains cached,
// and later transfers return that error without retrying bootstrap. Recovery
// requires recreating the CNCL transport (normally its TransferEngine
// instance) on both peers; one-sided restart is unsupported.
//
// CNCL matches cnclSend/cnclRecv per rank pair in issue order and forbids
// initializing the same clique id twice within one process. Consequently:
// - both endpoints of a session must live in separate processes, and
// - the per-session submit lock spans the reservation RPC and the cnclSend
//   enqueue so both endpoints issue their operations in the same order.
//
// Only WRITE is supported. READ is rejected with
// Status::NotSupportedTransport without submitting a CNCL operation: a READ
// would have to make the data owner enqueue cnclSend before the reader's
// matching cnclRecv, which cannot be ordered safely against concurrent
// WRITEs on the same session without a second reservation round.
class CnclTransport final : public Transport {
   public:
    CnclTransport();
    ~CnclTransport() override;

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

    // Completion is caller-driven (no background poller), so a batch that is
    // freed without further polling — abandoned after a failed submit or an
    // outer timeout — would leak its groups and notifiers. Settle them
    // synchronously instead.
    void abortBatch(BatchID batch_id) override;

   protected:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> metadata,
                std::shared_ptr<Topology> topology) override;

   private:
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location, bool remote_accessible,
                            bool update_metadata = true) override;
    int unregisterLocalMemory(void* addr, bool update_metadata = true) override;
    int registerLocalMemoryBatch(const std::vector<BufferEntry>& buffer_list,
                                 const std::string& location) override;
    int unregisterLocalMemoryBatch(
        const std::vector<void*>& addr_list) override;

    const char* getName() const override { return "cncl"; }

    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mooncake

#endif  // CNCL_TRANSPORT_H_
