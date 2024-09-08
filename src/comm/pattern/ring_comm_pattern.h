#pragma once

#include <cuda_runtime.h>

#include "comm/pattern/comm_pattern.h"
#include "transport.h"

namespace gccl {

struct RingTransferInfo {
  // Device id, stage id, transfer id
  std::vector<std::vector<std::vector<int>>> tr_ids;
};

struct RingConn {
  SendDevMem *send_dev_mem;
  RecvDevMem *recv_dev_mem;

  ConnInfo conn_info;
  void *send_resources;
  void *recv_resources;
  transportProxyInfo *send_proxy, *recv_proxy;
};
struct RingCommPatternInfo {
  int n_stages;
  size_t extra_buff_size;
  size_t *max_comm_size;  // max communication size for each stage
  long long *send_ids;
  size_t *send_off;
  long long *recv_ids;
  size_t *recv_off;
  size_t *cpu_send_off, *cpu_recv_off, *cpu_max_comm_size;

  size_t buffer_size, max_feat_size;
  RingConn forward_conn, backward_conn;
  void *dev_extra_mem;

  BinStream &serialize(BinStream &stream) const;
  BinStream &deserialize(BinStream &stream);
  void CopyGraphInfoToDev();
  void Print() const;
};

class RingCommPattern : public CommPattern {
 public:
  RingCommPattern(const std::vector<int> &dev_topo,
                  const std::vector<ConnType> &conn_type)
      : CommPattern(dev_topo, conn_type) {}
  ~RingCommPattern() override {}

  std::vector<CommPatternInfo> BuildCommPatternInfos(
      Config *config, const std::vector<std::map<int, int>> &local_mappings,
      const TransferRequest &req, int nparts) override;

  void SetupConnection(CommPatternInfo *info, Coordinator *coor, int bid,
                       const std::vector<int> &conn_peer) override;

  void SortTransferInfoByLocalId(
      RingTransferInfo &tr_info,
      const std::vector<std::map<int, int>> &local_mappings, int n_parts);
  void StartProxy(Coordinator *coor, CommPatternInfo *info) override;
  void SaveProxy(Coordinator *coor, CommPatternInfo *info, int feat_size,
                 int n_threads, bool forward) override;
};

}  // namespace gccl
