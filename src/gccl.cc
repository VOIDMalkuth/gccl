#include "gccl.h"

#include "base/bin_stream.h"
#include "comm/comm_info.h"

#include "comm/comm_scheduler.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "global_state.h"
#include "communicator.h"
#include "config.h"
#include "graph.h"
#include "param.h"
#include "utils.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
using namespace std::chrono;

namespace gccl {

extern GlobalState gccl_global;

void PartitionGraph(gcclComm_t comm, int n, int *xadj, int *adjncy,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj, int **sg_adjncy,
                    int mini_n, int *mini_xadj, int *mini_adjncy, int *mini_gid2mid, int *mini_node_weights, int *mini_edge_weights) {
  // To internal graph
  // Pass to scheduler
  auto begin =system_clock::now();

  Graph g;
  if (comm->GetCoordinator()->IsRoot()) {
    g = Graph(n, xadj, adjncy);
    if (mini_n > 0) {
      LOG(INFO) << "Building mini graph";
      g.mini_graph = std::make_shared<Graph>(mini_n, mini_xadj, mini_adjncy);
      g.mini_graph->node_weights = std::vector<int>(mini_node_weights, mini_node_weights + mini_n);
      g.mini_graph->edge_weights = std::vector<int>(mini_edge_weights, mini_edge_weights + mini_xadj[mini_n]);
      g.gid2mid = std::vector<int>(mini_gid2mid, mini_gid2mid + n);
    }
  }
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  auto *config = comm->GetConfig();
  
  auto build_graph_end = system_clock::now();
  std::cout << "Build graph took" << duration_cast<seconds>(build_graph_end-begin).count() << "s\n" << std::endl;
  
  comm_sch->BuildPartitionInfo(coor, config, g, "", info, sgn, sg_xadj,
                               sg_adjncy);
  gccl_global.previous_g = std::make_unique<Graph>(g);
  auto build_part_info_done = system_clock::now();
  std::cout << "Build part info took" << duration_cast<seconds>(build_part_info_done - build_graph_end).count() << "s\n" << std::endl;
}

void PartitionGraph(gcclComm_t comm, const char *cached_dir,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj, int **sg_adjncy,
                    int mini_n, int *mini_xadj, int *mini_adjncy, int *mini_gid2mid, int *mini_node_weights, int *mini_edge_weights) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  auto *config = comm->GetConfig();
  Graph g;

  auto build_graph_end = system_clock::now();
  comm_sch->BuildPartitionInfo(coor, config, g, cached_dir, info, sgn, sg_xadj,
                               sg_adjncy);
  gccl_global.previous_g = std::make_unique<Graph>(g);
  auto build_part_info_done = system_clock::now();
  std::cout << "Build part info took" << duration_cast<seconds>(build_part_info_done - build_graph_end).count() << "s\n" << std::endl;
}

void PartitionGraphWithPrepartInfo(gcclComm_t comm, gcclCommInfo_t *info, int *sgn, int **sg_xadj, int **sg_adjncy,
                                   size_t bin_stream_size, char *bin_stream_data) {
  // To internal graph
  // Pass to scheduler
  auto begin = system_clock::now();

  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  auto *config = comm->GetConfig();

  BinStream bin_stream;
  if (bin_stream_data != nullptr && bin_stream_size != 0) {
    bin_stream = BinStream(bin_stream_data, bin_stream_size);
  }
  Graph g;
  comm_sch->BuildPartitionInfo(coor, config, g, "", info, sgn, sg_xadj, sg_adjncy, true, bin_stream);
  gccl_global.previous_g = std::make_unique<Graph>(g);
  auto build_part_info_done = system_clock::now();
  std::cout << "Build part info took" << duration_cast<seconds>(build_part_info_done - begin).count() << "s\n" << std::endl;
}

void PrePartitionGraph(int n_peers, int n, int *xadj, int *adjncy,
                    int mini_n, int *mini_xadj, int *mini_adjncy, int *mini_gid2mid, int *mini_node_weights, int *mini_edge_weights,
                    size_t *bin_stream_size, char **bin_stream_data) {
  auto begin = system_clock::now();
  Graph g = Graph(n, xadj, adjncy);
  if (mini_n > 0) {
    LOG(INFO) << "Building mini graph";
    g.mini_graph = std::make_shared<Graph>(mini_n, mini_xadj, mini_adjncy);
    g.mini_graph->node_weights = std::vector<int>(mini_node_weights, mini_node_weights + mini_n);
    g.mini_graph->edge_weights = std::vector<int>(mini_edge_weights, mini_edge_weights + mini_xadj[mini_n]);
    g.gid2mid = std::vector<int>(mini_gid2mid, mini_gid2mid + n);
  }
  
  PrePartitionInfo pre_res = CommScheduler::PrePartitionGraph(n_peers, g);

  auto prepart_end = system_clock::now();
  std::cout << "Prepartition took" << duration_cast<seconds>(prepart_end - begin).count() << "s\n" << std::endl;

  BinStream stream;
  pre_res.serialize(stream);

  *bin_stream_size = stream.size();
  CopyVectorToRawPtr(bin_stream_data, stream.get_buffer_vector());

  auto serialize_end = system_clock::now();
  std::cout << "Serialize took" << duration_cast<seconds>(serialize_end - prepart_end).count() << "s\n" << std::endl;
  std::cout << "Prepartition all took" << duration_cast<seconds>(serialize_end - begin).count() << "s\n" << std::endl;
}

void PartitionGraph(gcclComm_t comm, int n, int *xadj, int *adjncy,
  gcclCommInfo_t *info, int *sgn, int **sg_xadj,
  int **sg_adjncy) {
  PartitionGraph(comm, n, xadj, adjncy, info, sgn, sg_xadj, sg_adjncy, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void PartitionGraph(gcclComm_t comm, const char *cached_dir,
  gcclCommInfo_t *info, int *sgn, int **sg_xadj,
  int **sg_adjncy) {
  PartitionGraph(comm, cached_dir, info, sgn, sg_xadj, sg_adjncy, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void GraphDetailedInfo(gcclComm_t comm, int **gid2pid, int **num_local_nodes, int **gid2lid_unordered) {
  auto *comm_sch = comm->GetCommScheduler();
  comm_sch->GraphDetailedInfo(gid2pid, num_local_nodes, gid2lid_unordered);
}

void InitLogs(const char *name) {
  google::InitGoogleLogging(name);
  LOG(INFO) << "Init logs name is " << name;
}

void DeInitLogs() {
  LOG(INFO) << "DeInitLogs is called!";
  google::ShutdownGoogleLogging();
}

int GetDeviceId(gcclComm_t comm) { return comm->GetCoordinator()->GetDevId(); }

int GetLocalNNodes(gcclComm_t comm) {
  auto *comm_sch = comm->GetCommScheduler();
  return comm_sch->GetLocalNNodes();
}
void FreeCommInfo(gcclCommInfo_t info) {
  delete info;
}

void DispatchFloat(gcclComm_t comm, float *data, int feat_size,
                   int local_n_nodes, float *local_data, int no_remote) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  comm_sch->DispatchData(coor, (char *)data, (size_t)feat_size, sizeof(float),
                         (size_t)local_n_nodes, (char *)local_data, no_remote);
}

void DispatchInt(gcclComm_t comm, int *data, int feat_size, int local_n_nodes,
                 int *local_data, int no_remote) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  comm_sch->DispatchData(coor, (char *)data, (size_t)feat_size, sizeof(int),
                         (size_t)local_n_nodes, (char *)local_data, no_remote);
}

void SetConfig(gcclComm_t comm, const std::string &config_json) {
  SetConfigInternal(comm->GetConfig(), config_json);
}

}  // namespace gccl
