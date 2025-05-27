#pragma once

#include <vector>
#include <map>

#include "comm/comm_info.h"
#include "config.h"
#include "conn/connection.h"
#include "coordinator.h"
#include "graph.h"


namespace gccl {

class PrePartitionInfo {
  public:
    int n_peers;
    Graph graph;
    std::vector<int> pre_parts;
    std::vector<LocalGraphInfo> pre_all_local_graph_infos;
    std::vector<std::map<int, int>> pre_local_mappings;
    TransferRequest pre_requests;
    std::vector<Graph> pre_subgraphs;


    BinStream &serialize(BinStream &stream) const {
      stream << n_peers;
      stream << graph;
      stream << pre_parts;
      stream << pre_all_local_graph_infos;
      stream << pre_local_mappings;
      stream << pre_requests;
      stream << pre_subgraphs;
      return stream;
    }

    BinStream &deserialize(BinStream &stream) {
      stream >> n_peers;
      stream >> graph;
      stream >> pre_parts;
      stream >> pre_all_local_graph_infos;
      stream >> pre_local_mappings;
      stream >> pre_requests;
      stream >> pre_subgraphs;
      return stream;
    }
};
  

class CommScheduler {
 public:
  const std::vector<std::map<int, int>> &GetLocalMappings() const {
    return local_mappings_;
  }
  int GetLocalNNodes() const { return my_local_graph_info_.n_local_nodes; }
  const std::vector<std::shared_ptr<CommPattern>> &GetCommPatterns() const {
    return comm_patterns_;
  }

  void BuildPartitionInfo(Coordinator *coor, Config *config, Graph &g,
                          const std::string &graph_dir, CommInfo **info,
                          int *sgn, int **sg_xadj, int **sg_adjncy, bool use_prepart, BinStream &bin_stream);

  void BuildPartitionInfo(Coordinator *coor, Config *config, Graph &g,
                          const std::string &graph_dir, CommInfo **info,
                          int *sgn, int **sg_xadj, int **sg_adjncy) {
    BinStream bs;
    BuildPartitionInfo(coor, config, g, graph_dir, info, sgn, sg_xadj, sg_adjncy, false, bs);
  }

  void BuildLocalMappings(Graph &g, int nparts, const std::vector<int> &parts);

  static TransferRequest BuildTransferRequest(Graph &g, int nparts,
                                       const std::vector<int> &parts);

  std::vector<TransferRequest> AllocateRequestToBlock(
      const TransferRequest &all_req, int n_parts, int n_blocks);
  CommInfo *ScatterCommInfo(Coordinator *coordinator,
                            const std::vector<CommInfo> &infos);

  static std::vector<Graph> BuildSubgraphs(
      const Graph &g, const std::vector<std::map<int, int>> &local_mappings,
      const std::vector<int> &parts, int nparts);
  
  static std::vector<Graph> BuildSubgraphsPara(
    const Graph &g, const std::vector<std::map<int, int>> &local_mappings,
    const std::vector<int> &parts, int nparts);

  void DispatchData(Coordinator *coor, char *data, size_t feat_size, size_t data_size,
                    int local_n_nodes, char *local_data, int no_remote);

  void LoadCachedPartition(Coordinator *coor, const std::string &dir, int *sgn,
                           int **sg_xadj, int **sg_adjncy);
  
  void GraphDetailedInfo(int **gid2pid, int **num_local_nodes, int **gid2lid_unordered);

  static PrePartitionInfo PrePartitionGraph(int n_peers, Graph &graph);

  void LoadPrepartResult(Coordinator *coor, BinStream &bin_stream, Graph &g,
    int *sgn, int **sg_xadj, int **sg_adjncy);

 private:
  void ReadCachedState(const std::string &part_dir, int rank, bool is_root);
  void WriteCachedState(Coordinator* coor, const std::string &part_dir, int rank, bool is_root);
  void ScatterLocalGraphInfos(Coordinator *coor);
  void PartitionGraph(Coordinator *coor, Graph &g, const std::string &dir,
                      int n_parts, int *sgn, int **sg_xadj, int **sg_adjncy);
  static std::pair<std::vector<LocalGraphInfo>, std::vector<std::map<int, int>>> BuildLocalMappingsInternal(
    Graph &g, int nparts, const std::vector<int> &parts);
  

  std::vector<std::shared_ptr<CommPattern>> comm_patterns_;

  std::vector<std::map<int, int>> local_mappings_;  // All
  std::vector<int> parts_;                          // All
  TransferRequest requests_;
  std::vector<LocalGraphInfo> all_local_graph_infos_;  // All
  LocalGraphInfo my_local_graph_info_;                 // Mine
  Graph my_graph_;                                     // Mine
  std::map<int, int> my_local_mapping_;                // Mine
};

}  // namespace gccl
