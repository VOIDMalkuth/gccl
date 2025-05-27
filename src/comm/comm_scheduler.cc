#include "comm/comm_scheduler.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <chrono>
#include <iostream>

#include "glog/logging.h"
#include "nlohmann/json.hpp"

#include "base/bin_stream.h"
#include "conn/connection.h"
#include "gpu/common.h"
#include "graph.h"
#include "param.h"
#include "partitioner.h"
#include "topo/dev_graph.h"
#include "utils.h"
#include "json_utils.h"

#include<iostream>
#include <utility>

using namespace std::chrono;

namespace gccl {

void BuildAllgatherCommScheme(
    std::vector<CommInfo> &infos, Config *config,
    const std::vector<std::shared_ptr<CommPattern>> &patterns,
    const std::vector<std::map<int, int>> &local_mappings,
    const std::vector<TransferRequest> &per_block_reqs, int n_parts,
    int n_blocks) {
  
  for (int i = 0; i < n_blocks; ++i) {
    auto pattern_infos = patterns[i]->BuildCommPatternInfos(
      config, local_mappings, per_block_reqs[i], n_parts);
    for (int j = 0; j < n_parts; ++j) {
      infos[j].allgather_scheme.comm_pattern_infos[i] = pattern_infos[j];
    }
  }
  
  
  for (int i = 0; i < n_parts; ++i) {
    infos[i].allgather_scheme.n_blocks = n_blocks;
  }
}

// * 1. 对于每一个边，如果其邻居都在分区内，则加入这个边
// * 2. 对于每一个边，如果其邻居有一个不在分区内，则加入这个边，并记录这个边的两个节点，记为cross边，需要接受这个节点的图加入这个边
// todo: parallelize this, possible with help of mini graph
std::vector<Graph> CommScheduler::BuildSubgraphs(
    const Graph &g, const std::vector<std::map<int, int>> &local_mappings,
    const std::vector<int> &parts, int nparts) {
  std::vector<std::vector<std::pair<int, int>>> edges(nparts);
  std::vector<Graph> sgs;
  int n_cross_edges = 0;
  auto func = [&n_cross_edges, &local_mappings, &parts, &edges](int u, int v) {
    int pu = parts[u];
    int pv = parts[v];
    int lu = local_mappings[pu].at(u);
    int lv = local_mappings[pv].at(v);
    if (pu == pv) {
      edges[pu].push_back({lu, lv});
    } else {
      n_cross_edges++;
      int remote_u = local_mappings[pv].at(u);
      edges[pv].push_back({remote_u, lv});
    }
  };
  g.ApplyEdge(func);
  for (auto &part_edge : edges) {
    sgs.push_back(Graph(part_edge));
  }
  LOG(INFO) << "Number of cross edges is " << n_cross_edges;
  return sgs;
}

std::vector<Graph> CommScheduler::BuildSubgraphsPara(
    const Graph &g, const std::vector<std::map<int, int>> &local_mappings,
    const std::vector<int> &parts, int nparts) {
  std::vector<std::vector<std::pair<int, int>>> edges(nparts);
  std::vector<Graph> sgs;
  int n_cross_edges = 0;

  const int chunks = 128;
  const int chunk_size = (g.n_nodes + chunks - 1) / chunks;
  std::vector<std::vector<std::vector<std::pair<int, int>>>> edges_chunk(chunks, std::vector<std::vector<std::pair<int, int>>>(nparts));

  #pragma omp parallel for schedule(dynamic) shared(g, parts, local_mappings, edges, edges_chunk, nparts, chunks, chunk_size) reduction(+:n_cross_edges)
  for (int oi = 0; oi < chunks; ++oi) {
    for (int ii = 0; ii < chunk_size; ii++) {
      int i = oi * chunk_size + ii;
      if (i >= g.n_nodes) {
        break;
      }
      int u = i;
      int pu = parts[u];
      int lu = local_mappings[pu].at(u);
      for (int j = g.xadj[i]; j < g.xadj[i + 1]; ++j) {  
        int v = g.adjncy[j];
        int pv = parts[v];
        int lv = local_mappings[pv].at(v);
        
        if (pu == pv) {
          edges_chunk[oi][pv].push_back({lu, lv});
        } else {
          n_cross_edges += 1;
          int remote_u = local_mappings[pv].at(u);
          edges_chunk[oi][pv].push_back({remote_u, lv});
        }
      }
    }
  }

  std::cout << "[info] edges_chunk finished" << std::endl;

  #pragma omp parallel for schedule(dynamic) shared(edges, edges_chunk, nparts, chunks)
  for (int j = 0; j < nparts; j++) {
    for (int i = 0; i < chunks; i++) {
      edges[j].insert(edges[j].end(), edges_chunk[i][j].begin(), edges_chunk[i][j].end());
    }
  }

  std::cout << "[info] edges insert finished" << std::endl;

  sgs.resize(nparts);
  #pragma omp parallel for shared(edges, nparts, sgs)
  for (int i = 0; i < nparts; i++) {
    sgs[i] = Graph(edges[i]);
  }

  std::cout << "[info] graph creation finished" << std::endl;

  LOG(INFO) << "Number of cross edges is " << n_cross_edges;
  return sgs;
}

void GetConnPeers(std::vector<CommInfo> &infos, Config *config, int n_peers) {
  std::shared_ptr<DevGraph> dev_graph;
  if (config->dev_graph_file.size() == 0) {
    dev_graph = std::make_shared<DevGraph>(DefaultDevGraph(n_peers));
  } else {
    dev_graph = std::make_shared<DevGraph>(config->dev_graph_file);
  }
  std::vector<std::vector<int>> conn_peers(n_peers);
  for (int i = 0; i < n_peers; ++i) {
    for (int j = 0; j < n_peers; ++j) {
      if (dev_graph->GetTransportLevel(i, j) <=
              StringToTransportLevel(config->transport_level) &&
          i != j) {
        conn_peers[i].push_back(j);
      }
    }
    infos[i].n_conn_peers = conn_peers[i].size();
    CopyVectorToRawPtr(&infos[i].conn_peers, conn_peers[i]);
  }
}

void CommScheduler::BuildPartitionInfo(Coordinator *coor, Config *config,
                                       Graph &g, const std::string &graph_dir,
                                       CommInfo **info, int *sgn, int **sg_xadj,
                                       int **sg_adjncy, bool use_prepart, BinStream &bs) {
  // Build info
  // Scatter info
  // Exchange info
  int n_peers = coor->GetNPeers();
  int rank = coor->GetRank();
  int n_blocks = config->n_blocks;
  int dev_id = coor->GetDevId();
  const auto &peer_infos = coor->GetPeerInfos();

  bool read_cache =
      graph_dir.size() > 0 &&
      CheckDirExists(graph_dir + "/part-" + std::to_string(n_peers));
  coor->Barrier();

  if (read_cache) {
    LOG(ERROR) << "Doesn't support reading cached partition for now"; // later: implement this
    assert(false);
    LoadCachedPartition(coor, graph_dir, sgn, sg_xadj, sg_adjncy);
  } else if (use_prepart) {
    LoadPrepartResult(coor, bs, g, sgn, sg_xadj, sg_adjncy);
  } else {
    PartitionGraph(coor, g, graph_dir, n_peers, sgn, sg_xadj, sg_adjncy);
  }

  std::vector<CommInfo> infos;
  std::vector<Graph> subgraphs;

  GCCLSetCudaDevice(dev_id);

  std::vector<ConnType> conn_type(n_peers);
  for (int i = 0; i < n_peers; ++i) {
    conn_type[i] = GetConnType(peer_infos[rank], peer_infos[i]);
  }

  // Build patterns
  CommPatternType type;
  if (config->comm_pattern == "RING") {
    type = CommPatternType::Ring;
    LOG(INFO) << "Using Ring communication";
  } else if (config->comm_pattern == "ALLTOALL") {
    type = CommPatternType::AllToAll;
    LOG(INFO) << "Using all to all communication";
  } else if (config->comm_pattern == "GREEDY") {
    type = CommPatternType::Greedy;
    LOG(INFO) << "Using greedy communication";
  } else {
    CHECK(false);
  }
  comm_patterns_.clear();
  for (int i = 0; i < n_blocks; ++i) {
    comm_patterns_.push_back(
        CommPatternFactory::GetCommPattern(type, config->rings[i], conn_type));
  }

  if (coor->IsRoot()) {
    auto t0 = GetTime();
    // Root build infos
    infos = std::vector<CommInfo>(n_peers);
    auto per_block_reqs = AllocateRequestToBlock(requests_, n_peers, n_blocks);
    auto t1 = GetTime();
    BuildAllgatherCommScheme(infos, config, comm_patterns_, local_mappings_,
                             per_block_reqs, n_peers, n_blocks);
    auto t2 = GetTime();
    LOG(INFO) << "Using time to allocalte req " << TimeDiff(t0, t1) << " build allgather comm " << TimeDiff(t1, t2);
    LOG(INFO) << "Using time for SPST algorithm: " << TimeDiff(t1, t2) << " ms";
    std::cout << "Using time to allocalte req " << TimeDiff(t0, t1) << " build allgather comm " << TimeDiff(t1, t2) << std::endl;
    std::cout << "Using time for SPST algorithm: " << TimeDiff(t1, t2) << " ms" << std::endl;

    GetConnPeers(infos, config, n_peers);
  }
  auto t0 = GetTime();
  *info = ScatterCommInfo(coor, infos);
  auto t1 = GetTime();
  std::cout << "Using time for ScatterCommInfo: " << TimeDiff(t0, t1) << " ms" << std::endl;
  (*info)->Print();
  CommInfoSetupConnection(*info, coor, comm_patterns_);
  auto t2 = GetTime();
  std::cout << "Using time for CommInfoSetupConnection: " << TimeDiff(t1, t2) << " ms" << std::endl;
  (*info)->CopyGraphInfoToDev();
  CopyCommInfoToDev(*info);
  LOG(INFO) << "Using time to scatter comm info: " << TimeDiff(t0, t1);
  LOG(INFO) << "Using time to setup conn: " << TimeDiff(t0, t1);
  LOG(INFO) << "Build partition info done";
  google::FlushLogFiles(google::GLOG_INFO);
}

std::vector<std::map<int, int>> ReadLocalMappings(const json &j) {
  std::vector<std::map<int, int>> ret;
  j.at("local_mappings").get_to(ret);
  return ret;
}

void CommScheduler::ReadCachedState(const std::string &part_dir, int rank,
                                    bool is_root) {
  if (is_root) {
    auto root_info_file = part_dir + "root.json";
    json root_info = ReadJsonFromFile(root_info_file);
    root_info.at("local_mappings").get_to(local_mappings_);
    root_info.at("requests").get_to(requests_);
    root_info.at("parts").get_to(parts_);
  }
  auto my_graph_file = part_dir + "subgraph-" + std::to_string(rank) + ".txt";
  auto local_info_file = part_dir + "local-" + std::to_string(rank) + ".json";
  json local_info = ReadJsonFromFile(local_info_file);
  local_info.at("local_graph_info").get_to(my_local_graph_info_);
  my_graph_ = Graph(my_graph_file);
}

void CommScheduler::WriteCachedState(Coordinator* coor, const std::string &part_dir, int rank,
                                     bool is_root) {
  if (part_dir.size() == 0) return;
  LOG(INFO) << "Writing cache....";
  if (is_root) {
    CreateDir(part_dir);
    auto root_info_file = part_dir + "root.json";
    json root_info;
    root_info["local_mappings"] = local_mappings_;
    root_info["requests"] = requests_;
    root_info["parts"] = parts_;
    WriteJsonToFile(root_info, root_info_file);
  }
  coor->Barrier();
  while(!CheckDirExists(part_dir)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    sched_yield();
  }
  auto my_graph_file = part_dir + "subgraph-" + std::to_string(rank) + ".txt";
  auto local_info_file = part_dir + "local-" + std::to_string(rank) + ".json";
  json local_info;
  local_info["local_graph_info"] = my_local_graph_info_;
  WriteJsonToFile(local_info, local_info_file);
  my_graph_.WriteToFile(my_graph_file);
  LOG(INFO) << "Finished writing cache....";
}

void CommScheduler::LoadCachedPartition(Coordinator *coor,
                                        const std::string &dir, int *sgn,
                                        int **sg_xadj, int **sg_adjncy) {
  LOG(INFO) << "Loading cached dir";
  int n_peers = coor->GetNPeers();
  auto part_dir = dir + "/part-" + std::to_string(n_peers) + "/";
  ReadCachedState(part_dir, coor->GetRank(), coor->IsRoot());
  BuildRawGraph(my_graph_, sgn, sg_xadj, sg_adjncy);
  LOG(INFO) << "Finished loading.";
  LOG(INFO) << " # Subgraph nodes: " << my_graph_.n_nodes;
  LOG(INFO) << " # Subgraph edges: " << my_graph_.n_edges;
  LOG(INFO) << " # Subgraph local nodes: " << GetLocalNNodes();
}

void CommScheduler::PartitionGraph(Coordinator *coor, Graph &g,
                                   const std::string &dir, int n_parts,
                                   int *sgn, int **sg_xadj, int **sg_adjncy) {
  auto graph_file = dir + "/graph.txt";
  auto part_dir = dir + "/part-" + std::to_string(n_parts) + "/";
  int rank = coor->GetRank();
  int n_peers = coor->GetNPeers();
  std::vector<Graph> subgraphs;
  if (coor->IsRoot()) {
    Graph graph;
    if (dir.size() == 0) {
      graph = g;
    } else {
      graph = Graph(graph_file);
    }

    auto part_begin = system_clock::now();

    parts_ = PartitionGraphInternal(graph, n_peers);
    
    auto part_done = system_clock::now();
    std::cout << ">>Part took " << duration_cast<milliseconds>(part_done - part_begin).count() << "ms\n";
    
    BuildLocalMappings(graph, n_peers, parts_);
    auto build_map_done = system_clock::now();
    std::cout << ">>Build map took " << duration_cast<milliseconds>(build_map_done - part_done).count() << "ms\n";

    requests_ = BuildTransferRequest(graph, n_peers, parts_);
    auto build_request_done = system_clock::now();
    std::cout << ">>Build request took " << duration_cast<milliseconds>(build_request_done - build_map_done).count() << "ms\n";

    subgraphs = BuildSubgraphsPara(graph, local_mappings_, parts_, n_peers);
    auto build_subgraph_done = system_clock::now();
    std::cout << ">>Build subgraph took " << duration_cast<milliseconds>(build_subgraph_done - build_request_done).count() << "ms\n";
  }
  auto scatter_begin = system_clock::now();
  my_graph_ = coor->Scatter(subgraphs);
  auto scatter_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Scatter subgraph took " << duration_cast<milliseconds>(scatter_done - scatter_begin).count() << "ms\n";
  BuildRawGraph(my_graph_, sgn, sg_xadj, sg_adjncy);
  auto build_raw_graph_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Build raw graph took " << duration_cast<milliseconds>(build_raw_graph_done - scatter_done).count() << "ms\n";
  my_local_graph_info_ = coor->Scatter(all_local_graph_infos_);
  auto scatter_local_graph_info_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Scatter local graph took " << duration_cast<milliseconds>(scatter_local_graph_info_done - build_raw_graph_done).count() << "ms\n";
  WriteCachedState(coor, dir.size() == 0 ? "" : part_dir, rank, coor->IsRoot());
  auto write_cached_state_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Write cached state took " << duration_cast<milliseconds>(write_cached_state_done - scatter_local_graph_info_done).count() << "ms\n";

  auto my_local_mapping_scatter_begin = system_clock::now();
  my_local_mapping_ = coor->Scatter(local_mappings_);
  auto my_local_mapping_scatter_done = system_clock::now();
  if (coor->IsRoot())
    std::cout << ">>Scatter local_mapping took " << duration_cast<milliseconds>(my_local_mapping_scatter_done - my_local_mapping_scatter_begin).count() << "ms\n";
}

void CommScheduler::GraphDetailedInfo(int **gid2pid, int **num_local_nodes, int **gid2lid_unordered) {  
  CopyVectorToRawPtr(gid2pid, parts_);
  std::vector<int> gid2lid_unordered_vec;
  std::vector<int> num_local_nodes_vec;
  for (int i = 0; i < local_mappings_.size(); i++) {
    num_local_nodes_vec.push_back(local_mappings_[i].size());
    // 只写local node的信息，否则remote的会盖住正确的local
    auto graph_info = all_local_graph_infos_[i];
    for (auto &pair : local_mappings_[i]) {
      if (pair.second >= graph_info.n_local_nodes) {
        continue;
      }
      gid2lid_unordered_vec.push_back(pair.first);
      gid2lid_unordered_vec.push_back(pair.second);
    }
  }
  CopyVectorToRawPtr(num_local_nodes, num_local_nodes_vec);
  CopyVectorToRawPtr(gid2lid_unordered, gid2lid_unordered_vec);
}

// Build local_mappings_ and all_local_graph_info_
// * 对于每一个节点
// * 1. 将其放到分区的local_nodes里
// * 2. 对于每一个邻居节点，如果邻居节点不在同一个分区，将其放到remote_nodes里，并去重
// * 3. 对于每一个分区，将local_nodes和remote_nodes合并，得到一个分区的所有节点，并记录原始节点与新分区内节点的编号关系
std::pair<std::vector<LocalGraphInfo>, std::vector<std::map<int, int>>>
CommScheduler::BuildLocalMappingsInternal(Graph &g, int n_parts,
                                       const std::vector<int> &parts) {
  auto total_start_time = std::chrono::high_resolution_clock::now();
  
  std::vector<LocalGraphInfo> all_local_graph_infos(n_parts);
  std::vector<std::vector<int>> local_nodes(n_parts);
  std::vector<std::vector<int>> remote_nodes(n_parts);
  
  // Part 1: Assign nodes to local partitions
  auto part1_start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < g.n_nodes; ++i) {
    local_nodes[parts[i]].push_back(i);
  }
  auto part1_end_time = std::chrono::high_resolution_clock::now();
  auto part1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(part1_end_time - part1_start_time).count();
  std::cout << "\tPart 1 (Assign nodes to local partitions): " << part1_duration << " ms\n";
  
  // Part 2: Find remote nodes for each partition
  auto part2_start_time = std::chrono::high_resolution_clock::now();
  auto edge_func = [&parts, &remote_nodes](int u, int v) {
    int bucket_u = parts[u];
    int bucket_v = parts[v];
    if (bucket_u != bucket_v) {
      remote_nodes[bucket_v].push_back(u);
    }
  };
  g.ApplyEdge(edge_func);
  auto part2_end_time = std::chrono::high_resolution_clock::now();
  auto part2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(part2_end_time - part2_start_time).count();
  std::cout << "\tPart 2 (Find remote nodes): " << part2_duration << " ms\n";

  // Part 3: Remove duplicates from remote nodes
  auto part3_start_time = std::chrono::high_resolution_clock::now();
  for (auto &vec : remote_nodes) {
    UniqueVec(vec);
  }
  auto part3_end_time = std::chrono::high_resolution_clock::now();
  auto part3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(part3_end_time - part3_start_time).count();
  std::cout << "\tPart 3 (Remove duplicates): " << part3_duration << " ms\n";
  
  // Part 4: Build mappings
  auto part4_start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::map<int, int>> mappings(n_parts);
  int remote_cnt = 0;
  for (const auto &r : remote_nodes) {
    remote_cnt += r.size();
  }
  DLOG(INFO) << "Number of remote nodes is " << remote_cnt;
  
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_parts; ++i) {
    int cnt = 0;
    for (auto u : local_nodes[i]) {
      mappings[i][u] = cnt++;
    }
    all_local_graph_infos[i].n_local_nodes = cnt;
    for (auto u : remote_nodes[i]) {
      mappings[i][u] = cnt++;
    }
    all_local_graph_infos[i].n_nodes = cnt;
  }
  auto part4_end_time = std::chrono::high_resolution_clock::now();
  auto part4_duration = std::chrono::duration_cast<std::chrono::milliseconds>(part4_end_time - part4_start_time).count();
  std::cout << "\tPart 4 (Build mappings): " << part4_duration << " ms\n";
  
  // Total time
  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();
  std::cout << "Total BuildLocalMappingsInternal time: " << total_duration << " ms\n";
  
  return {all_local_graph_infos, mappings};
}

void CommScheduler::BuildLocalMappings(Graph &g, int n_parts,
  const std::vector<int> &parts) {
  std::tie(all_local_graph_infos_, local_mappings_) = BuildLocalMappingsInternal(g, n_parts, parts);
}

// * 1. 对于每一个节点，如果其邻居节点不在同一个分区，将其加入到req_ids中 （u->v单向）
// * 2. 对于每一个分区，去重
TransferRequest CommScheduler::BuildTransferRequest(
    Graph &g, int nparts, const std::vector<int> &parts) {
  TransferRequest req;
  req.req_ids.resize(nparts, std::vector<std::vector<int>>(nparts));
  auto edge_func = [&req, &parts](int u, int v) {
    int bu = parts[u];
    int bv = parts[v];
    if (bu != bv) {
      req.req_ids[bu][bv].push_back(u);
    }
  };
  g.ApplyEdge(edge_func);
  for (int i = 0; i < nparts; ++i) {
    for (int j = 0; j < nparts; ++j) {
      UniqueVec(req.req_ids[i][j]);
    }
  }
  return req;
}
// <Node id, device id>, [Req dev list]
std::vector<std::pair<std::pair<int, int>, std::vector<int>>> BuildNodeReqs(
    const TransferRequest &all_req, int n_parts) {
  std::vector<std::pair<std::pair<int, int>, std::vector<int>>> ret;
  for (int i = 0; i < n_parts; ++i) {
    std::map<int, std::vector<int>> node_to_reqs;
    for (int j = 0; j < n_parts; ++j) {
      for (auto v : all_req.req_ids[i][j]) {
        node_to_reqs[v].push_back(j);
      }
    }
    for (const auto &pair : node_to_reqs) {
      int v = pair.first;
      const auto &req_list = pair.second;
      ret.push_back({{v, i}, req_list});
    }
  }
  return ret;
}

std::vector<TransferRequest> RandAllocateRequests(
    const TransferRequest &all_req, int n_parts, int n_blocks) {
  auto nodes_reqs = BuildNodeReqs(all_req, n_parts);
  std::vector<int> ids(nodes_reqs.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::random_shuffle(ids.begin(), ids.end());
  std::vector<TransferRequest> ret(n_blocks);
  for (int i = 0; i < n_blocks; ++i) {
    ret[i].req_ids = Build3DVector<int>(n_parts, n_parts);
  }
  for (int i = 0; i < nodes_reqs.size(); ++i) {
    int node_id = nodes_reqs[ids[i]].first.first;
    int dev_id = nodes_reqs[ids[i]].first.second;
    const auto &req_list = nodes_reqs[ids[i]].second;
    int bid = i % n_blocks;
    for (auto req_id : req_list) {
      ret[bid].req_ids[dev_id][req_id].push_back(node_id);
    }
  }
  return ret;
}

std::vector<TransferRequest> CommScheduler::AllocateRequestToBlock(
    const TransferRequest &all_req, int n_parts, int n_blocks) {
  // TODO algorithm to allocate requests
  return RandAllocateRequests(all_req, n_parts, n_blocks);
}

CommInfo *CommScheduler::ScatterCommInfo(Coordinator *coordinator,
                                         const std::vector<CommInfo> &infos) {
  if (coordinator->IsRoot()) {
    std::vector<std::shared_ptr<BinStream>> infos_binstream;
    for (const auto &info : infos) {
      std::shared_ptr<BinStream> stream = std::make_shared<BinStream>();
      info.serialize(*stream);
      infos_binstream.emplace_back(std::move(stream));
    }
    for (int i = 0; i < coordinator->GetNPeers(); ++i) {
      coordinator->SendBinstreamTo(i, infos_binstream[i]);
    }
  }
  auto myinfo_binstream = coordinator->RecvBinstreamFromRoot();
  CommInfo *myinfo = new CommInfo;
  myinfo->deserialize(*myinfo_binstream);
  return myinfo;
}

void CommScheduler::DispatchData(Coordinator *coor, char *data, size_t feat_size,
                                 size_t data_size, int local_n_nodes,
                                 char *local_data, int no_remote) {
  std::vector<std::vector<char>> vecs;
  size_t record_size = feat_size * data_size;  // in bytes
  if (coor->IsRoot()) {
    for (int i = 0; i < local_mappings_.size(); ++i) {
      const auto &mp = local_mappings_[i];
      int sub_local_n_nodes = all_local_graph_infos_[i].n_local_nodes;

      int sub_n_nodes = mp.size();
      
      std::vector<char> char_data(sub_n_nodes * record_size, 0);
      for (const auto &pair : mp) {
        // u -> v
        int u = pair.first;
        int v = pair.second;
        CHECK(v < sub_n_nodes);
        if (no_remote && v >= sub_local_n_nodes) continue;
        memcpy(char_data.data() + v * record_size, data + u * record_size,
               record_size);
      }
      vecs.emplace_back(std::move(char_data));
    }
  }

  auto my_data = coor->Scatter(vecs);
  CHECK_EQ(my_data.size(), local_n_nodes * record_size);
  memcpy(local_data, my_data.data(), local_n_nodes * record_size);
}

void CommScheduler::ScatterLocalGraphInfos(Coordinator *coor) {
  my_local_graph_info_ = coor->Scatter(all_local_graph_infos_);
}

PrePartitionInfo CommScheduler::PrePartitionGraph(int n_peers, Graph &graph) {
  auto part_begin = system_clock::now();
  std::vector<int> pre_parts = PartitionGraphInternal(graph, n_peers);
  auto part_done = system_clock::now();
  std::cout << ">>PrePart took " << duration_cast<milliseconds>(part_done - part_begin).count() << "ms\n";
  
  std::vector<LocalGraphInfo> pre_all_local_graph_infos;
  std::vector<std::map<int, int>> pre_local_mappings;
  std::tie(pre_all_local_graph_infos, pre_local_mappings) = BuildLocalMappingsInternal(graph, n_peers, pre_parts);
  auto build_map_done = system_clock::now();
  std::cout << ">>PreBuild map took " << duration_cast<milliseconds>(build_map_done - part_done).count() << "ms\n";

  TransferRequest pre_requests = BuildTransferRequest(graph, n_peers, pre_parts);
  auto build_request_done = system_clock::now();
  std::cout << ">>PreBuild request took " << duration_cast<milliseconds>(build_request_done - build_map_done).count() << "ms\n";

  std::vector<Graph> pre_subgraphs = BuildSubgraphsPara(graph, pre_local_mappings, pre_parts, n_peers);
  auto build_subgraph_done = system_clock::now();
  std::cout << ">>PreBuild subgraph took " << duration_cast<milliseconds>(build_subgraph_done - build_request_done).count() << "ms\n";

  PrePartitionInfo pre_part_info;
  pre_part_info.n_peers = n_peers;
  std::string repart_opt = GetEnvParam("REPART_OPT", std::string("FALSE"));
  if (repart_opt == "TRUE") {
    pre_part_info.graph = graph;
  }
  pre_part_info.pre_parts = pre_parts;
  pre_part_info.pre_all_local_graph_infos = pre_all_local_graph_infos;
  pre_part_info.pre_local_mappings = pre_local_mappings;
  pre_part_info.pre_requests = pre_requests;
  pre_part_info.pre_subgraphs = pre_subgraphs;
  return pre_part_info;
}

void CommScheduler::LoadPrepartResult(Coordinator *coor, BinStream &bin_stream, Graph &g,
                                   int *sgn, int **sg_xadj, int **sg_adjncy) {
  int rank = coor->GetRank();
  int n_peers = coor->GetNPeers();
  std::vector<Graph> subgraphs;
  if (coor->IsRoot()) {
    int prev_n_peers;
    bin_stream >> prev_n_peers;
    assert(prev_n_peers == n_peers);
    std::string repart_opt = GetEnvParam("REPART_OPT", std::string("FALSE"));
    if (repart_opt == "TRUE") {
      bin_stream >> g;
    } else {
      Graph new_g;
      bin_stream >> new_g;
    }
    bin_stream >> parts_;
    bin_stream >> all_local_graph_infos_;
    bin_stream >> local_mappings_;
    bin_stream >> requests_;
    bin_stream >> subgraphs;
  }

  auto scatter_begin = system_clock::now();
  my_graph_ = coor->Scatter(subgraphs);
  auto scatter_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Scatter subgraph took " << duration_cast<milliseconds>(scatter_done - scatter_begin).count() << "ms\n";
  BuildRawGraph(my_graph_, sgn, sg_xadj, sg_adjncy);
  auto build_raw_graph_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Build raw graph took " << duration_cast<milliseconds>(build_raw_graph_done - scatter_done).count() << "ms\n";
  my_local_graph_info_ = coor->Scatter(all_local_graph_infos_);
  auto scatter_local_graph_info_done = system_clock::now();
  if (coor->IsRoot())
  	std::cout << ">>Scatter local graph took " << duration_cast<milliseconds>(scatter_local_graph_info_done - build_raw_graph_done).count() << "ms\n";

  auto my_local_mapping_scatter_begin = system_clock::now();
  my_local_mapping_ = coor->Scatter(local_mappings_);
  auto my_local_mapping_scatter_done = system_clock::now();
  if (coor->IsRoot())
    std::cout << ">>Scatter local_mapping took " << duration_cast<milliseconds>(my_local_mapping_scatter_done - my_local_mapping_scatter_begin).count() << "ms\n";
}

}  // namespace gccl
