
#include "comm/comm_scheduler.h"

#include <algorithm>
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

using namespace std::chrono;

namespace gccl {
void CommScheduler::BuildMappingsTransfer( // AndSubgraph
        const Graph &g,
        const std::vector<std::map<int, int>> &local_mappings,
        const std::vector<int> &parts,
        int n_parts) {
    all_local_graph_infos_.resize(n_parts);
    std::vector<std::vector<int>> local_nodes(n_parts);
    std::vector<std::vector<int>> remote_nodes(n_parts);
    for (int i = 0; i < g.n_nodes; ++i) {
        local_nodes[parts[i]].push_back(i);
    }

    // * begin build transfer request
    TransferRequest req;
    req.req_ids.resize(n_parts, std::vector<std::vector<int>>(n_parts));
    // * end build transfer request

    auto edge_func = [&parts, &remote_nodes](int u, int v) {
        int bu = parts[u];
        int bv = parts[v];
        if (bu != bv) {
            remote_nodes[bv].push_back(u);
            // * begin build transfer request
            req.req_ids[bu][bv].push_back(u);
            // * end build transfer request
        }
    };
    g.ApplyEdge(edge_func);
    for (auto &vec : remote_nodes) {
        UniqueVec(vec);
    }
    // * begin build transfer request
    for (int i = 0; i < nparts; ++i) {
        for (int j = 0; j < nparts; ++j) {
            UniqueVec(req.req_ids[i][j]);
        }
    }
    // * end build transfer request
    std::vector<std::map<int, int>> mappings(n_parts);
    int remote_cnt = 0;
    for (const auto &r : remote_nodes) {
        remote_cnt += r.size();
    }
    DLOG(INFO) << "Number of remote nodes is " << remote_cnt;
    for (int i = 0; i < n_parts; ++i) {
        int cnt = 0;
        for (auto u : local_nodes[i]) {
            mappings[i][u] = cnt++;
        }
        all_local_graph_infos_[i].n_local_nodes = cnt;
        for (auto u : remote_nodes[i]) {
            mappings[i][u] = cnt++;
        }
        all_local_graph_infos_[i].n_nodes = cnt;
    }
    local_mappings_ = std::move(mappings);
}


}