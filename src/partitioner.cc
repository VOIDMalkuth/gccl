#include "partitioner.h"

#include <stddef.h>
#include <stdio.h>
#include <set>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "glog/logging.h"
#include "metis.h"
#include "param.h"
#include "global_state.h"

#include <chrono>

using namespace std::chrono;

namespace gccl {

extern GlobalState gccl_global;

void export_to_metis_format(int n, int *xadj, int *adjncy, int *node_weights, 
                           int *edge_weights, int *parts, const char *output_file) {
    
    // Open the output file
    FILE *f = fopen(output_file, "w");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", output_file);
        return;
    }
    
    // Calculate number of edges in METIS format (count each edge only once for undirected graphs)
    int num_edges = 0;
    std::set<std::pair<int, int>> edge_set;
    
    for (int i = 0; i < n; i++) {
        for (int j = xadj[i]; j < xadj[i+1]; j++) {
            int neighbor = adjncy[j];
            edge_set.insert(std::make_pair(std::min(i, neighbor), std::max(i, neighbor)));
        }
    }
    
    num_edges = edge_set.size();
    
    // Determine format code
    int format_code = 0;
    bool has_edge_weights = edge_weights != nullptr;
    bool has_node_weights = node_weights != nullptr;
    
    if (has_edge_weights) {
        format_code += 1;
    }
    if (has_node_weights) {
        format_code += 10;
    }
    
    // Write header
    if (format_code > 0) {
        fprintf(f, "%d %d %d\n", n, num_edges, format_code);
    } else {
        fprintf(f, "%d %d\n", n, num_edges);
    }
    
    // Write adjacency list for each node
    for (int node_id = 0; node_id < n; node_id++) {
        // Add node weight if present
        if (has_node_weights) {
            fprintf(f, "%d ", node_weights[node_id]);
        }
        
        // Write neighbors with weights if needed
        for (int j = xadj[node_id]; j < xadj[node_id+1]; j++) {
            int neighbor = adjncy[j];
            // METIS uses 1-indexed node IDs
            if (has_edge_weights) {
                fprintf(f, "%d %d ", neighbor + 1, edge_weights[j]);
            } else {
                fprintf(f, "%d ", neighbor + 1);
            }
        }
        fprintf(f, "\n");
    }
    
    // Write partition information if provided
    if (parts != nullptr) {
        for (int node_id = 0; node_id < n; node_id++) {
            fprintf(f, "%d\n", parts[node_id]);
        }
    }
    
    fclose(f);
    printf("Graph exported to METIS format successfully\n");
}

////////////////////////////////////////////////


void PartitionGraphMetis(int n, int *xadj, int *adjncy, int nparts, int *objval,
                         int *parts) {
  // METIS will set parts to zero if nparts is one.
  if (nparts == 1) {
    for (int i = 0; i < n; ++i) {
      parts[i] = 0;
    }
    *objval = 0;
    return;
  }
  int ncon = 1;
  // auto result = METIS_PartGraphKway(&n, &ncon, xadj, adjncy, NULL, NULL,
  // NULL, &nparts, NULL,
  //                     NULL, NULL, objval, parts);
  auto start = system_clock::now();
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = 1;
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
  options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO | METIS_DBG_INFO | METIS_DBG_TIME;
  long int nn = n;
  auto result =
      METIS_PartGraphRecursive(&n, &ncon, xadj, adjncy, NULL, NULL, NULL,
                               &nparts, NULL, NULL, options, objval, parts);
  //auto result =
  //    METIS_PartGraphKway(&n, &ncon, xadj, adjncy, NULL, NULL, NULL,
  //                             &nparts, NULL, NULL, options, objval, parts);
  CHECK_EQ(result, METIS_OK);
  auto end = system_clock::now();
  // LOG(INFO) << "Finished partition graph, took " << duration_cast<seconds>(end - start).count() << "\n";
}

void PartitionGraphMetisWithEdgeAndNodeWeight(int n, int *xadj, int *adjncy, int nparts, int *objval,
                                              int *node_weights, int *edge_weights, int *parts) {
  // METIS will set parts to zero if nparts is one.
  if (nparts == 1) {
    for (int i = 0; i < n; ++i) {
      parts[i] = 0;
    }
    *objval = 0;
    return;
  }
  int ncon = 1;
  // auto result = METIS_PartGraphKway(&n, &ncon, xadj, adjncy, NULL, NULL,
  // NULL, &nparts, NULL,
  //                     NULL, NULL, objval, parts);
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = 1;
  options[METIS_OPTION_UFACTOR] = 5;
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
  options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO | METIS_DBG_INFO | METIS_DBG_TIME;
  long int nn = n;

  auto result =
      METIS_PartGraphRecursive(&n, &ncon, xadj, adjncy, node_weights, NULL, edge_weights,
                               &nparts, NULL, NULL, options, objval, parts);
  //auto result =
  //    METIS_PartGraphKway(&n, &ncon, xadj, adjncy, NULL, NULL, NULL,
  //                             &nparts, NULL, NULL, options, objval, parts);
  CHECK_EQ(result, METIS_OK);
}

void RePartitionGraph(int n, int *xadj, int *adjncy, int nparts, int *objval,
                      int *node_weights, int *edge_weights, int prev_nparts, int *prev_parts, int *parts) {
  // todo: integrate with already setup part result
  // 1. export
  export_to_metis_format(n, xadj, adjncy, node_weights, edge_weights, prev_parts, "/dev/shm/orig.metis");
  if (nparts == 1) {
    for (int i = 0; i < n; ++i) {
      parts[i] = 0;
    }
    *objval = 0;
    return;
  }

  // 2. do partition
  std::string cmd = "bash /workspace/METIS_wpb/run_wpb_container.sh /dev/shm/orig.metis " + std::to_string(prev_nparts) + " " + std::to_string(nparts);
  int result = system(cmd.c_str());
  // 3. load part result
  std::ifstream partres("/dev/shm/repart.parts");
  for (int i = 0; i < n; i++) {
    std::string line;
    std::getline(partres, line);
    if (!line.empty()) {
        parts[i] = std::stoi(line);
    }
  }

  // PartitionGraphMetisWithEdgeAndNodeWeight(n, xadj, adjncy, nparts, objval, node_weights, edge_weights, parts);
}

std::vector<int> PartitionGraphInternal(Graph &graph, int nparts) {
  auto start = system_clock::now();

  std::vector<int> parts(graph.n_nodes);
  std::string part_opt = GetEnvParam("PART_OPT", std::string("RECUR_MINI"));

  if (part_opt == "RECUR_MINI" && graph.mini_graph == nullptr) {
    LOG(INFO) << "No mini graph, using METIS to partition graph";
    part_opt = "METIS";
  }

  if (part_opt == "METIS") {
    LOG(INFO) << "Using METIS to partition graph";
    int objval;
    PartitionGraphMetis(graph.n_nodes, graph.xadj.data(), graph.adjncy.data(),
                        nparts, &objval, parts.data());
  } else if (part_opt == "RECUR_MINI") {
    LOG(INFO) << "Using METIS to partition mini graph";
    int objval;
    std::vector<int> mini_parts(graph.mini_graph->n_nodes);
    std::string repart_opt = GetEnvParam("REPART_OPT", std::string("FALSE"));
    if (repart_opt == "TRUE" && gccl_global.previous_g != nullptr && gccl_global.previous_g->mini_graph != nullptr && gccl_global.previous_g->mini_graph->n_parts != 1) {
      RePartitionGraph(graph.mini_graph->n_nodes,
                        graph.mini_graph->xadj.data(),
                        graph.mini_graph->adjncy.data(),
                        nparts, &objval,
                        graph.mini_graph->node_weights.data(),
                        graph.mini_graph->edge_weights.data(),
                        gccl_global.previous_g->mini_graph->n_parts,
                        gccl_global.previous_g->mini_graph->parts.data(),
                        mini_parts.data());
    } else {
      PartitionGraphMetisWithEdgeAndNodeWeight(graph.mini_graph->n_nodes,
                                             graph.mini_graph->xadj.data(),
                                             graph.mini_graph->adjncy.data(),
                                             nparts, &objval,
                                             graph.mini_graph->node_weights.data(),
                                             graph.mini_graph->edge_weights.data(),
                                             mini_parts.data());
    }
    graph.mini_graph->parts = mini_parts;
    graph.mini_graph->n_parts = nparts;
    for (int i = 0; i < graph.n_nodes; ++i) {
      parts[i] = mini_parts[graph.gid2mid[i]];
    }
  } else if (part_opt == "NAIVE") {
    LOG(INFO) << "Using naive way to partition graph";
    for (int i = 0; i < graph.n_nodes; ++i) {
      parts[i] = i % nparts;
    }
  } else {
    CHECK(false) << "Unknown partition option" << part_opt;
  }

  auto end = system_clock::now();
  LOG(INFO) << "Finished partition graph with edge and node weight, took " << duration_cast<seconds>(end - start).count() << "\n";
  return parts;
}

}  // namespace gccl
