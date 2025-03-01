#include "partitioner.h"

#include <stddef.h>

#include "glog/logging.h"
#include "metis.h"
#include "param.h"

#include <chrono>

using namespace std::chrono;

namespace gccl {

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
  options[METIS_OPTION_UFACTOR] = 10;
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
    PartitionGraphMetisWithEdgeAndNodeWeight(graph.mini_graph->n_nodes,
                                             graph.mini_graph->xadj.data(),
                                             graph.mini_graph->adjncy.data(),
                                             nparts, &objval,
                                             graph.mini_graph->node_weights.data(),
                                             graph.mini_graph->edge_weights.data(),
                                             mini_parts.data());
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
