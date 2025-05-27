#pragma once

#include <memory>
#include <vector>

#include "base/bin_stream.h"

namespace gccl {

struct Graph {
  template <typename EdgeFunc>
  void ApplyEdge(EdgeFunc func) const {
    for (int i = 0; i < n_nodes; ++i) {
      for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
        func(i, adjncy[j]);
      }
    }
  }
  // should be deprecated
  Graph() {}
  Graph(int n, const std::vector<std::pair<int, int>> &edges);
  Graph(const std::vector<std::pair<int, int>> &edges);
  Graph(int n, int *xadj, int *adjncy);
  Graph(const std::string &file);

  void WriteToFile(const std::string &file) const;

  BinStream &serialize(BinStream &bs) const;
  BinStream &deserialize(BinStream &bs);

  std::vector<int> xadj;    // offset
  std::vector<int> adjncy;  // neighbours
  int n_nodes;              // n_nodes_ + 1 == xadj_.size()
  int n_edges;              // n_edges_ == adjncy_.size()
  int n_parts;
  std::vector<int> parts;

  std::shared_ptr<Graph> mini_graph = nullptr; // for partitioning on root node only
  std::vector<int> gid2mid; // i.e. full_graph_map[i] = j means node i in full graph corresponds to node j in mini graph
  std::vector<int> node_weights;
  std::vector<int> edge_weights; // i.e. each edge represents how much connection a node has
};

struct LocalGraphInfo {
  int n_nodes;
  int n_local_nodes;
};

void BuildRawGraph(const Graph &g, int *n, int **xadj, int **adjncy);

}  // namespace gccl
