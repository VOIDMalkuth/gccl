#pragma once

#include <cuda_runtime.h>

#include "gpu/kernel/primitives.h"

namespace gccl {
namespace {

template <typename T>
std::vector<T> CopyCudaPtrToVec(T *dev_ptr, int size) {
  std::vector<T> ret(size);
  cudaMemcpy(ret.data(), dev_ptr, size * sizeof(T), cudaMemcpyDefault);
  return ret;
}

}  // namespace
}  // namespace gccl
