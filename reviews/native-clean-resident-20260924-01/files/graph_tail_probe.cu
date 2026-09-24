// CUDA 12.0 feasibility probe for a device-resident adaptive controller.
// This file is not a neural model and no speed result from it is scientific.
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

void checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__global__ void count_and_relaunch(int* count, int limit) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next = ++(*count);
    if (next < limit) {
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
  }
}
}  // namespace

extern "C" const char* native_tail_error() { return last_error.c_str(); }

extern "C" int native_tail_fixture(int limit, int* observed) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaStream_t stream = nullptr;
  int* count = nullptr;
  last_error.clear();
  try {
    if (limit < 1 || limit > 64 || observed == nullptr) {
      throw std::runtime_error("invalid fixture argument");
    }
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    checked(cudaMalloc(&count, sizeof(int)), "count allocation");
    checked(cudaMemsetAsync(count, 0, sizeof(int), stream), "count reset");
    checked(cudaGraphCreate(&graph, 0), "graph creation");
    cudaKernelNodeParams node = {};
    node.func = reinterpret_cast<void*>(count_and_relaunch);
    node.gridDim = dim3(1);
    node.blockDim = dim3(1);
    void* arguments[] = {&count, &limit};
    node.kernelParams = arguments;
    cudaGraphNode_t kernel;
    checked(cudaGraphAddKernelNode(&kernel, graph, nullptr, 0, &node), "kernel node");
    checked(cudaGraphInstantiate(&executable, graph,
                                 cudaGraphInstantiateFlagDeviceLaunch), "device graph instantiate");
    checked(cudaGraphUpload(executable, stream), "device graph upload");
    checked(cudaGraphLaunch(executable, stream), "first graph launch");
    checked(cudaStreamSynchronize(stream), "tail completion");
    checked(cudaMemcpy(observed, count, sizeof(int), cudaMemcpyDeviceToHost), "count readback");
    if (*observed != limit) {
      throw std::runtime_error("tail self-launch did not execute expected count");
    }
  } catch (const std::exception& exc) {
    last_error = exc.what();
  }
  if (executable) cudaGraphExecDestroy(executable);
  if (graph) cudaGraphDestroy(graph);
  if (count) cudaFree(count);
  if (stream) cudaStreamDestroy(stream);
  return last_error.empty() ? 0 : -1;
}

extern "C" int native_probe_imported_graph(void* source_graph) {
  cudaGraphExec_t executable = nullptr;
  last_error.clear();
  try {
    if (!source_graph) throw std::runtime_error("null captured graph");
    checked(cudaGraphInstantiate(&executable,
                                 reinterpret_cast<cudaGraph_t>(source_graph),
                                 cudaGraphInstantiateFlagDeviceLaunch),
            "imported graph instantiate");
  } catch (const std::exception& exc) {
    last_error = exc.what();
  }
  if (executable) cudaGraphExecDestroy(executable);
  return last_error.empty() ? 0 : -1;
}
