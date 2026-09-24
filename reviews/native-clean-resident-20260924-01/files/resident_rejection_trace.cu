#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void* resident_create(void*, void*, double*, double*, double*,
                                  double*, long, long);
extern "C" int resident_advance(void*, long, long*, long, long, const double*,
                                 long, long, long*, double*);
extern "C" void resident_destroy(void*);
extern "C" const char* resident_error();

static void check(cudaError_t code, const char* label) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(code));
}

__global__ void scalar_trial(const double* clock, const double* state,
                             double* fine, double* status,
                             int* trial_count, double* trial_h,
                             double* trial_used, double* trial_state) {
  if (blockIdx.x || threadIdx.x) return;
  const int index = atomicAdd(trial_count,1);
  if (index < 64) {
    trial_h[index] = clock[1];
    trial_used[index] = clock[0];
    trial_state[index] = state[0];
  }
  const double h = clock[1];
  fine[0] = state[0] + 1000.0 * h;
  status[0] = h > 500e-9 ? 2.0 : .05;
  status[1] = 0.0;
  status[2] = 0.0;
}

struct Oracle {
  double state = 0, used = 0, error = 0;
  long next = 1000, accepted = 0, rejected = 0, minimum = 1000;
  std::vector<double> trial_h, trial_used, trial_state;
};

static Oracle host_oracle() {
  Oracle out;
  const double end = 2000e-9;
  const double event = 1500e-9;
  while (out.used < end) {
    const double stop = event > out.used ? event : end;
    const double available = stop - out.used;
    const double h = std::min(available, std::min(out.next, 1000L) * 1e-9);
    if (!(h > 0)) throw std::runtime_error("host oracle stalled");
    out.trial_h.push_back(h);
    out.trial_used.push_back(out.used);
    out.trial_state.push_back(out.state);
    const double err = h > 500e-9 ? 2.0 : .05;
    if (err <= 1.0) {
      out.state += 1000.0 * h;
      out.used = h == available ? stop : out.used + h;
      ++out.accepted;
      out.minimum = std::min(out.minimum, static_cast<long>(std::ceil(h * 1e9)));
      out.error = std::max(out.error, err);
      out.next = std::min(1000L,std::max(100L,
          static_cast<long>(std::floor(h * 1e9 * (err < .1 ? 2.0 : 1.0)))));
    } else {
      ++out.rejected;
      out.next = static_cast<long>(std::floor(h * 1e9 * .5));
      if (out.next < 100) throw std::runtime_error("host accuracy limit");
    }
  }
  return out;
}

int main() {
  cudaStream_t stream = nullptr;
  cudaGraph_t trial = nullptr;
  double *clock = nullptr, *status = nullptr, *state = nullptr, *fine = nullptr;
  double *trial_h = nullptr, *trial_used = nullptr, *trial_state = nullptr;
  int* trial_count = nullptr;
  void* runner = nullptr;
  try {
    const auto start = std::chrono::steady_clock::now();
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    check(cudaMalloc(&clock,2*sizeof(double)),"clock");
    check(cudaMalloc(&status,3*sizeof(double)),"status");
    check(cudaMalloc(&state,sizeof(double)),"state");
    check(cudaMalloc(&fine,sizeof(double)),"fine");
    check(cudaMalloc(&trial_count,sizeof(int)),"trial count");
    check(cudaMalloc(&trial_h,64*sizeof(double)),"trial h log");
    check(cudaMalloc(&trial_used,64*sizeof(double)),"trial used log");
    check(cudaMalloc(&trial_state,64*sizeof(double)),"trial state log");
    check(cudaMemsetAsync(state,0,sizeof(double),stream),"initial state");
    check(cudaMemsetAsync(trial_count,0,sizeof(int),stream),"trial count reset");
    check(cudaGraphCreate(&trial,0),"trial graph");
    cudaKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(scalar_trial);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    void* arguments[] = {&clock,&state,&fine,&status,&trial_count,
                         &trial_h,&trial_used,&trial_state};
    params.kernelParams = arguments;
    cudaGraphNode_t node;
    check(cudaGraphAddKernelNode(&node,trial,nullptr,0,&params),"trial kernel");
    runner = resident_create(trial,stream,clock,status,state,fine,1,1);
    if (!runner) throw std::runtime_error(resident_error());
    check(cudaGraphDestroy(trial),"source graph release");
    trial = nullptr;
    const double event = 1500e-9;
    long next = 1000, counts[3] = {-1,-1,-1};
    double error = -1;
    if (resident_advance(runner,2000,&next,100,1000,&event,1,10000,
                         counts,&error))
      throw std::runtime_error(resident_error());
    double actual = -1;
    check(cudaMemcpy(&actual,state,sizeof(double),cudaMemcpyDeviceToHost),
          "state readback");
    const Oracle expected = host_oracle();
    int observed_trials = -1;
    check(cudaMemcpy(&observed_trials,trial_count,sizeof(int),
                     cudaMemcpyDeviceToHost),"trial count readback");
    if (observed_trials < 0 || observed_trials > 64)
      throw std::runtime_error("diagnostic log capacity exceeded");
    std::vector<double> gh(observed_trials),gu(observed_trials),gx(observed_trials);
    check(cudaMemcpy(gh.data(),trial_h,observed_trials*sizeof(double),
                     cudaMemcpyDeviceToHost),"trial h readback");
    check(cudaMemcpy(gu.data(),trial_used,observed_trials*sizeof(double),
                     cudaMemcpyDeviceToHost),"trial used readback");
    check(cudaMemcpy(gx.data(),trial_state,observed_trials*sizeof(double),
                     cudaMemcpyDeviceToHost),"trial state readback");
    std::cout << "GPU_TRIALS " << observed_trials << " CPU_TRIALS "
              << expected.trial_h.size() << "\n";
    const size_t total = std::max(gh.size(),expected.trial_h.size());
    for (size_t i=0; i<total; ++i) {
      std::cout << "TRIAL " << i;
      if (i<gh.size())
        std::cout << " GPU_H " << std::hexfloat << gh[i]
                  << " GPU_USED " << gu[i] << " GPU_X " << gx[i];
      if (i<expected.trial_h.size())
        std::cout << " CPU_H " << std::hexfloat << expected.trial_h[i]
                  << " CPU_USED " << expected.trial_used[i]
                  << " CPU_X " << expected.trial_state[i];
      std::cout << "\n";
    }
    const bool pass = expected.rejected >= 1 && expected.accepted >= 1 &&
        std::isfinite(actual) && std::abs(actual-expected.state) <= 1e-12 &&
        next == expected.next && counts[0] == expected.accepted &&
        counts[1] == expected.rejected && counts[2] == expected.minimum &&
        error == expected.error;
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"pass\":" << (pass ? "true" : "false")
              << ",\"state\":" << actual
              << ",\"expected_state\":" << expected.state
              << ",\"accepted\":" << counts[0]
              << ",\"rejected\":" << counts[1]
              << ",\"minimum_ns\":" << counts[2]
              << ",\"next_ns\":" << next
              << ",\"max_error\":" << error
              << ",\"wall_s\":" << wall << "}\n";
    resident_destroy(runner);
    cudaFree(trial_state);cudaFree(trial_used);cudaFree(trial_h);
    cudaFree(trial_count);
    cudaFree(fine);cudaFree(state);cudaFree(status);cudaFree(clock);
    cudaStreamDestroy(stream);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    std::cerr << "FAIL: " << exc.what() << "\n";
    if (runner) resident_destroy(runner);
    if (trial) cudaGraphDestroy(trial);
    if (fine) cudaFree(fine);
    if (trial_state) cudaFree(trial_state);
    if (trial_used) cudaFree(trial_used);
    if (trial_h) cudaFree(trial_h);
    if (trial_count) cudaFree(trial_count);
    if (state) cudaFree(state);
    if (status) cudaFree(status);
    if (clock) cudaFree(clock);
    if (stream) cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }
}
