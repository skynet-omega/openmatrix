// Experimental model-independent CUDA graph scheduler. The supplied child
// graph owns the neuron model; this scheduler only owns time, rejection and
// state commit. No Python operation occurs between adaptive trials.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

void checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

enum Failure : int {
  ok = 0, no_progress = 1, invalid_trial = 2, invalid_domain = 3,
  accuracy_limit = 4, trial_limit = 5, device_launch_failure = 6
};

struct Control {
  double* clock;
  const double* status;
  double* state;
  const double* fine;
  const double* events;
  long n;
  long event_count;
  long next_ns;
  long min_ns;
  long max_ns;
  long min_accepted_ns;
  long accepted;
  long rejected;
  long attempts;
  long max_attempts;
  long event_index;
  double end_s;
  double used_s;
  double attempted_h_s;
  double attempted_stop_s;
  double max_error;
  int accept;
  int failure;
  int launch_status;
};

__global__ void prepare(Control* c) {
  if (blockIdx.x || threadIdx.x) return;
  if (c->attempts >= c->max_attempts) {
    c->failure = trial_limit;
    return;
  }
  ++c->attempts;
  while (c->event_index < c->event_count &&
         c->events[c->event_index] <= c->used_s) ++c->event_index;
  const double stop = c->event_index < c->event_count
      ? c->events[c->event_index] : c->end_s;
  const double available = stop - c->used_s;
  const long proposed_ns = min(c->next_ns, c->max_ns);
  const double h = fmin(available, static_cast<double>(proposed_ns) * 1e-9);
  if (!(h > 0.0) || c->used_s + h == c->used_s) {
    c->failure = no_progress;
    return;
  }
  c->attempted_h_s = h;
  c->attempted_stop_s = stop;
  c->clock[0] = c->used_s;
  c->clock[1] = h;
  // A clipped endpoint is the scheduled floating-point timestamp itself.
  // Reconstructing it as start + (stop - start) can change its event side.
  c->clock[2] = h == available ? stop : c->used_s + h;
}

__global__ void decide(Control* c) {
  if (blockIdx.x || threadIdx.x) return;
  if (c->failure) return;
  const double error = c->status[0];
  if (!isfinite(error) || !isfinite(c->status[1]) || c->status[1] != 0.0) {
    c->failure = invalid_trial;
    return;
  }
  const double h = c->attempted_h_s;
  if (error <= 1.0) {
    if (c->status[2] != 0.0) {
      c->failure = invalid_domain;
      return;
    }
    // An interior event can clip a valid proposal to an arbitrarily short
    // tail. Preserve its pre-cut size only under the existing small-error
    // rule; the next trial still clips at events and checks its own error.
    // Compute this before advancing used_s. Epoch ends never qualify.
    const long previous_ns = c->next_ns;
    const double requested_s = static_cast<double>(
        min(previous_ns, c->max_ns)) * 1e-9;
    const double available_s = c->attempted_stop_s - c->used_s;
    const bool recover_proposal = c->event_index < c->event_count &&
        c->attempted_stop_s < c->end_s && h == available_s &&
        h < requested_s && error < .1;
    c->accept = 1;
    c->used_s = c->clock[2];
    ++c->accepted;
    c->min_accepted_ns = min(c->min_accepted_ns,
                              static_cast<long>(ceil(h * 1e9)));
    c->max_error = fmax(c->max_error, error);
    const long proposal = static_cast<long>(floor(h * 1e9 *
                                                   (error < .1 ? 2.0 : 1.0)));
    c->next_ns = min(c->max_ns, max(c->min_ns, proposal));
    if (recover_proposal)
      c->next_ns = min(c->max_ns, max(c->next_ns, previous_ns));
  } else {
    c->accept = 0;
    ++c->rejected;
    const long smaller = static_cast<long>(floor(h * 1e9 * .5));
    if (smaller < c->min_ns) c->failure = accuracy_limit;
    else c->next_ns = smaller;
  }
}

__global__ void commit(Control* c) {
  if (!c->accept || c->failure) return;
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < c->n) c->state[i] = c->fine[i];
}

__global__ void relaunch(Control* c) {
  if (blockIdx.x || threadIdx.x) return;
  if (c->failure || c->used_s >= c->end_s) return;
  const cudaError_t result = cudaGraphLaunch(cudaGetCurrentGraphExec(),
                                             cudaStreamGraphTailLaunch);
  if (result != cudaSuccess) {
    c->failure = device_launch_failure;
    c->launch_status = static_cast<int>(result);
  }
}

struct Runner {
  cudaGraph_t outer = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaStream_t stream = nullptr;
  Control* device = nullptr;
  double* events = nullptr;
  long capacity = 0;
  Control base = {};

  Runner(cudaGraph_t trial, cudaStream_t execution_stream, double* clock,
         double* status, double* state, double* fine, long n, long max_events)
      : stream(execution_stream), capacity(max_events) {
    if (!trial || !stream || !clock || !status || !state || !fine ||
        n <= 0 || max_events < 0 || max_events > 1000000)
      throw std::runtime_error("invalid resident graph contract");
    try {
      checked(cudaMalloc(&device, sizeof(Control)), "control allocation");
      checked(cudaMalloc(&events, std::max(1L, max_events) *
                                       static_cast<long>(sizeof(double))),
              "event allocation");
      base.clock = clock;
      base.status = status;
      base.state = state;
      base.fine = fine;
      base.events = events;
      base.n = n;
      checked(cudaGraphCreate(&outer, 0), "outer graph creation");
      cudaGraphNode_t previous = nullptr;
      add_kernel(&previous, prepare, dim3(1), dim3(1));
      cudaGraphNode_t child;
      checked(cudaGraphAddChildGraphNode(&child, outer, &previous, 1, trial),
              "trial child graph");
      previous = child;
      add_kernel(&previous, decide, dim3(1), dim3(1));
      add_kernel(&previous, commit,
                 dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256));
      add_kernel(&previous, relaunch, dim3(1), dim3(1));
      checked(cudaGraphInstantiate(&executable, outer,
                                   cudaGraphInstantiateFlagDeviceLaunch),
              "resident device graph instantiate");
      checked(cudaGraphUpload(executable, stream), "resident graph upload");
      checked(cudaStreamSynchronize(stream), "resident graph initial upload");
    } catch (...) {
      release();
      throw;
    }
  }

  void add_kernel(cudaGraphNode_t* previous, void (*function)(Control*),
                  dim3 grid, dim3 block) {
    cudaKernelNodeParams params = {};
    params.func = reinterpret_cast<void*>(function);
    params.gridDim = grid;
    params.blockDim = block;
    void* arguments[] = {&device};
    params.kernelParams = arguments;
    cudaGraphNode_t node;
    checked(cudaGraphAddKernelNode(&node, outer, *previous ? previous : nullptr,
                                   *previous ? 1 : 0, &params), "kernel node");
    *previous = node;
  }

  void release() {
    if (stream) cudaStreamSynchronize(stream);
    if (executable) cudaGraphExecDestroy(executable);
    if (outer) cudaGraphDestroy(outer);
    if (device) cudaFree(device);
    if (events) cudaFree(events);
    executable = nullptr;
    outer = nullptr;
    device = nullptr;
    events = nullptr;
  }
  ~Runner() { release(); }
};
}  // namespace

extern "C" const char* resident_error() { return last_error.c_str(); }

// The versioned creation symbol prevents a two-clock-field Python runtime
// from pairing with this three-field endpoint contract (and vice versa).
extern "C" void* resident_create_endpoint_v1(void* captured_trial_graph, void* stream,
                                  double* clock, double* status,
                                  double* state, double* fine,
                                  long n, long event_capacity) {
  last_error.clear();
  try {
    return new Runner(reinterpret_cast<cudaGraph_t>(captured_trial_graph),
                      reinterpret_cast<cudaStream_t>(stream), clock, status,
                      state, fine, n, event_capacity);
  } catch (const std::exception& exc) {
    last_error = exc.what();
    return nullptr;
  }
}

extern "C" void resident_destroy(void* handle) {
  delete reinterpret_cast<Runner*>(handle);
}

extern "C" int resident_advance(void* handle, long duration_ns, long* next_ns,
                                 long min_ns, long max_ns,
                                 const double* event_times_s, long event_count,
                                 long max_attempts, long* counts,
                                 double* max_error) {
  last_error.clear();
  try {
    auto* runner = reinterpret_cast<Runner*>(handle);
    if (!runner || !next_ns || !counts || !max_error || duration_ns <= 0 ||
        *next_ns <= 0 || min_ns <= 0 || max_ns < min_ns ||
        event_count < 0 || event_count > runner->capacity ||
        max_attempts <= 0 || max_attempts > 10000 ||
        (event_count && !event_times_s))
      throw std::runtime_error("invalid epoch contract");
    const double end_s = static_cast<double>(duration_ns) * 1e-9;
    for (long k = 0; k < event_count; ++k) {
      const double event = event_times_s[k];
      if (!std::isfinite(event) || event < 0.0 || event > end_s ||
          (k && event < event_times_s[k - 1]))
        throw std::runtime_error("invalid event boundary");
    }
    Control host = runner->base;
    host.event_count = event_count;
    host.next_ns = *next_ns;
    host.min_ns = min_ns;
    host.max_ns = max_ns;
    host.min_accepted_ns = max_ns;
    host.max_attempts = max_attempts;
    host.end_s = end_s;
    if (event_count)
      checked(cudaMemcpyAsync(runner->events, event_times_s,
                              event_count * sizeof(double),
                              cudaMemcpyHostToDevice, runner->stream),
              "event upload");
    checked(cudaMemcpyAsync(runner->device, &host, sizeof(host),
                            cudaMemcpyHostToDevice, runner->stream),
            "control upload");
    checked(cudaGraphLaunch(runner->executable, runner->stream),
            "resident graph launch");
    checked(cudaMemcpyAsync(&host, runner->device, sizeof(host),
                            cudaMemcpyDeviceToHost, runner->stream),
            "control readback");
    checked(cudaStreamSynchronize(runner->stream), "resident epoch completion");
    if (host.failure)
      throw std::runtime_error("resident trial failure code " +
                               std::to_string(host.failure) +
                               ", device launch status " +
                               std::to_string(host.launch_status));
    *next_ns = host.next_ns;
    counts[0] = host.accepted;
    counts[1] = host.rejected;
    counts[2] = host.min_accepted_ns;
    *max_error = host.max_error;
    return 0;
  } catch (const std::exception& exc) {
    last_error = exc.what();
    return -1;
  }
}
