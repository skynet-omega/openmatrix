"""GPU-only batched CSC accumulator microbenchmark on a real CNS graph.

This is not an organism runner. Owner-specific coefficients and integration are
absent. All 59 updates execute inside one captured CUDA graph; Python prepares
inputs and reports results outside the hot loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


CUDA = r'''
extern "C" __global__ void csr_seed(
 int n,const long long* ptr,const int* src,const double* w,
 const double* release,const double* caps,const bool* visual,
 double scale,double* excit,double* inhib) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(row>=n)return;
 double a=0.,b=0.;
 for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32){
  int c=src[e];double weight=w[e];
  if(visual[row]){
   double v=(weight*scale)*release[c];
   if(v>=0.)a+=v;else b-=v;
  }else a+=weight*(release[c]*caps[c]);
 }
 for(int d=16;d>0;d/=2){
  a+=__shfl_down_sync(0xffffffff,a,d);
  b+=__shfl_down_sync(0xffffffff,b,d);
 }
 if(lane==0){excit[row]=a;inhib[row]=b;}
}

extern "C" __global__ void csc_delta(
 int n,const long long* ptr,const int* dst,const double* w,
 const double* release,double* cache,const double* caps,const bool* visual,
 const bool* event,double scale,double eps,double* excit,double* inhib) {
 int c=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(c>=n)return;
 double previous=cache[c],current=release[c],difference=current-previous;
 bool update=event[c]?(difference!=0.):(fabs(difference)>eps);
 if(update){
  for(long long e=ptr[c]+lane;e<ptr[c+1];e+=32){
   int row=dst[e];double weight=w[e];
   if(visual[row]){
    if(weight>=0.)atomicAdd(excit+row,(weight*scale)*difference);
    else atomicAdd(inhib+row,(-weight*scale)*difference);
   }else atomicAdd(excit+row,weight*(difference*caps[c]));
  }
  if(lane==0)cache[c]=current;
 }
}
'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def selftest():
    """Tiny graph with visual signed and ordinary postsynaptic rows."""
    import cupy as cp
    import scipy.sparse as sp
    n = 3
    ptr = np.array([0, 2, 4, 6], dtype=np.int64)
    src = np.array([0, 1, 0, 2, 1, 2], dtype=np.int32)
    w = np.array([2., -1., 1.5, -0.5, -3., 4.], dtype=np.float64)
    caps = np.array([2., 3., 4.])
    visual = np.array([True, False, True])
    event = np.array([False, True, False])
    states = np.array([[.1, .2, .3], [.1000001, .25, .31], [.15, .22, .35]])
    csc = sp.csr_matrix((w, src, ptr), shape=(n, n)).tocsc()
    module = cp.RawModule(code=CUDA, options=('--std=c++11', '--fmad=false'))
    seed, delta = [module.get_function(name) for name in ('csr_seed', 'csc_delta')]
    dp, ds, dw, dcp, ddst, dcw = [cp.asarray(x) for x in
                                   (ptr, src, w, csc.indptr.astype(np.int64),
                                    csc.indices.astype(np.int32), csc.data)]
    d_caps, d_visual, d_event = cp.asarray(caps), cp.asarray(visual), cp.asarray(event)
    a, b, cache = cp.empty(n), cp.empty(n), cp.asarray(states[0])
    launch = lambda state: seed((1,), (128,), (np.int32(n), dp, ds, dw,
                                              cp.asarray(state), d_caps, d_visual,
                                              np.float64(1.), a, b))
    launch(states[0])
    for state in states[1:]:
        d_state = cp.asarray(state)
        delta((1,), (128,), (np.int32(n), dcp, ddst, dcw,
                             d_state, cache, d_caps, d_visual, d_event,
                             np.float64(1.), np.float64(1e-6), a, b))
        cp.cuda.get_current_stream().synchronize()
        observed_a, observed_b = cp.asnumpy(a), cp.asnumpy(b)
        exact_a, exact_b = np.zeros(n), np.zeros(n)
        cached = cp.asnumpy(cache)
        for row in range(n):
            for e in range(ptr[row], ptr[row+1]):
                c = src[e]
                if visual[row]:
                    if w[e]>=0:exact_a[row]+=w[e]*cached[c]
                    else:exact_b[row]+=-w[e]*cached[c]
                else:exact_a[row]+=w[e]*cached[c]*caps[c]
        need(np.max(np.abs(observed_a-exact_a)) < 1e-12 and
             np.max(np.abs(observed_b-exact_b)) < 1e-12,
             'Synthetic CSC arithmetic')
    return {'status': 'PASS_SYNTHETIC_CSC', 'CUDA_executed': True}


def run(out):
    import cupy as cp
    import scipy.sparse as sp
    started = time.monotonic()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    plan_path = here / 'CSC_QUANTIZED_CURRENT_PLAN_35.json'
    p = json.loads(plan_path.read_text())
    need(not out.exists() and p['schema'] == 'csc_quantized_gpu_plan_v1',
         'Unique run and plan required')
    need(sha(Path(__file__)) == p['script_sha256'], 'Code changed')
    need({name: sha(root/name) for name in p['frozen_inputs_sha256']} ==
         p['frozen_inputs_sha256'], 'Inputs changed')
    free_before, _ = cp.cuda.runtime.memGetInfo()
    mapping = json.loads((root/p['mapping']).read_text())
    with np.load(root/p['graph_archive'], allow_pickle=False) as z:
        ptr, src, w, caps, visual = [z[mapping[name]] for name in
                                      ('cuda/indptr', 'cuda/indices', 'cuda/weights',
                                       'cuda/caps', 'cuda/visual')]
    with np.load(root/p['current_weights_archive'], allow_pickle=False) as z:
        current_w = z['weights']
    need(current_w.shape == w.shape and current_w.dtype == w.dtype,
         'Current weight layout')
    w = current_w
    with np.load(root/p['release_capsule'], allow_pickle=False) as z:
        release = z['release']
        degree = z['outdegree']
        event_rows = z['event_rows']
    n = len(caps)
    need(n == 166700 and len(src) == 25582938 and len(w) == len(src)
         and release.shape == (60, n), 'Graph/release layout')
    csc_started = time.monotonic()
    csc = sp.csr_matrix((w, src, ptr), shape=(n,n)).tocsc(copy=True)
    need(csc.nnz == len(w) and
         np.array_equal(np.diff(csc.indptr), degree), 'CSC conversion changed edges')
    csc_build_s = time.monotonic() - csc_started
    event = np.zeros(n, dtype=bool)
    event[event_rows] = True
    module = cp.RawModule(code=CUDA, options=('--std=c++11', '--fmad=false',
                                              '--prec-div=true', '--prec-sqrt=true'))
    seed, delta = [module.get_function(name) for name in ('csr_seed','csc_delta')]
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        dptr, dsrc, dw, dcp, ddst, dcw = [cp.asarray(x) for x in
                                           (ptr, src, w, csc.indptr.astype(np.int64),
                                            csc.indices.astype(np.int32), csc.data)]
        dcaps, dvisual, devent = cp.asarray(caps), cp.asarray(visual), cp.asarray(event)
        dr = cp.asarray(release)
        cache = cp.empty(n, dtype=cp.float64)
        a, b = cp.empty(n, dtype=cp.float64), cp.empty(n, dtype=cp.float64)
        exact_a, exact_b = cp.empty_like(a), cp.empty_like(b)
        snapshots_a, snapshots_b = cp.empty((60,n)), cp.empty((60,n))
        csc_args = (np.int32(n), dcp, ddst, dcw, dcaps, dvisual, devent,
                    np.float64(p['scale']), np.float64(p['absolute_release_threshold']),
                    a, b)
        seed_grid = ((n*32+255)//256,)
        delta_grid = ((n*32+255)//256,)
        def reset():
            cp.copyto(cache, dr[0])
            seed(seed_grid, (256,), (np.int32(n), dptr, dsrc, dw, dr[0],
                                      dcaps, dvisual, np.float64(p['scale']), a, b))
        reset()
        stream.synchronize()
        stream.begin_capture()
        for j in range(1,60):
            delta(delta_grid, (256,), (np.int32(n), dcp, ddst, dcw,
                                       dr[j], cache, dcaps, dvisual, devent,
                                       np.float64(p['scale']),
                                       np.float64(p['absolute_release_threshold']), a, b))
        graph_speed = stream.end_capture()
        reset()
        stream.synchronize()
        stream.begin_capture()
        cp.copyto(snapshots_a[0], a)
        cp.copyto(snapshots_b[0], b)
        for j in range(1,60):
            delta(delta_grid, (256,), (np.int32(n), dcp, ddst, dcw,
                                       dr[j], cache, dcaps, dvisual, devent,
                                       np.float64(p['scale']),
                                       np.float64(p['absolute_release_threshold']), a, b))
            cp.copyto(snapshots_a[j], a)
            cp.copyto(snapshots_b[j], b)
        graph_science = stream.end_capture()
        # The science graph produces a complete trajectory of approximate sums.
        reset()
        stream.synchronize()
        graph_science.launch(stream)
        stream.synchronize()
        approximate_a = cp.asnumpy(snapshots_a, stream=stream)
        approximate_b = cp.asnumpy(snapshots_b, stream=stream)
        # Compare with 60 exact full-CSR sums using the same captured weights.
        exact_max_a = exact_max_b = 0.
        per_query = []
        for j in range(60):
            seed(seed_grid, (256,), (np.int32(n), dptr, dsrc, dw, dr[j],
                                      dcaps, dvisual, np.float64(p['scale']),
                                      exact_a, exact_b))
            ea, eb = cp.asnumpy(exact_a, stream=stream), cp.asnumpy(exact_b, stream=stream)
            da = float(np.max(np.abs(approximate_a[j]-ea)))
            db = float(np.max(np.abs(approximate_b[j]-eb)))
            exact_max_a=max(exact_max_a,da);exact_max_b=max(exact_max_b,db)
            per_query.append({'query':j,'excit_max_abs':da,'inhib_max_abs':db})
        seed_error = max(per_query[0]['excit_max_abs'], per_query[0]['inhib_max_abs'])
        # CUDA graph time includes all 59 sparse kernels, not Python launch gaps.
        gpu_times=[]
        for _ in range(p['timing_repetitions']):
            reset()
            stream.synchronize()
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            graph_speed.launch(stream)
            stop.record(stream)
            stop.synchronize()
            gpu_times.append(float(cp.cuda.get_elapsed_time(start,stop)))
        terminal_cache = cp.asnumpy(cache,stream=stream)
        expected_cache = release[0].copy()
        for j in range(1,60):
            d = release[j] - expected_cache
            active = np.where(event, d != 0., np.abs(d) > p['absolute_release_threshold'])
            expected_cache[active] = release[j,active]
        need(np.array_equal(terminal_cache,expected_cache),'Terminal cache differs')
    free_after, _ = cp.cuda.runtime.memGetInfo()
    result = {'schema':'csc_quantized_gpu_result_v1',
              'status':'COMPLETE_MICROBENCH_ONLY',
              'plan_sha256':sha(plan_path),'code_sha256':sha(Path(__file__)),
              'graph_edges':len(w),'csc_build_s':csc_build_s,
              'gpu_59_query_ms':gpu_times,
              'gpu_ms_per_query_median':float(np.median(gpu_times)/59),
              'excit_max_abs':exact_max_a,'inhib_max_abs':exact_max_b,
              'seed_error_probe':seed_error,'per_query':per_query,
              'wall_s':time.monotonic()-started,
              'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'gpu_allocation_delta_upper_bytes':int(max(0,free_before-free_after)),
              'scope':'Real CNS graph and same-run release vectors; batched CUDA graph for base E/I sums only. No specialized owners, CNS integration, body, or whole-organism speed.'}
    budget=p['budget']
    result['budget_ok']=(result['wall_s']<=budget['wall_seconds_max'] and
                         result['maxrss_kib']<=budget['ram_gib_max']*1024**2 and
                         result['gpu_allocation_delta_upper_bytes']<=budget['incremental_vram_gib_max']*1024**3)
    result['numerical_gate_ok']=(exact_max_a<=p['maximum_current_abs'] and
                                 exact_max_b<=p['maximum_current_abs'])
    result['speed_gate_ok']=result['gpu_ms_per_query_median']<=p['gpu_ms_per_query_max']
    out.mkdir()
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest()))
    else:
        need(args.out is not None,'Output required')
        r=run(args.out)
        print(json.dumps({k:r[k] for k in ('status','gpu_ms_per_query_median',
                                          'excit_max_abs','inhib_max_abs','numerical_gate_ok',
                                          'speed_gate_ok','wall_s','budget_ok')}))
