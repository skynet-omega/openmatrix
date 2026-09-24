"""Bounded, non-invasive sideband of real base-CSR RHS inside a CNS CUDA graph.

The trace kernels write separate buffers and cannot affect the model state.
This is a coefficient diagnostic, not a replacement integrator or stage test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EPOCH = ROOT/'motor_nuevo/epoch_cost_20260923'
PLAN = HERE/'TRACE_PLAN_01.json'
OUT = HERE/'trace_real_rhs_1ms_03'
BASE = HERE/'fusion_base_1ms_01/final_state/session.npz'

SOURCE = r'''
extern "C" __global__ void trace_time(const double* clock, double frac,
 unsigned long long* count, double* times, int first, int slots) {
 if(blockIdx.x==0 && threadIdx.x==0) {
  unsigned long long j=atomicAdd(count,1ULL);
  if(j>=first && j<first+slots) times[j-first]=clock[0]+frac*clock[1];
 }
}
extern "C" __global__ void trace_rhs(const double* rhs,const unsigned long long* count,
 double* out,int n,int first,int slots) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 long long j=(long long)count[0]-1LL-first;
 if(i<n && j>=0 && j<slots) out[j*(long long)n+i]=rhs[i];
}
'''

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    if not __debug__:
        raise RuntimeError('Historical loader rejects optimized Python')
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    started=time.perf_counter()
    plan=json.loads(PLAN.read_text())
    budget=plan['budget']
    result={'schema':'real_cns_rhs_trace_result_v1','status':'STARTED',
            'stage_admission':False,'simulated_trial_ms':0,
            'plan_sha256':file_hash(PLAN),'source_sha256':file_hash(Path(__file__))}
    obj=session=None
    old_coeff=original_kernel=None
    graph_class=None
    try:
        import cupy as cp
        sys.path[:0]=[str(OLD/'work/motor14_20260922'),str(OLD/'work/motor13_20260922'),
                      str(ROOT/'motor_nuevo/pipeline_review_20260922')]
        from motor_runtime import load
        from runtime_session import RuntimeSession
        from csr_encoding_probe import EXPECTED
        obj,d,_,_,_,_=load(OUT/'preparation_inputs')
        h=obj.core.hybrid
        W=h.brain.W
        n=int(W.shape[0]); slots=int(budget['captured_rhs_max']);first=12
        if n!=166700 or W.nnz!=25582938:
            raise ValueError('Unexpected graph dimensions')
        w_initial=np.asarray(h.weights64,dtype=np.float64).copy()
        if digest(w_initial)!=plan['identity']['graph_weights_sha256'] or \
           digest(np.asarray(W.indices))!=plan['identity']['graph_indices_sha256']:
            raise ValueError('Graph identity mismatch')
        initial=np.asarray(h.state[h.transmission_start:h.transmission_start+n],dtype=np.float64)
        if digest(initial)!=plan['identity']['initial_transmission_sha256']:
            raise ValueError('Initial transmission mismatch')
        if EXPECTED['weights']!=plan['identity']['graph_weights_sha256']:
            raise ValueError('Reference identity mismatch')
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            if name in sys.modules:
                raise RuntimeError('Preloaded event owner: '+name)
        sys.path.insert(0,str(EPOCH))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for mod in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(mod.__file__).resolve().parent!=EPOCH:
                raise RuntimeError('Wrong event source '+mod.__name__)
        session=RuntimeSession(h,'causal_cuda')
        import graph_core
        graph_class=graph_core.NativeGraph
        original_kernel=h.kernel
        old_coeff=graph_class.coeff
        mod=cp.RawModule(code=SOURCE,options=('--std=c++11','--fmad=false'))
        mark=mod.get_function('trace_time');capture=mod.get_function('trace_rhs')
        trace_count=cp.zeros(1,dtype=cp.uint64)
        trace_times=cp.full(slots,np.nan,dtype=cp.float64)
        trace_rhs=cp.full((slots,n),np.nan,dtype=cp.float64)
        gate={'pending':False,'base_calls':0}

        def traced_coeff(self,y,frac):
            if gate['pending']:
                raise RuntimeError('Nested base coefficient')
            z=self.project(y,self.clock,frac) if self.project else y
            mark((1,),(1,),(self.clock,np.float64(frac),trace_count,
                           trace_times,np.int32(first),np.int32(slots)))
            gate['pending']=True
            try:
                a,b=self.coefficient(z)
            finally:
                if gate['pending']:
                    raise RuntimeError('Base CSR kernel was not launched once')
            self.check(self.grid,(256,),(z,a,b,np.int32(self.n),self.flag))
            return z,a,b

        def traced_base(grid,block,args,*extra,**kw):
            if gate['pending']:
                if args[4].shape!=(n,) or args[4].dtype!=cp.float64:
                    raise ValueError('Base RHS shape/dtype changed')
                capture(((n+255)//256,),(256,),
                        (args[4],trace_count,trace_rhs,np.int32(n),
                         np.int32(first),np.int32(slots)))
                gate['pending']=False
                gate['base_calls']+=1
            return original_kernel(grid,block,args,*extra,**kw)

        graph_class.coeff=traced_coeff
        h.kernel=traced_base
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>budget['RAM_GiB_max']:
            raise MemoryError('RAM cap after load')
        obj.step()
        cp.cuda.runtime.deviceSynchronize()
        if time.perf_counter()-started>budget['wall_seconds_max']:
            raise TimeoutError('Wall cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>budget['RAM_GiB_max']:
            raise MemoryError('RAM cap after step')
        used=cp.cuda.runtime.memGetInfo()
        vram_used=used[1]-used[0]
        if vram_used>budget['VRAM_GiB_max']*1024**3:
            raise MemoryError('VRAM cap after step')
        runtime=session.report()
        expected=plan['baseline_cns']
        for k in ('accepted','rejected','epochs'):
            if runtime['CNS'][k]!=expected[k]:
                raise ValueError('CNS count mismatch: '+k)
        if runtime['events']['events']!=expected['events']:
            raise ValueError('Physical event count mismatch')
        if int(trace_count.get()[0])!=first+6*(expected['accepted']+expected['rejected']):
            raise ValueError('Trace coefficient count mismatch')
        if gate['base_calls']!=18:
            raise ValueError('Graph build must have 12 warmup and six captured launches')
        rhs=cp.asnumpy(trace_rhs)
        times=cp.asnumpy(trace_times)
        if rhs.shape!=(slots,n) or not np.isfinite(rhs).all() or not np.isfinite(times).all():
            raise ValueError('Incomplete finite RHS trace')
        # The adaptive midpoint graph re-queries earlier trial stages by
        # design; preserve launch order rather than invent a monotone clock.
        nonmonotone=int(np.count_nonzero(np.diff(times)<0))
        with np.load(BASE,allow_pickle=False) as saved:
            expected_state=np.asarray(saved['array_261'],dtype=np.float64)
        actual=np.asarray(h.state,dtype=np.float64)
        if actual.shape!=expected_state.shape or digest(actual)!=plan['identity']['final_state_sha256']:
            raise ValueError('Final state changed by sideband')
        # Store only the fixed base-CSR identity and the exact inputs it consumed.
        w=w_initial
        weightsha=hashlib.sha256(w.tobytes()).hexdigest()
        capsule=OUT/'base_csr_rhs_first64.npz'
        np.savez_compressed(capsule,indptr=np.asarray(W.indptr,dtype=np.int64),
            indices=np.asarray(W.indices,dtype=np.int32),weights=w,rhs=rhs,
            query_s=times,operator_epoch=np.zeros(slots,dtype=np.int64),
            budget=np.full(n,plan['followup_cache_budget_per_receptor_abs'],dtype=np.float64),
            weights_sha256=np.array([weightsha]*slots),
            context=np.array('real_sham_first64_CNS_base_CSR_rhs_fixed_initial_W_only'))
        result.update(status='COMPLETE_SHADOW_ONLY',simulated_trial_ms=1,
                      trace_count=int(trace_count.get()[0]),captured_rhs=slots,
                      timestamp_first_s=float(times[0]),timestamp_last_s=float(times[-1]),
                      timestamp_unique=int(np.unique(times).size),
                      timestamp_backward_pairs=nonmonotone,
                      base_kernel_build_calls=gate['base_calls'],
                      capsule_sha256=file_hash(capsule),capsule_bytes=capsule.stat().st_size,
                      final_state_sha256=digest(actual),
                      runtime_counts={'CNS':runtime['CNS'],'events':runtime['events']},
                      scope_limit=plan['scope_limit'])
    except BaseException:
        result.update(status='FAILED_RETAINED',error=traceback.format_exc())
    finally:
        if old_coeff is not None:
            graph_class.coeff=old_coeff
        if original_kernel is not None and obj is not None:
            obj.core.hybrid.kernel=original_kernel
        if session is not None:
            try:session.close()
            except BaseException as exc:result.update(status='FAILED_CLEANUP',cleanup=repr(exc))
        if obj is not None:
            try:obj.close()
            except BaseException as exc:result.update(status='FAILED_CLEANUP',cleanup=repr(exc))
        result['wall_s']=time.perf_counter()-started
        save(OUT/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='runtime_counts'},ensure_ascii=False,allow_nan=False))
    return 0 if result['status']=='COMPLETE_SHADOW_ONLY' else 2

if __name__=='__main__':
    raise SystemExit(main())
