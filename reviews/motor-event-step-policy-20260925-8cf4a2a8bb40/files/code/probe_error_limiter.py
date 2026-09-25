"""Locate neurons limiting the real adaptive CNS timestep, without changing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EPOCH=ROOT/'motor_nuevo/epoch_cost_20260923'
SLOTS=256
TRACE=r'''
extern "C" __global__ void mark_error(
 const long long* enabled,unsigned long long* count,const double* clock,
 const double* status,double* times,double* accepted_error,int slots){
 if(blockIdx.x||threadIdx.x||enabled[0]<0)return;
 unsigned long long slot=atomicAdd(count,(unsigned long long)1);
 if(slot<(unsigned long long)slots){
  times[2*slot]=clock[0];times[2*slot+1]=clock[1];
  accepted_error[slot]=status[0];
 }
}
extern "C" __global__ void capture_error_blocks(
 const long long* enabled,const unsigned long long* count,
 const double* full,const double* fine,int n,int normn,double atol,double rtol,
 double* block_values,int* block_indices,int slots,int blocks){
 if(enabled[0]<0)return;
 unsigned long long slot=count[0]-1;
 if(slot>=(unsigned long long)slots)return;
 __shared__ double values[256];__shared__ int indices[256];
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 double e=-1.;
 if(i<normn){
  double a=full[i],b=fine[i];
  e=fabs(a-b)/(3.*(atol+rtol*fmax(fabs(a),fabs(b))));
 }
 values[threadIdx.x]=e;indices[threadIdx.x]=i;
 __syncthreads();
 for(int k=128;k>0;k/=2){
  if(threadIdx.x<k){
   double other=values[threadIdx.x+k];int oi=indices[threadIdx.x+k];
   if(other>values[threadIdx.x] ||
      (other==values[threadIdx.x] && oi<indices[threadIdx.x])){
    values[threadIdx.x]=other;indices[threadIdx.x]=oi;
   }
  }
  __syncthreads();
 }
 if(threadIdx.x==0){
  long long offset=(long long)slot*blocks+blockIdx.x;
  block_values[offset]=values[0];block_indices[offset]=indices[0];
 }
}
'''


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(out):
    started=time.monotonic();out=out.resolve()
    need(__debug__ and not out.exists(),'Normal Python and unique run required')
    plan_path=HERE/'ERROR_LIMITER_PLAN_38.json';p=json.loads(plan_path.read_text())
    need(p['schema']=='error_limiter_plan_v1' and sha(Path(__file__))==p['script_sha256'],
         'Plan or script changed')
    actual={x:sha(ROOT/x) for x in p['frozen_inputs_sha256']}
    need(actual==p['frozen_inputs_sha256'],'Input changed')
    import cupy as cp
    sys.path[:0]=[str(EPOCH)]
    import run_set
    from runtime_session import RuntimeSession
    free_before,_=cp.cuda.runtime.memGetInfo()
    context={'installed':0,'ready':0}
    old_init=RuntimeSession.__init__
    def installed(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        context['installed']+=1
        adapter=session.adapter;b=adapter.brain;n=len(b.state)
        need(n==359373,'State layout')
        import graph_core
        cls=graph_core.NativeGraph;old_trial=cls.trial
        context['class']=cls;context['old_trial']=old_trial
        blocks=(n+255)//256
        module=cp.RawModule(code=TRACE,options=('--std=c++11','--fmad=false'))
        mark=module.get_function('mark_error')
        capture=module.get_function('capture_error_blocks')
        enabled=cp.asarray([-1],dtype=cp.int64)
        count=cp.zeros(1,dtype=cp.uint64)
        times=cp.empty((SLOTS,2),dtype=cp.float64)
        reference=cp.empty(SLOTS,dtype=cp.float64)
        values=cp.empty((SLOTS,blocks),dtype=cp.float64)
        indices=cp.empty((SLOTS,blocks),dtype=cp.int32)
        context.update(enabled=enabled,count=count,times=times,reference=reference,
                       values=values,indices=indices,blocks=blocks,norm_size=None)
        def trial(self):
            old_trial(self)
            mark((1,),(1,),(enabled,count,self.clock,self.status,times,reference,
                               np.int32(SLOTS)))
            capture((blocks,),(256,),(enabled,count,self.full,self.fine,
                                      np.int32(self.n),np.int32(self.norm_size),
                                      np.float64(self.atol),np.float64(self.rtol),
                                      values,indices,np.int32(SLOTS),np.int32(blocks)))
        cls.trial=trial
        old_build=adapter.build
        def build(drive,light):
            old_build(drive,light)
            context['norm_size']=adapter.core.norm_size
            need(context['norm_size']==359373,'Norm contract')
            enabled.set(np.asarray([0],dtype=np.int64))
            cp.cuda.get_current_stream().synchronize()
            context['ready']+=1
        adapter.build=build
    RuntimeSession.__init__=installed
    old_argv=sys.argv
    sys.argv=['run_set.py','--out',str(out),'--odor','sham','--engine','causal_cuda',
              '--ms','1','--observe','off','--cuda-profile','off','--profile','off',
              '--kc-capture','off']
    code,error=None,None
    try:code=run_set.main()
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc),
                                      'traceback':traceback.format_exc()}
    finally:
        sys.argv=old_argv;RuntimeSession.__init__=old_init
        if 'class' in context:context['class'].trial=context['old_trial']
    report={'schema':'error_limiter_result_v1','plan_sha256':sha(plan_path),
            'runner_exit_code':code,'error':error,
            'runtime_installed_count':context['installed'],
            'build_ready_count':context['ready'],'status':'INCOMPLETE'}
    if code==0 and error is None and context['ready']==1:
        cp.cuda.get_current_stream().synchronize()
        m=int(context['count'].get()[0])
        need(m==p['expected_trials'] and m<=SLOTS,'Trial count changed')
        times=cp.asnumpy(context['times'][:m])
        reference=cp.asnumpy(context['reference'][:m])
        values=cp.asnumpy(context['values'][:m])
        indices=cp.asnumpy(context['indices'][:m])
        best_block=np.argmax(values,axis=1)
        rows=np.arange(m)
        maxima=values[rows,best_block]
        winners=indices[rows,best_block]
        need(np.isfinite(times).all() and np.isfinite(reference).all() and
             np.isfinite(maxima).all() and np.all((winners>=0)&(winners<context['norm_size'])),
             'Diagnostic nonfinite/index')
        need(np.max(np.abs(maxima-reference))<=p['maximum_error_reconstruction_abs'],
             'Block maxima differ from actual acceptance norm')
        np.savez_compressed(out/'ERROR_LIMITER_ARRAYS.npz',times=times,
                            reference_error=reference,block_values=values,
                            block_indices=indices,winner_indices=winners,
                            reconstructed_error=maxima)
        unique,counts=np.unique(winners,return_counts=True)
        order=np.argsort(-counts,kind='stable')
        n=166700;trans=177758
        regions={'soma':int(np.count_nonzero(winners<n)),
                 'photo_pre_transmission':int(np.count_nonzero((winners>=n)&(winners<trans))),
                 'transmission':int(np.count_nonzero((winners>=trans)&(winners<trans+n))),
                 'specialized_tail':int(np.count_nonzero(winners>=trans+n))}
        report.update(status='COMPLETE_DIAGNOSTIC_ONLY',trials=m,
                      blocks_per_trial=context['blocks'],norm_size=context['norm_size'],
                      limiter_unique_rows=len(unique),limiter_regions=regions,
                      top_limiter_rows=[{'index':int(unique[k]),'trials':int(counts[k])}
                                        for k in order[:16]],
                      h_s_quantiles=np.quantile(times[:,1],[0,.25,.5,.75,1]).tolist(),
                      error_quantiles=np.quantile(maxima,[0,.25,.5,.75,1]).tolist(),
                      reconstruction_max_abs=float(np.max(np.abs(maxima-reference))),
                      arrays_sha256=sha(out/'ERROR_LIMITER_ARRAYS.npz'))
    report['wall_s']=time.monotonic()-started
    report['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    free_after,_=cp.cuda.runtime.memGetInfo()
    report['gpu_allocation_delta_upper_bytes']=int(max(0,free_before-free_after))
    report['disk_bytes']=sum(x.stat().st_size for x in out.rglob('*') if x.is_file()) if out.exists() else 0
    b=p['budget']
    report['budget_ok']=(report['wall_s']<=b['wall_seconds_max'] and
                         report['maxrss_kib']<=b['ram_gib_max']*1024**2 and
                         report['gpu_allocation_delta_upper_bytes']<=b['incremental_vram_gib_max']*1024**3 and
                         report['disk_bytes']<=b['disk_bytes_max'])
    if report['status']=='COMPLETE_DIAGNOSTIC_ONLY' and not report['budget_ok']:
        report['status']='OVER_BUDGET'
    report['scope']='One 1-ms real sham organism with read-only per-trial error-index instrumentation. Does not change integration or prove a faster method.'
    need(out.exists(),'Run did not create output')
    (out/'ERROR_LIMITER_RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--out',type=Path,required=True);a=x.parse_args()
    signal.signal(signal.SIGALRM,
                  lambda *_:(_ for _ in ()).throw(TimeoutError('120s diagnostic budget')))
    signal.alarm(120)
    r=run(a.out)
    print(json.dumps({k:r.get(k) for k in ('status','trials','limiter_unique_rows',
                                         'limiter_regions','top_limiter_rows','wall_s',
                                         'budget_ok','error')},allow_nan=False))
