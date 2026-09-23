"""A non-neural model uses the same device controller without changing it."""
from pathlib import Path
import numpy as np,cupy as cp,json
H=Path(__file__).resolve().parent
code=(H/'block_runtime.hpp').read_text()+r'''
struct Decay {
 double*state;const double*rates;double trial_value;int n,i;
 __device__ double poly(double x,double r,double h){return x*(1-r*h+.5*r*r*h*h);}
 __device__ double trial(long long ns){
  double h=ns*1e-9,x=state[n*3+i],r=rates[n*3+i];
  double full=poly(x,r,h),a=poly(x,r,(ns/2)*1e-9);
  trial_value=poly(a,r,(ns-ns/2)*1e-9);
  double e=fabs(trial_value-full)/(3.e-10);
  for(int j=0;j<3;j++)e=fmax(e,__shfl_sync(7,e,j));return e;
 }
 __device__ int commit(long long,long long){state[n*3+i]=trial_value;__syncwarp(7);return 0;}
};
extern "C" __global__ void run(double*x,const double*r,long long*stats,double*error,int*status){
 int n=blockIdx.x,i=threadIdx.x;if(i>=3)return;
 Decay model={x,r,0.,n,i};int code=advance_independent_block(model,125000,25000,200,10000,stats+n*3,error+n);
 if(i==0)status[n]=code;
}
'''
k=cp.RawKernel(code,'run',options=('--fmad=false',));rates=np.geomspace(1.,2000.,192).reshape(64,3)
x=cp.ones((64,3));r=cp.asarray(rates);stats=cp.zeros((64,3),dtype=cp.int64);error=cp.zeros(64);status=cp.zeros(64,dtype=cp.int32)
k((64,),(32,),(x,r,stats,error,status));actual=x.get();codes=status.get();max_error=float(abs(actual-np.exp(-rates*125e-6)).max())
if np.any(codes) or max_error>1e-7:raise RuntimeError('Independent decay model failed: '+str((codes.tolist(),max_error)))
report={'status':'PASS','blocks':64,'state_dimension':3,'analytic_max_abs':max_error,'accepted_range':stats.get()[:,0][[0,-1]].tolist(),'core_unchanged':True}
(H/'SCHEDULER_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
