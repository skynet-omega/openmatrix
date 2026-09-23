"""Model-supplied ADD/SET events for a two-state affine cascade.

No anatomy, threshold, clipping or inferred reset. The source owns its event map.
This backend admits this mathematical primitive, not arbitrary DAE/chemistry.
"""
from pathlib import Path
import sys,numpy as np,cupy as cp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from event_ports import FilterPorts,CODE
SOURCE=CODE[:CODE.index('extern "C"')]+r'''
extern "C" __global__ void reset_port(double*y,const long long*qr,const long long*sr,const double*q,const double*s,const double*tq,const double*ts,const double*et,const double*ej,const int*counts,const double*post,const int*kind,int n,int width,const double*clock,double fraction){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double t=clock[0]+fraction*clock[1],a=q[i],b=s[i],previous=0.;
 for(int k=0;k<counts[i];k++){
  int p=i*width+k;if(et[p]>t)break;double u=et[p]-previous;
  b=b*exp(-u/ts[0])+a*conv(u,tq[i],ts[0]);a=a*exp(-u/tq[i]);
  a=kind[p]?post[p]:a+ej[p];previous=et[p];
 }
 double u=t-previous;b=b*exp(-u/ts[0])+a*conv(u,tq[i],ts[0]);a=a*exp(-u/tq[i]);
 y[qr[i]]=a;y[sr[i]]=b;
}
'''
class ResetFilterPorts(FilterPorts):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.post=cp.zeros((self.n,self.width));self.kind=cp.zeros((self.n,self.width),dtype=cp.int32)
  self.kernel=cp.RawKernel(SOURCE,'reset_port',options=('--fmad=false',))
 def update(self,w):
  posts=getattr(w,'post_values',[None]*len(w.times))
  if len(posts)!=len(w.times):raise ValueError('Incomplete reset metadata')
  order=np.lexsort((np.arange(len(w.times)),np.asarray(w.times),np.asarray(w.rows)))
  # Deterministic per-source chronological order, including simultaneous events.
  from types import SimpleNamespace
  sorted_w=SimpleNamespace(**{k:getattr(w,k) for k in ('q','s','tau','ts')},times=np.asarray(w.times)[order],rows=np.asarray(w.rows,dtype=np.int64)[order],jumps=np.asarray(w.jumps)[order])
  super().update(sorted_w);post=np.zeros((self.n,self.width));kind=np.zeros_like(post,dtype=np.int32);used=np.zeros(self.n,dtype=int)
  for index in order:
   row=int(w.rows[index]);slot=used[row];value=posts[index]
   if value is not None:
    if not np.isfinite(value):raise ValueError('Nonfinite reset value')
    post[row,slot]=value;kind[row,slot]=1
   used[row]+=1
  self.post.set(post);self.kind.set(kind)
 def project(self,y,clock,fraction):
  out=y.copy();self.kernel(((self.n+255)//256,),(256,),(out,self.qr,self.sr,self.q,self.s,self.tq,self.ts,self.times,self.jumps,self.counts,self.post,self.kind,np.int32(self.n),np.int32(self.width),clock,np.float64(fraction)));return out
