"""Generic timestamped jump ports for q'=-q/tq, s'=(q-s)/ts.

Events come from the physical state owner. Capacity overflow rejects the epoch;
no event is dropped. The port has no anatomical knowledge or fitted parameters.
"""
import numpy as np
import cupy as cp

CODE=r'''
__device__ double conv(double t,double tq,double ts){
 double z=t*(1/ts-1/tq),den=1-ts/tq;
 if(fabs(z)<1e-5)return t/ts*exp(-t/ts)*(1+z/2+z*z/6+z*z*z/24);
 return (z>=0?exp(-t/tq)*(-expm1(-fmax(z,0.))):exp(-t/ts)*expm1(fmin(z,0.)))/(den==0?1.:den);
}
extern "C" __global__ void port(double*y,const long long*qr,const long long*sr,const double*q,const double*s,const double*tq,const double*ts,const double*et,const double*ej,const bool*setop,const double*post,const int*counts,int n,int width,const double*clock,double fraction){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double t=clock[0]+fraction*clock[1];
 double a=q[i]*exp(-t/tq[i]),b=s[i]*exp(-t/ts[0])+q[i]*conv(t,tq[i],ts[0]);
 bool has_set=false;
 for(int k=0;k<counts[i];k++){int p=i*width+k;if(et[p]<=t){double u=t-et[p];a+=ej[p]*exp(-u/tq[i]);b+=ej[p]*conv(u,tq[i],ts[0]);}}
 for(int k=0;k<counts[i];k++){int p=i*width+k;if(et[p]<=t&&setop[p])has_set=true;}
 if(has_set){
   double qc=q[i],sc=s[i],prev=0.;
   for(int k=0;k<counts[i];k++){
     int p=i*width+k;double mark=et[p];if(mark>t)break;
     double dt=mark-prev;
     sc=sc*exp(-dt/ts[0])+qc*conv(dt,tq[i],ts[0]);
     qc=qc*exp(-dt/tq[i]);
     qc=setop[p]?post[p]:qc+ej[p];
     prev=mark;
   }
   double dt=t-prev;
   a=qc*exp(-dt/tq[i]);
   b=sc*exp(-dt/ts[0])+qc*conv(dt,tq[i],ts[0]);
 }
 y[qr[i]]=a;y[sr[i]]=b;
}
'''
class FilterPorts:
 def __init__(self,q_rows,s_rows,width=8):
  self.n=len(q_rows);self.width=width
  self.qr=cp.asarray(q_rows,dtype=cp.int64);self.sr=cp.asarray(s_rows,dtype=cp.int64)
  self.q=cp.zeros(self.n);self.s=cp.zeros(self.n);self.tq=cp.ones(self.n);self.ts=cp.ones(1)
  self.times=cp.zeros((self.n,width));self.jumps=cp.zeros_like(self.times)
  self.sets=cp.zeros((self.n,width),dtype=cp.bool_);self.posts=cp.zeros_like(self.times)
  self.counts=cp.zeros(self.n,dtype=cp.int32)
  self.kernel=cp.RawKernel(CODE,'port',options=('--fmad=false',))
 def update(self,w):
  count=np.bincount(np.asarray(w.rows,dtype=np.int64),minlength=self.n).astype(np.int32)
  if len(w.q)!=self.n or np.any(count>self.width):raise ValueError('Event port layout/capacity exceeded')
  t=np.zeros((self.n,self.width));j=np.zeros_like(t);post=np.zeros_like(t);sets=np.zeros_like(t,dtype=bool);used=np.zeros(self.n,dtype=int)
  order=sorted(range(len(w.times)),key=lambda k:(w.rows[k],w.times[k],k))
  for k in order:
   tt,rr,jj=w.times[k],w.rows[k],w.jumps[k];p=used[rr]
   t[rr,p]=tt;j[rr,p]=jj;sets[rr,p]=np.isfinite(w.posts[k]);post[rr,p]=w.posts[k] if sets[rr,p] else 0.;used[rr]+=1
  for dst,src in ((self.q,w.q),(self.s,w.s),(self.tq,w.tau),(self.ts,np.array([w.ts])),(self.times,t),(self.jumps,j),(self.sets,sets),(self.posts,post),(self.counts,count)):
   dst.set(np.ascontiguousarray(src))
 def project(self,y,clock,fraction):
  out=y.copy()
  self.kernel(((self.n+255)//256,),(256,),(out,self.qr,self.sr,self.q,self.s,self.tq,self.ts,self.times,self.jumps,self.sets,self.posts,self.counts,np.int32(self.n),np.int32(self.width),clock,np.float64(fraction)))
  return out
