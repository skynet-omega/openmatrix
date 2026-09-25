// Conserved 17-coordinate physical model; generic algebra is in dense_warp.cuh.
#include "dense_warp.cuh"
#include "independent_blocks.cuh"

__device__ void rates(double v,double*s,double*t){
 s[0]=1./(1.+exp(-.1121*(v+29.13)));s[1]=1./(1.+exp(.2*(v+47.)));
 s[2]=1./(1.+exp(-.2717*(v+48.77)));s[3]=1./(1.+exp(-.0502*(v+12.85)));
 t[0]=(.1270+3.434/(1.+exp((v+45.35)/5.98)))/1000.;
 t[1]=(.36+exp((v+20.65)/-10.47))/1000.;t[2]=.001;
 t[3]=(2.03+1.96/(1.+exp((v-30.83)/3.12)))/1000.;
}

__device__ __noinline__ void warp_midpoint(int count,double dt,double rest,const double*v,const double*g,
 const double*ge,const double*gi,const double*cur,const double*C,const double*G,
 const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
 double*vo,double*go,double*errors){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;
 const unsigned mask=0x1ffff;double A[17],x=0.,pred=0.,s[4],tau[4],gm[4];
 double f0,f1,f2,ratio=2./dt,v0=v[n*17+i];
 rates(v0+rest,s,tau);for(int q=0;q<4;q++)gm[q]=s[q]+(g[n*68+i*4+q]-s[q])*exp(-.5*dt/tau[q]);
 f0=gm[0]*gm[0]*gm[0]*gm[1];f1=gm[2];f2=gm[3]*gm[3]*gm[3]*gm[3];
 for(int stage=0;stage<2;stage++){
  double rhs=cur[n*17+i];
  for(int z=0;z<4;z++)rhs+=(-rest*ge[n*4+z]+(-68.-rest)*gi[n*4+z])*shuntb[z*17+i];
  
#pragma unroll
for(int j=0;j<17;j++){
   int ij=i*17+j;double k=G[ij];for(int z=0;z<4;z++)k+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+ij];
   A[j]=k;
  }
  for(int port=0;port<17;port++){
   double ff[3]={__shfl_sync(mask,f0,port),__shfl_sync(mask,f1,port),__shfl_sync(mask,f2,port)};
   for(int q=0;q<3;q++){
    int z=port*3+q;rhs+=ff[q]*ena[z]*chanb[z*17+i];
    
#pragma unroll
for(int j=0;j<17;j++)A[j]+=ff[q]*chanG[z*289+i*17+j];
   }
  }
  if(stage==1)rhs*=2.;
  
#pragma unroll
for(int j=0;j<17;j++){
   double vj=__shfl_sync(mask,v0,j),k=A[j];
   rhs+=(ratio*C[i*17+j]-(stage==1?k:0.))*vj;A[j]=k+ratio*C[i*17+j];
  }
  double err=0.;
  auto answer=neurocore::dense_warp_solve<17>(A,rhs,1e-12);
  bool solved=answer.ok;err=answer.error;x=answer.x;
  if(i==0)errors[n*2+stage]=solved?err:__longlong_as_double(0x7ff0000000000000LL);
  if(!solved)return;
  if(stage==0){
   pred=x;rates(pred+rest,s,tau);for(int q=0;q<4;q++){
    double old=g[n*68+i*4+q];gm[q]=s[q]+(old-s[q])*exp(-.5*dt/tau[q]);go[n*68+i*4+q]=s[q]+(old-s[q])*exp(-dt/tau[q]);
   }
   bool bad=false;for(int q=0;q<4;q++){double val=go[n*68+i*4+q];bad|=!isfinite(val)||val<0.||val>1.;}
   if(__any_sync(mask,bad)){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
   f0=gm[0]*gm[0]*gm[0]*gm[1];f1=gm[2];f2=gm[3]*gm[3]*gm[3]*gm[3];
  }
 }
 if(__any_sync(mask,!isfinite(x))){if(i==0)errors[n*2+1]=__longlong_as_double(0x7ff0000000000000LL);return;}
 vo[n*17+i]=x;
}
// Physical event/filter update compiled separately from numerical trials.
// The adapter declares observation vectors, axonal coordinates and thresholds.
__device__ void commit_event(int n,int dim,int width,int split,const long long*clock,
 const double*v,double rest,const double*observe,double threshold,double prominence,
 const double*caps,const double*tau,double*q,double*last,double*slope0,double*trough,
 long long*counts,long long*clipped,int*event_count,double*event_time,double*event_jump,double*event_post,int*flag){
 int i=blockIdx.x;
 long long ns=split==1?clock[1]/2:clock[1]-clock[1]/2;
 double dt=ns*1e-9,voltage=0.;for(int j=0;j<dim;j++)voltage+=v[i*dim+j]*observe[j];voltage+=rest;
 double slope=voltage-last[i];bool peak=slope0[i]>0&&slope<=0;
 bool event=peak&&last[i]>threshold&&last[i]-trough[i]>=prominence;
 double before=q[i]*exp(-dt/tau[i]);double after=before+(event?1./(caps[i]*tau[i]):0.);
 clipped[i]+=(after>1.);q[i]=fmin(after,1.);counts[i]+=event;
 double jump=q[i]-before;
 if(event){
  int k=event_count[i];if(k>=width){atomicOr(flag,1);}else{
   event_time[i*width+k]=(clock[0]+(split==1?clock[1]/2:clock[1]))*1e-9;
   event_jump[i*width+k]=jump;event_post[i*width+k]=q[i];event_count[i]=k+1;
  }
 }
 if(!isfinite(q[i])||!isfinite(voltage))atomicOr(flag,2);
 trough[i]=peak?voltage:fmin(trough[i],voltage);last[i]=voltage;slope0[i]=slope;
}
__device__ void commit_axon(int n,int dim,int axons,int split,const long long*clock,
 const int*coords,const double*v,double rest,double ts,const double*caps,const double*tau,const double*gain,
 double threshold,double prominence,double*q,double*s,double*last,double*slope0,double*trough,long long*counts,long long*clipped,int*flag){
 int j=blockIdx.x*axons+threadIdx.x;int cell=j/axons,port=j%axons;
 double dt=(split==1?clock[1]/2:clock[1]-clock[1]/2)*1e-9;
 double tq=tau[cell],eq=exp(-dt/tq),es=exp(-dt/ts),factor;
 if(fabs(tq-ts)<=1e-15)factor=dt/ts*es;else factor=tq*(eq-es)/(tq-ts);
 double voltage=v[cell*dim+coords[port]]+rest,slope=voltage-last[j];bool peak=slope0[j]>0.&&slope<=0.;
 bool event=peak&&last[j]>threshold&&last[j]-trough[j]>=prominence;
 double snext=s[j]*es+gain[cell]*q[j]*factor,qn=q[j]*eq+(event?1./(caps[cell]*tq):0.);
 clipped[j]+=(qn>1.);q[j]=fmin(qn,1.);s[j]=snext;counts[j]+=event;
 if(!isfinite(q[j])||!isfinite(s[j])||!isfinite(voltage))atomicOr(flag,4);
 trough[j]=peak?voltage:fmin(trough[j],voltage);last[j]=voltage;slope0[j]=slope;
}
struct SpatialModel {
 int count,n,i;double rest;
 double*v,*g;const double *ge,*gi,*cur,*C,*G,*chanG,*chanb,*shuntG,*shuntb,*ena;
 double *vf,*va,*vb,*gf,*ga,*gb,*errors;long long*clock;
 const double*observe,*caps,*tau;double*q,*last,*slope,*trough;long long*spikes,*clipped;
 int*ec;double*et,*ej,*ep;int*flag;const int*coords;double ts;const double*gain;
 double*aq,*as,*alast,*aslope,*atrough;long long*acount,*aclip;
 __device__ double trial(long long h){
  const unsigned mask=0x1ffff;
  if(i<2)errors[n*2+i]=0.;__syncwarp(mask);
  warp_midpoint(count,h*1e-9,rest,v,g,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,vf,gf,errors);
  __syncwarp(mask);bool bad=!isfinite(errors[n*2])||!isfinite(errors[n*2+1]);
  warp_midpoint(count,(h/2)*1e-9,rest,v,g,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,va,ga,errors);
  __syncwarp(mask);bad|=!isfinite(errors[n*2])||!isfinite(errors[n*2+1]);
  warp_midpoint(count,(h-h/2)*1e-9,rest,va,ga,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,vb,gb,errors);
  __syncwarp(mask);bad|=!isfinite(errors[n*2])||!isfinite(errors[n*2+1]);
  double a=double(h/2)/double(h),b=1.-a,cubes=a*a*a+b*b*b,factor=cubes/(1.-cubes);
  double e=fabs(vb[n*17+i]-vf[n*17+i])*factor/2e-5;
  for(int k=0;k<4;k++)e=fmax(e,fabs(gb[n*68+i*4+k]-gf[n*68+i*4+k])*factor/2e-7);
  bad|=!isfinite(e);
  if(__any_sync(mask,bad))return __longlong_as_double(0x7ff0000000000000LL);
  for(int o=16;o>0;o/=2){double x=__shfl_down_sync(mask,e,o);if(i+o<17)e=fmax(e,x);}
  return __shfl_sync(mask,e,0);
 }
 __device__ int commit(long long used,long long h){
  const unsigned mask=0x1ffff;
  if(i==0){clock[n*2]=used;clock[n*2+1]=h;}__syncwarp(mask);
  for(int split=1;split<=2;split++){
   const double*vv=split==1?va:vb;
   if(i==0)commit_event(count,17,8,split,clock+n*2,vv,rest,observe,-40.,20.,caps,tau,q,last,slope,trough,spikes,clipped,ec,et,ej,ep,flag+n);
   if(i<12)commit_axon(count,17,12,split,clock+n*2,coords,vv,rest,ts,caps,tau,gain,-40.,20.,aq,as,alast,aslope,atrough,acount,aclip,flag+n);
   __syncwarp(mask);
  }
  v[n*17+i]=vb[n*17+i];for(int k=0;k<4;k++)g[n*68+i*4+k]=gb[n*68+i*4+k];
  __syncwarp(mask);return flag[n];
 }
};
extern "C" __global__ void cell_epoch(int count,long long duration,long long maximum,double rest,
 double*v,double*g,const double*ge,const double*gi,const double*cur,const double*C,const double*G,
 const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
 double*vf,double*va,double*vb,double*gf,double*ga,double*gb,double*errors,long long*clock,
 const double*observe,const double*caps,const double*tau,double*q,double*last,double*slope,double*trough,
 long long*spikes,long long*clipped,int*ec,double*et,double*ej,double*ep,int*flag,const int*coords,double ts,const double*gain,
 double*aq,double*as,double*alast,double*aslope,double*atrough,long long*acount,long long*aclip,
 long long*statistics,double*max_error,int*status){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;
 SpatialModel model={count,n,i,rest,v,g,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,vf,va,vb,gf,ga,gb,errors,clock,
 observe,caps,tau,q,last,slope,trough,spikes,clipped,ec,et,ej,ep,flag,coords,ts,gain,aq,as,alast,aslope,atrough,acount,aclip};
 int code=advance_independent_block(model,duration,maximum,200,10000,statistics+n*3,max_error+n);
 if(i==0)status[n]=code;
}
