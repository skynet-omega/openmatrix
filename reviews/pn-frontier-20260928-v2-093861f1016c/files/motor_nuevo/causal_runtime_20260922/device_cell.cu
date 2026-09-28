struct SpatialModel {
 int count,n,i;double rest;
 double*v,*g;const double *ge,*gi,*cur,*C,*G,*chanG,*chanb,*shuntG,*shuntb,*ena;
 double *vf,*va,*vb,*gf,*ga,*gb,*errors;long long*clock;
 const double*observe,*caps,*tau;double*q,*last,*slope,*trough;long long*spikes,*clipped;
 int*ec;double*et,*ej;int*flag;const int*coords;double ts;const double*gain;
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
   if(i==0)commit_event(count,17,8,split,clock+n*2,vv,rest,observe,-40.,20.,caps,tau,q,last,slope,trough,spikes,clipped,ec,et,ej,flag+n);
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
 long long*spikes,long long*clipped,int*ec,double*et,double*ej,int*flag,const int*coords,double ts,const double*gain,
 double*aq,double*as,double*alast,double*aslope,double*atrough,long long*acount,long long*aclip,
 long long*statistics,double*max_error,int*status){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;
 SpatialModel model={count,n,i,rest,v,g,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,vf,va,vb,gf,ga,gb,errors,clock,
 observe,caps,tau,q,last,slope,trough,spikes,clipped,ec,et,ej,flag,coords,ts,gain,aq,as,alast,aslope,atrough,acount,aclip};
 int code=advance_independent_block(model,duration,maximum,200,10000,statistics+n*3,max_error+n);
 if(i==0)status[n]=code;
}
