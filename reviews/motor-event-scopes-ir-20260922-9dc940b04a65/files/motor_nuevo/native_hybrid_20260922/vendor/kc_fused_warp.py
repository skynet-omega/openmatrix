"""One warp per KC:17 row owners, full FP64 matrix and residual retained."""
import numpy as np
import cupy as cp
from kc_fused_step import CODE

WARP=CODE.split('extern "C"')[0]+r'''
extern "C" __global__ void warp_midpoint(int count,double dt,double rest,const double*v,const double*g,
 const double*ge,const double*gi,const double*cur,const double*C,const double*G,
 const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
 double*vo,double*go,double*errors){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;
 const unsigned mask=0x1ffff;double A[17],original[17],b,br,x=0.,pred=0.,s[4],tau[4],gm[4];
 double f0,f1,f2,ratio=2./dt,v0=v[n*17+i];
 rates(v0+rest,s,tau);for(int q=0;q<4;q++)gm[q]=s[q]+(g[n*68+i*4+q]-s[q])*exp(-.5*dt/tau[q]);
 f0=gm[0]*gm[0]*gm[0]*gm[1];f1=gm[2];f2=gm[3]*gm[3]*gm[3]*gm[3];
 for(int stage=0;stage<2;stage++){
  double rhs=cur[n*17+i];
  for(int z=0;z<4;z++)rhs+=(-rest*ge[n*4+z]+(-68.-rest)*gi[n*4+z])*shuntb[z*17+i];
  for(int j=0;j<17;j++){
   int ij=i*17+j;double k=G[ij];for(int z=0;z<4;z++)k+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+ij];
   A[j]=k;
  }
  for(int port=0;port<17;port++){
   double ff[3]={__shfl_sync(mask,f0,port),__shfl_sync(mask,f1,port),__shfl_sync(mask,f2,port)};
   for(int q=0;q<3;q++){
    int z=port*3+q;rhs+=ff[q]*ena[z]*chanb[z*17+i];
    for(int j=0;j<17;j++)A[j]+=ff[q]*chanG[z*289+i*17+j];
   }
  }
  if(stage==1)rhs*=2.;
  for(int j=0;j<17;j++){
   double vj=__shfl_sync(mask,v0,j),k=A[j];
   rhs+=(ratio*C[i*17+j]-(stage==1?k:0.))*vj;A[j]=k+ratio*C[i*17+j];original[j]=A[j];
  }
  b=rhs;br=rhs;
  for(int j=0;j<17;j++){
   double pivot=__shfl_sync(mask,A[j],j),bj=__shfl_sync(mask,b,j);
   if(!(pivot>0.)||!isfinite(pivot)){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
   double factor=i>j?A[j]/pivot:0.;
   for(int k=j+1;k<17;k++){double ajk=__shfl_sync(mask,A[k],j);if(i>j)A[k]-=factor*ajk;}
   if(i>j)b-=factor*bj;
  }
  for(int j=16;j>=0;j--){
   if(i==j)x=b/A[j];double xj=__shfl_sync(mask,x,j);if(i<j)b-=A[j]*xj;
  }
  double residual=-br,row=0.;for(int j=0;j<17;j++){double xj=__shfl_sync(mask,x,j);residual+=original[j]*xj;row+=fabs(original[j]);}
  double rmax=fabs(residual),amax=row,bmax=fabs(br),xmax=fabs(x);
  for(int o=16;o>0;o/=2){
   double rr=__shfl_down_sync(mask,rmax,o),aa=__shfl_down_sync(mask,amax,o),bb=__shfl_down_sync(mask,bmax,o),xx=__shfl_down_sync(mask,xmax,o);
   if(i+o<17){rmax=fmax(rmax,rr);amax=fmax(amax,aa);bmax=fmax(bmax,bb);xmax=fmax(xmax,xx);}
  }
  double err=__shfl_sync(mask,rmax/fmax(amax*xmax+bmax,1e-300),0);
  if(i==0)errors[n*2+stage]=err;
  if(!isfinite(err)||err>1e-12){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
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
'''
KERNEL=cp.RawKernel(WARP,'warp_midpoint',options=('--fmad=false',))

def step(batch,v,g,ge,gi,dt,current,*,check=True):
    if batch.ports!=17 or v.dtype!=cp.float64 or g.dtype!=cp.float64:raise ValueError('FP64 17-coordinate model required')
    expected=((batch.n,17),(batch.n,17,4),(batch.n,4),(batch.n,4),(batch.n,17))
    for x,shape in zip((v,g,ge,gi,current),expected):
        if x.shape!=shape or x.dtype!=cp.float64 or not x.flags.c_contiguous:raise ValueError('Invalid fused KC buffer layout')
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite fused KC timestep required')
    out=cp.empty_like(v);gates=cp.empty_like(g);errors=cp.zeros((len(v),2),dtype=cp.float64)
    KERNEL((len(v),),(32,), (np.int32(len(v)),np.float64(dt),np.float64(batch.rest),v,g,ge,gi,current,
        batch.C,batch.G,batch.chanG,batch.chanb,batch.shuntG,batch.shuntb,batch.ena,out,gates,errors))
    if check:
        if not bool(cp.isfinite(errors).all()):raise FloatingPointError('Warp KC solve failed')
        return out,gates
    return out,gates,errors
