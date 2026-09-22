"""FP64 fused 17-coordinate midpoint step; unchanged mathematical candidate.

Each thread owns one entire cell. Dense positive-pivot LU is followed by the
original matrix backward-error check. A failed solve is rejected, never clipped.
No fast-math/FMA contraction is requested. This kernel does not own event state.
"""
import numpy as np
import cupy as cp

CODE=r'''
__device__ void rates(double v,double*s,double*t){
 s[0]=1./(1.+exp(-.1121*(v+29.13)));s[1]=1./(1.+exp(.2*(v+47.)));
 s[2]=1./(1.+exp(-.2717*(v+48.77)));s[3]=1./(1.+exp(-.0502*(v+12.85)));
 t[0]=(.1270+3.434/(1.+exp((v+45.35)/5.98)))/1000.;
 t[1]=(.36+exp((v+20.65)/-10.47))/1000.;t[2]=.001;
 t[3]=(2.03+1.96/(1.+exp((v-30.83)/3.12)))/1000.;
}
__device__ bool solve(double*A,double*b,double*x,double*error){
 double original[289],right[17];
 for(int k=0;k<289;k++)original[k]=A[k];for(int i=0;i<17;i++)right[i]=b[i];
 for(int j=0;j<17;j++){
  double p=A[j*17+j];if(!(p>0.)||!isfinite(p))return false;
  for(int i=j+1;i<17;i++){
   double factor=A[i*17+j]/p;
   for(int k=j+1;k<17;k++)A[i*17+k]-=factor*A[j*17+k];
   b[i]-=factor*b[j];
  }
 }
 for(int i=16;i>=0;i--){double v=b[i];for(int j=i+1;j<17;j++)v-=A[i*17+j]*x[j];x[i]=v/A[i*17+i];}
 double amax=0.,bmax=0.,xmax=0.,rmax=0.;
 for(int i=0;i<17;i++){
  double r=-right[i],a=0.;for(int j=0;j<17;j++){r+=original[i*17+j]*x[j];a+=fabs(original[i*17+j]);}
  rmax=fmax(rmax,fabs(r));amax=fmax(amax,a);bmax=fmax(bmax,fabs(right[i]));xmax=fmax(xmax,fabs(x[i]));
 }
 *error=rmax/fmax(amax*xmax+bmax,1e-300);
 return isfinite(*error)&&*error<=1e-12;
}
extern "C" __global__ void midpoint(int count,double dt,double rest,const double*v,const double*g,
 const double*ge,const double*gi,const double*cur,const double*C,const double*G,
 const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
 double*vo,double*go,double*errors){
 int n=blockDim.x*blockIdx.x+threadIdx.x;if(n>=count)return;
 double f[51],A[289],b[17],pred[17],answer[17],s[4],tau[4],gm[4],ratio=2./dt;
 for(int i=0;i<17;i++){
  rates(v[n*17+i]+rest,s,tau);for(int q=0;q<4;q++)gm[q]=s[q]+(g[n*68+i*4+q]-s[q])*exp(-.5*dt/tau[q]);
  f[i*3]=gm[0]*gm[0]*gm[0]*gm[1];f[i*3+1]=gm[2];f[i*3+2]=gm[3]*gm[3]*gm[3]*gm[3];
 }
 for(int stage=0;stage<2;stage++){
  for(int i=0;i<17;i++){
   double rhs=cur[n*17+i];
   for(int z=0;z<4;z++)rhs+=(-rest*ge[n*4+z]+(-68.-rest)*gi[n*4+z])*shuntb[z*17+i];
   for(int z=0;z<51;z++)rhs+=f[z]*ena[z]*chanb[z*17+i];
   if(stage==1)rhs*=2.;
   for(int j=0;j<17;j++){
    int ij=i*17+j;double k=G[ij];
    for(int z=0;z<4;z++)k+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+ij];
    for(int z=0;z<51;z++)k+=f[z]*chanG[z*289+ij];
    A[ij]=k+ratio*C[ij];rhs+=(ratio*C[ij]-(stage==1?k:0.))*v[n*17+j];
   }
   b[i]=rhs;
  }
  double err=0.;if(!solve(A,b,stage==0?pred:answer,&err)){errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
  errors[n*2+stage]=err;
  if(stage==0){
   for(int i=0;i<17;i++){
    rates(pred[i]+rest,s,tau);
    for(int q=0;q<4;q++){
     double old=g[n*68+i*4+q];gm[q]=s[q]+(old-s[q])*exp(-.5*dt/tau[q]);
     go[n*68+i*4+q]=s[q]+(old-s[q])*exp(-dt/tau[q]);
    }
    f[i*3]=gm[0]*gm[0]*gm[0]*gm[1];f[i*3+1]=gm[2];f[i*3+2]=gm[3]*gm[3]*gm[3]*gm[3];
   }
  }
 }
 for(int i=0;i<17;i++)vo[n*17+i]=answer[i];
}
'''
KERNEL=cp.RawKernel(CODE,'midpoint',options=('--fmad=false',))

def step(batch,v,g,ge,gi,dt,current):
    if batch.ports!=17 or v.dtype!=cp.float64 or g.dtype!=cp.float64:raise ValueError('FP64 17-coordinate model required')
    out=cp.empty_like(v);gates=cp.empty_like(g);errors=cp.zeros((len(v),2),dtype=cp.float64)
    KERNEL(((len(v)+31)//32,),(32,), (np.int32(len(v)),np.float64(dt),np.float64(batch.rest),v,g,ge,gi,current,
        batch.C,batch.G,batch.chanG,batch.chanb,batch.shuntG,batch.shuntb,batch.ena,out,gates,errors))
    if not bool(cp.isfinite(out).all()) or not bool(cp.isfinite(gates).all()) or bool(cp.any((gates<0)|(gates>1))) or not bool(cp.isfinite(errors).all()):
        raise FloatingPointError('Fused KC solve or gate state failed')
    return out,gates
