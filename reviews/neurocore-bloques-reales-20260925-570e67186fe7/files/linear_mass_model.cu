// A second model of the same block runtime: M dy/dt = f - K y.
// M and K are full dense matrices, shared by the batch. No neuronal anatomy.
#include "dense_warp.cuh"
#include "independent_blocks.cuh"

template<int N> struct LinearMass {
 int n,i; double*state; const double *mass,*stiffness,*forcing;
 double atol,rtol,fine;
 static constexpr unsigned mask=0xffffffffu>>(32-N);
 __device__ neurocore::DenseWarpResult step(double initial,long long h){
  double dt=double(h)*1e-9,half=.5*dt,row[N],rhs=dt*forcing[n*N+i];
  for(int j=0;j<N;j++){
   double m=mass[i*N+j],k=stiffness[i*N+j];
   row[j]=m+half*k;
   rhs+=(m-half*k)*__shfl_sync(mask,initial,j);
  }
  return neurocore::dense_warp_solve<N>(row,rhs,1e-12);
 }
 __device__ double trial(long long h){
  double initial=state[n*N+i];
  auto full=step(initial,h);if(!full.ok)return CUDART_INF;
  auto half=step(initial,h/2);if(!half.ok)return CUDART_INF;
  auto last=step(half.x,h-h/2);if(!last.ok)return CUDART_INF;
  fine=last.x;
  double a=double(h/2)/double(h),b=1.-a,cubes=a*a*a+b*b*b;
  double e=fabs(fine-full.x)*(cubes/(1.-cubes))/(atol+rtol*fmax(fabs(fine),fabs(full.x)));
  if(__any_sync(mask,!isfinite(e)))return CUDART_INF;
  for(int offset=16;offset;offset/=2){
   int source=i+offset<N?i+offset:i;
   double other=__shfl_sync(mask,e,source);
   if(i+offset<N)e=fmax(e,other);
  }
  return __shfl_sync(mask,e,0);
 }
 __device__ int commit(long long,long long){state[n*N+i]=fine;__syncwarp(mask);return 0;}
};

extern "C" __global__ void linear_mass3(int count,long long duration,long long maximum,
 double*state,const double*mass,const double*stiffness,const double*forcing,
 double atol,double rtol,long long*statistics,double*max_error,int*status){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=3)return;
 LinearMass<3> model{n,i,state,mass,stiffness,forcing,atol,rtol,0.};
 int code=advance_independent_block(model,duration,maximum,2,10000,statistics+n*3,max_error+n);
 if(i==0)status[n]=code;
}
