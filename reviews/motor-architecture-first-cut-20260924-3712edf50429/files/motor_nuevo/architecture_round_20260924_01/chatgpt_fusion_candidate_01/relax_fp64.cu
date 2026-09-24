
extern "C" __global__ void relax_fp64(const double* z,const double* a,
 const double* b,const double* clock,double scale,int n,double* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){
  double t=__dmul_rn(scale,clock[1]);
  double arg=__dmul_rn(t,b[i]);
  double e=-expm1(arg);
  double d=__dsub_rn(a[i],z[i]);
  out[i]=__dadd_rn(z[i],__dmul_rn(e,d));
 }
}
