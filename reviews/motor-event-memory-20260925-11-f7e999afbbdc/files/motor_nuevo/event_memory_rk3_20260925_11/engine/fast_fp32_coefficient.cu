// Prospective FAST-mode operator probe only. This changes arithmetic to FP32.
extern "C" __global__ void coefficient_fast(
 int n, const long long* ptr, const int* idx, const float* w,
 const float* release, const float* caps, const bool* visual,
 const float* tau, const float* gain, const float* theta,
 const float* drive, const float* photo, float scale, bool connected,
 double* target, double* rate) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(row>=n) return;
 float a=0.f,b=0.f;
 for(long long e=ptr[row]+lane; e<ptr[row+1]; e+=32) {
   int c=idx[e];
   if(visual[row]) {
     float v=(w[e]*scale)*release[c];
     if(v>=0.f) a+=v; else b-=v;
   } else if(connected || !visual[c]) a+=w[e]*(release[c]*caps[c]);
 }
 for(int d=16;d>0;d/=2) {
   a+=__shfl_down_sync(0xffffffff,a,d);
   b+=__shfl_down_sync(0xffffffff,b,d);
 }
 if(lane==0) {
   if(visual[row]) {
     a+=photo[row]; float total=1.f+a+b;
     target[row]=(double)((.25f+a)/total);
     rate[row]=(double)(total/tau[row]);
   } else {
     target[row]=(double)fmaxf(0.f,tanhf(gain[row]*(a+drive[row]-theta[row])));
     rate[row]=(double)(1.f/tau[row]);
   }
 }
}
