
extern "C" __global__ void coefficient(
 int n, const long long* ptr, const int* idx, const double* w,
 const double* release, const double* caps, const bool* visual,
 const double* tau, const double* gain, const double* theta,
 const double* drive, const double* photo, double scale, bool connected,
 double* target, double* rate) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(row>=n) return;
 double a=0., b=0.;
 for(long long e=ptr[row]+lane; e<ptr[row+1]; e+=32) {
   int c=idx[e];
   if(visual[row]) {
     double v=(w[e]*scale)*release[c];
     if(v>=0.) a+=v; else b-=v;
   } else if(connected || !visual[c]) a+=w[e]*(release[c]*caps[c]);
 }
 for(int d=16;d>0;d/=2) {
   a+=__shfl_down_sync(0xffffffff,a,d);
   b+=__shfl_down_sync(0xffffffff,b,d);
 }
 if(lane==0) {
   if(visual[row]) {
     a+=photo[row]; double total=1.+a+b;
     target[row]=(.25+a)/total;rate[row]=total/tau[row];
   } else {
     target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));
     rate[row]=1./tau[row];
   }
 }
}
