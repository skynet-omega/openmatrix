// Prospective FAST-mode operator probe only. This changes arithmetic to FP32.
extern "C" __global__ void coefficient_fast(
 int n, const long long* ptr, const int* idx, const float* w,
 const float* release, const float* caps, const bool* visual,
 const float* tau, const float* gain, const float* theta,
 const float* drive, const float* photo, float scale, bool connected,
 double* target, double* rate, const int* observe_rows, double* observed) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(row>=n) return;
 float a=0.f,b=0.f;
 int observation=(row==observe_rows[0])?0:((row==observe_rows[1])?1:-1);
 float positive=0.f,negative=0.f;
 for(long long e=ptr[row]+lane; e<ptr[row+1]; e+=32) {
   int c=idx[e];
   if(visual[row]) {
     float v=(w[e]*scale)*release[c];
     if(v>=0.f) a+=v; else b-=v;
   } else if(connected || !visual[c]) {
     a+=w[e]*(release[c]*caps[c]);
     if(observation>=0) {
       float term=w[e]*(release[c]*caps[c]);
       if(term>=0.f)positive+=term;else negative+=term;
     }
   }
 }
 for(int d=16;d>0;d/=2) {
   a+=__shfl_down_sync(0xffffffff,a,d);
   b+=__shfl_down_sync(0xffffffff,b,d);
   if(observation>=0) { positive+=__shfl_down_sync(0xffffffff,positive,d);
     negative+=__shfl_down_sync(0xffffffff,negative,d); }
 }
 if(lane==0) {
   if(visual[row]) {
     a+=photo[row]; float total=1.f+a+b;
     target[row]=(double)((.25f+a)/total);
     rate[row]=(double)(total/tau[row]);
   } else {
     target[row]=(double)fmaxf(0.f,tanhf(gain[row]*(a+drive[row]-theta[row])));
     rate[row]=(double)(1.f/tau[row]);
     if(observation>=0) {
       double* out=observed+observation*10;
       out[0]=a;out[1]=positive;out[2]=negative;out[3]=drive[row];
       out[4]=theta[row];out[5]=gain[row];out[6]=target[row];
       out[7]=rate[row];out[8]=tau[row];out[9]=a+drive[row]-theta[row];
     }
   }
 }
}
