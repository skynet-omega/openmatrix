// FP64 recurrent base model. Every stage reads live transmission, never saved outputs.
__device__ double release_q(double q,bool visual) {
 return visual?fmin(1.,fmax(0.,2.*q-.375)):q;
}
extern "C" __global__ void rhs(int n,const double* y,const bool* visual,const double* target,const double* rate,double* f) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 f[i]=rate[i]*(target[i]-y[i]);f[n+i]=200.*(release_q(y[i],visual[i])-y[n+i]);
}
extern "C" __global__ void stage(int m,const double* y,const double* k,double h,double* z) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)z[i]=y[i]+h*k[i];
}
extern "C" __global__ void finish(int m,double* y,const double* a,const double* b,const double* c,const double* d,double h) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)y[i]+=h/6.*(a[i]+2.*b[i]+2.*c[i]+d[i]);
}
extern "C" __global__ void trapezoid(int n,const double* y,const double* f0,const bool* visual,const double* target,const double* rate,double h,double* z) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double q=(y[i]+.5*h*(f0[i]+rate[i]*target[i]))/(1.+.5*h*rate[i]);
 z[i]=q;z[n+i]=(y[n+i]+.5*h*(f0[n+i]+200.*release_q(q,visual[i])))/(1.+100.*h);
}
extern "C" __global__ void residual(int m,const double* y,const double* z,const double* f0,const double* f1,double h,double* maxima) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)maxima[i]=fmax(maxima[i],fabs(z[i]-y[i]-.5*h*(f0[i]+f1[i])));
}
extern "C" __global__ void sums(int n,const long long* ptr,const int* idx,const double* w,const double* s,const double* caps,const bool* visual,double scale,bool connected,double* aa,double* bb) {
 int r=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(r>=n)return;
 double a=0.,b=0.;for(long long e=ptr[r]+lane;e<ptr[r+1];e+=32){int c=idx[e];if(visual[r]){double v=(w[e]*scale)*s[c];if(v>=0)a+=v;else b-=v;}else if(connected||!visual[c])a+=w[e]*(s[c]*caps[c]);}
 for(int d=16;d>0;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);b+=__shfl_down_sync(0xffffffff,b,d);}if(lane==0){aa[r]=a;bb[r]=b;}
}
extern "C" __global__ void scatter(int n,const int* ptr,const int* row,const double* w,const double* s,double* previous,const double* caps,const bool* visual,double scale,bool connected,double quantum,double* aa,double* bb,unsigned long long* counts) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;double delta=s[c]-previous[c];if(fabs(delta)<=quantum)return;
 previous[c]=s[c];atomicAdd(counts,1ULL);atomicAdd(counts+1,(unsigned long long)(ptr[c+1]-ptr[c]));
 for(int e=ptr[c];e<ptr[c+1];e++){int r=row[e];if(visual[r]){double v=w[e]*scale*delta;if(w[e]>=0)atomicAdd(aa+r,v);else atomicAdd(bb+r,-v);}else if(connected||!visual[c])atomicAdd(aa+r,w[e]*(delta*caps[c]));}
}
extern "C" __global__ void cached_coefficient(int n,const double* aa,const double* bb,const bool* visual,const double* tau,const double* gain,const double* theta,const double* drive,const double* photo,double* target,double* rate) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;if(visual[r]){double a=aa[r]+photo[r],total=1.+a+bb[r];target[r]=(.25+a)/total;rate[r]=total/tau[r];}else {target[r]=fmax(0.,tanh(gain[r]*(aa[r]+drive[r]-theta[r])));rate[r]=1./tau[r];}
}
