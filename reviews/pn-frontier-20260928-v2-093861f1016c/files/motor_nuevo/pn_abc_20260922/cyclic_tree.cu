extern "C" __global__ void messages(const long long* steps,const long long* pairs,int start,int count,const double* d,const double* val,const double* b,double* msg,int* flag){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count)return;int t=id+start;const long long*s=steps+6*t;long long i=s[0],j=s[1],k=s[2],e=s[3],f=s[4];double pivot=d[i];if(!isfinite(pivot)||pivot<=0){atomicOr(flag,1);return;}if(j<0)return;
 int ij=pairs[2*e]==i?0:1;double out1=val[2*e+ij],in1=val[2*e+1-ij];msg[6*t]=-in1*out1/pivot;msg[6*t+1]=-in1*b[i]/pivot;if(k<0)return;
 int ik=pairs[2*f]==i?0:1;double out2=val[2*f+ik],in2=val[2*f+1-ik];msg[6*t+2]=-in2*out2/pivot;msg[6*t+3]=-in2*b[i]/pivot;msg[6*t+4]=-in1*out2/pivot;msg[6*t+5]=-in2*out1/pivot;
}
extern "C" __global__ void gather(const long long* targets,const long long* offsets,const long long* slots,int count,int n,const double* msg,double* d,double* val,double* b){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count)return;long long t=targets[id];double v=t<n?d[t]:(t<2*n?b[t-n]:val[t-2*n]);
 for(long long k=offsets[id];k<offsets[id+1];k++)v+=msg[slots[k]];
 if(t<n)d[t]=v;else if(t<2*n)b[t-n]=v;else val[t-2*n]=v;
}
