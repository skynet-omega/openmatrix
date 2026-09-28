extern "C" __global__ void eliminate_level(const long long* steps,const long long* pairs,const long long* order,int offset,int count,double* d,double* val,double* b,int* flag){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;const long long* s=steps+6*order[offset+t];long long i=s[0],j=s[1],k=s[2],e=s[3],f=s[4],c=s[5];double pivot=d[i];if(!isfinite(pivot)||pivot<=0){atomicOr(flag,1);return;}if(j<0)return;
 int ij=pairs[2*e]==i?0:1;double out1=val[2*e+ij],in1=val[2*e+1-ij];d[j]-=in1*out1/pivot;b[j]-=in1*b[i]/pivot;if(k<0)return;
 int ik=pairs[2*f]==i?0:1;double out2=val[2*f+ik],in2=val[2*f+1-ik];d[k]-=in2*out2/pivot;b[k]-=in2*b[i]/pivot;val[2*c]-=in1*out2/pivot;val[2*c+1]-=in2*out1/pivot;
}
extern "C" __global__ void back_level(const long long* steps,const long long* pairs,const long long* order,int offset,int count,const double* d,const double* val,const double* b,double* x){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;const long long*s=steps+6*order[offset+t];long long i=s[0],j=s[1],k=s[2],e=s[3],f=s[4];double v=b[i];if(j>=0)v-=val[2*e+(pairs[2*e]==i?0:1)]*x[j];if(k>=0)v-=val[2*f+(pairs[2*f]==i?0:1)]*x[k];x[i]=v/d[i];
}
extern "C" __global__ void core_solve(const long long* core,const long long* edges,int n,int ne,const double* d,const double* val,const double* b,double* x,double* matrix,double* y,int* flag){
 if(threadIdx.x||blockIdx.x)return;
 for(int i=0;i<n;i++){y[i]=b[core[i]];for(int j=0;j<n;j++)matrix[i*n+j]=i==j?d[core[i]]:0.;}
 for(int e=0;e<ne;e++){long long id=edges[3*e],i=edges[3*e+1],j=edges[3*e+2];matrix[i*n+j]=val[2*id];matrix[j*n+i]=val[2*id+1];}
 for(int k=0;k<n;k++){
  int pivot=k;for(int i=k+1;i<n;i++)if(fabs(matrix[i*n+k])>fabs(matrix[pivot*n+k]))pivot=i;
  if(!isfinite(matrix[pivot*n+k])||matrix[pivot*n+k]==0){atomicOr(flag,2);return;}
  if(pivot!=k){for(int j=k;j<n;j++){double z=matrix[k*n+j];matrix[k*n+j]=matrix[pivot*n+j];matrix[pivot*n+j]=z;}double z=y[k];y[k]=y[pivot];y[pivot]=z;}
  for(int i=k+1;i<n;i++){double factor=matrix[i*n+k]/matrix[k*n+k];for(int j=k+1;j<n;j++)matrix[i*n+j]-=factor*matrix[k*n+j];y[i]-=factor*y[k];}
 }
 for(int i=n-1;i>=0;i--){double t=y[i];for(int j=i+1;j<n;j++)t-=matrix[i*n+j]*x[core[j]];x[core[i]]=t/matrix[i*n+i];if(!isfinite(x[core[i]]))atomicOr(flag,4);}
}
