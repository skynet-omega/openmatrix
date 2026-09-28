extern "C" __global__ void core_parallel(const long long* core,const long long* edges,int n,int ne,const double* d,const double* val,const double* b,double* x,int* flag){
 extern __shared__ double memory[];double* A=memory;double* y=A+n*n;__shared__ int pivot;__shared__ int bad;int tid=threadIdx.x;
 for(int k=tid;k<n*n;k+=blockDim.x)A[k]=0.;__syncthreads();if(tid<n){A[tid*n+tid]=d[core[tid]];y[tid]=b[core[tid]];}
 __syncthreads();
 for(int e=tid;e<ne;e+=blockDim.x){long long id=edges[3*e],i=edges[3*e+1],j=edges[3*e+2];A[i*n+j]=val[2*id];A[j*n+i]=val[2*id+1];}
 __syncthreads();
 for(int k=0;k<n;k++){
  if(tid==0){pivot=k;for(int i=k+1;i<n;i++)if(fabs(A[i*n+k])>fabs(A[pivot*n+k]))pivot=i;bad=!isfinite(A[pivot*n+k])||A[pivot*n+k]==0.;if(bad)atomicOr(flag,2);}
  __syncthreads();if(bad)return;
  if(pivot!=k){for(int j=k+tid;j<n;j+=blockDim.x){double z=A[k*n+j];A[k*n+j]=A[pivot*n+j];A[pivot*n+j]=z;}if(tid==0){double z=y[k];y[k]=y[pivot];y[pivot]=z;}}
  __syncthreads();
  if(tid>k&&tid<n){double f=A[tid*n+k]/A[k*n+k];for(int j=k+1;j<n;j++)A[tid*n+j]-=f*A[k*n+j];y[tid]-=f*y[k];}
  __syncthreads();
 }
 if(tid==0)for(int i=n-1;i>=0;i--){double t=y[i];for(int j=i+1;j<n;j++)t-=A[i*n+j]*x[core[j]];x[core[i]]=t/A[i*n+i];if(!isfinite(x[core[i]]))atomicOr(flag,4);}
}
