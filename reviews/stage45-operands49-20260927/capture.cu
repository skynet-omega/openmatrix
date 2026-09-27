// Read-only copies. Records own their memory and never feed the CNS.
#define CELLS 6
#define STAGES 4
#define FIELDS 16
#define SCALAR 384
#define WIDTH 404
extern "C" __global__ void capture_rhs(
 const int* rows,const double* y,const double* target,const double* rate,
 const double* rhs,const double* clock,double fraction,int stage,
 const double* csr,double* scratch) {
 int c=threadIdx.x;if(c>=CELLS)return;
 double* o=scratch+(stage*CELLS+c)*FIELDS;
 o[0]=y[rows[c]];
 for(int j=0;j<10;j++)o[j+1]=csr[c*10+j];
 o[11]=target[rows[c]];o[12]=rate[rows[c]];o[13]=rhs[rows[c]];
 o[14]=clock[0]+fraction*clock[1];o[15]=fraction;
}
extern "C" __global__ void copy_stage(
 const float* operands,float* stages,int size,int stage) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<size)stages[stage*size+i]=operands[i];
}
extern "C" __global__ void copy_trial_operands(
 const float* stages,const unsigned long long* count,float* records,
 int size,int capacity) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long slot=*count;
 if(slot<(unsigned long long)capacity && i<size)records[slot*size+i]=stages[i];
}
extern "C" __global__ void capture_trial(
 const int* rows,const double* scratch,const double* clock,const double* status,
 const double* before,const double* after,unsigned long long* count,
 double* records,int capacity) {
 __shared__ unsigned long long slot;
 if(threadIdx.x==0)slot=*count;
 __syncthreads();
 if(slot<(unsigned long long)capacity) {
   double* out=records+slot*WIDTH;
   for(int t=threadIdx.x;t<SCALAR;t+=blockDim.x)out[t]=scratch[t];
   int t=threadIdx.x;
   if(t<3){out[SCALAR+t]=clock[t];out[SCALAR+3+t]=status[t];}
   if(t<CELLS){out[SCALAR+6+t]=before[rows[t]];out[SCALAR+12+t]=after[rows[t]];}
   if(t==0){out[402]=status[0]<=1. && status[1]==0. && status[2]==0.;out[403]=(double)slot;}
 }
 __syncthreads();
 if(threadIdx.x==0)*count=slot+1;
}
