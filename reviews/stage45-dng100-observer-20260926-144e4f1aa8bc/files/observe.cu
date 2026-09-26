// Diagnostic writes only. All neural outputs belong to the existing operator.
#define STAGES 4
#define CELLS 2
#define FIELDS 16
#define WIDTH 140
extern "C" __global__ void capture_rhs(
 const int* rows, const double* y, const double* target, const double* rate,
 const double* rhs, const double* clock, double fraction, int stage,
 const double* csr, double* scratch) {
 int cell=threadIdx.x;
 if(cell>=CELLS)return;
 const int row=rows[cell];
 double* o=scratch+(stage*CELLS+cell)*FIELDS;
 o[0]=y[row];
 for(int j=0;j<10;++j)o[j+1]=csr[cell*10+j];
 o[11]=target[row];o[12]=rate[row];o[13]=rhs[row];
 o[14]=clock[0]+fraction*clock[1];o[15]=fraction;
}
extern "C" __global__ void capture_trial(
 const int* rows,const double* scratch,const double* clock,const double* status,
 const double* before,const double* after,unsigned long long* count,
 double* records,int capacity) {
 __shared__ unsigned long long slot;
 if(threadIdx.x==0)slot=*count;
 __syncthreads();
 int t=threadIdx.x;
 if(slot<(unsigned long long)capacity) {
  double* out=records+slot*WIDTH;
  if(t<STAGES*CELLS*FIELDS)out[t]=scratch[t];
  if(t==128)out[t]=clock[0];
  if(t==129)out[t]=clock[1];
  if(t==130)out[t]=clock[2];
  if(t==131)out[t]=status[0];
  if(t==132)out[t]=status[1];
  if(t==133)out[t]=status[2];
  if(t==134)out[t]=before[rows[0]];
  if(t==135)out[t]=before[rows[1]];
  if(t==136)out[t]=after[rows[0]];
  if(t==137)out[t]=after[rows[1]];
  if(t==138)out[t]=status[0]<=1.0 && status[1]==0.0 && status[2]==0.0;
  if(t==139)out[t]=(double)slot;
 }
 __syncthreads();
 if(t==0)*count=slot+1;
}
