// Read-only sideband of values already calculated by the physical publisher.
// One writer per (selected cell, compartment); no model decision reads it.
__device__ double* kd_values = nullptr;
__device__ long long* kd_meta = nullptr;
__device__ int* kd_counts = nullptr;
__device__ int* kd_rows = nullptr;
__device__ int* kd_overflow = nullptr;
__device__ int kd_enabled = 0, kd_capacity = 0, kd_call = 0;
__device__ long long kd_origin = 0;
extern "C" __global__ void kd_bind(double* v,long long* m,int* c,int* rows,int* overflow,int capacity){
 if(blockIdx.x==0 && threadIdx.x==0){kd_values=v;kd_meta=m;kd_counts=c;kd_rows=rows;kd_overflow=overflow;kd_capacity=capacity;kd_enabled=0;}
}
extern "C" __global__ void kd_begin(int enabled,int call,long long origin){
 if(blockIdx.x==0 && threadIdx.x==0){kd_enabled=enabled;kd_call=call;kd_origin=origin;}
}
__device__ __forceinline__ void kd_save(int cell,int port,int split,const long long* clock,
 double voltage,double last,double increment,double old_increment,double trough,
 double q,double s,bool peak,bool event,long long count){
 if(!kd_enabled)return;
 int selected=kd_rows[cell];if(selected<0)return;
 int channel=selected*13+port, index=kd_counts[channel]++;
 if(index>=kd_capacity){atomicOr(kd_overflow,1);return;}
 int base=(channel*kd_capacity+index)*10;
 kd_values[base]=voltage;kd_values[base+1]=last;kd_values[base+2]=increment;
 kd_values[base+3]=old_increment;kd_values[base+4]=trough;
 kd_values[base+5]=q;kd_values[base+6]=s;kd_values[base+7]=peak;
 kd_values[base+8]=event;kd_values[base+9]=(double)count;
 int mb=(channel*kd_capacity+index)*4;
 kd_meta[mb]=kd_call;kd_meta[mb+1]=kd_origin+clock[0]+(split==1?clock[1]/2:clock[1]);
 kd_meta[mb+2]=clock[1];kd_meta[mb+3]=split;
}
