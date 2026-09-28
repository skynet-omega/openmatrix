// Model-independent mirror; fixed allocation is safe for CUDA Graph replay.
extern "C" __global__ void refresh_indexed(
    long long n, const long long* positions, const double* source, float* target) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        long long p=positions[i];
        target[p]=(float)source[p];
    }
}
extern "C" __global__ void audit_mirror(
    long long n, const double* source, const float* target,
    unsigned long long* failures, unsigned long long* uses) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0) atomicAdd(uses,1ULL);
    if(i<n && target[i]!=(float)source[i]) atomicAdd(failures,1ULL);
}

