// One warp owns one CSR row. Both arms use the same FP64 arithmetic and
// lane/warp reduction order; only graph-memory reuse across eight RHS differs.
#include <cuda_runtime.h>

__device__ __forceinline__ double row_reduce(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __dadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

extern "C" __global__ void csr_scalar_8x_reference(
    int n, const long long* __restrict__ indptr,
    const int* __restrict__ indices, const double* __restrict__ weights,
    const double* __restrict__ rhs, double* __restrict__ out,
    int rhs_id) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid >> 5;
    const int lane = tid & 31;
    if (row >= n) return;
    const double* x = rhs + (unsigned long long)rhs_id * n;
    double sum = 0.0;
    for (long long edge = indptr[row] + lane; edge < indptr[row + 1]; edge += 32) {
        const int col = indices[edge];
        const double weight = weights[edge];
        sum = __dadd_rn(sum, __dmul_rn(weight, x[col]));
    }
    sum = row_reduce(sum);
    if (lane == 0) out[(unsigned long long)rhs_id * n + row] = sum;
}

extern "C" __global__ void csr_batched_8(
    int n, const long long* __restrict__ indptr,
    const int* __restrict__ indices, const double* __restrict__ weights,
    const double* __restrict__ rhs, double* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid >> 5;
    const int lane = tid & 31;
    if (row >= n) return;
    const unsigned long long stride = (unsigned long long)n;
    const double *x0 = rhs, *x1 = rhs + stride, *x2 = rhs + 2 * stride,
                 *x3 = rhs + 3 * stride, *x4 = rhs + 4 * stride,
                 *x5 = rhs + 5 * stride, *x6 = rhs + 6 * stride,
                 *x7 = rhs + 7 * stride;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
    for (long long edge = indptr[row] + lane; edge < indptr[row + 1]; edge += 32) {
        const int col = indices[edge];
        const double weight = weights[edge];
        s0 = __dadd_rn(s0, __dmul_rn(weight, x0[col]));
        s1 = __dadd_rn(s1, __dmul_rn(weight, x1[col]));
        s2 = __dadd_rn(s2, __dmul_rn(weight, x2[col]));
        s3 = __dadd_rn(s3, __dmul_rn(weight, x3[col]));
        s4 = __dadd_rn(s4, __dmul_rn(weight, x4[col]));
        s5 = __dadd_rn(s5, __dmul_rn(weight, x5[col]));
        s6 = __dadd_rn(s6, __dmul_rn(weight, x6[col]));
        s7 = __dadd_rn(s7, __dmul_rn(weight, x7[col]));
    }
    s0 = row_reduce(s0); s1 = row_reduce(s1);
    s2 = row_reduce(s2); s3 = row_reduce(s3);
    s4 = row_reduce(s4); s5 = row_reduce(s5);
    s6 = row_reduce(s6); s7 = row_reduce(s7);
    if (lane == 0) {
        out[row] = s0; out[stride + row] = s1;
        out[2 * stride + row] = s2; out[3 * stride + row] = s3;
        out[4 * stride + row] = s4; out[5 * stride + row] = s5;
        out[6 * stride + row] = s6; out[7 * stride + row] = s7;
    }
}
