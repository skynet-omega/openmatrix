#pragma once

#include <cuda_runtime.h>
#ifndef __CUDACC_RTC__
#include <math.h>
#endif
#include <math_constants.h>

namespace neurocore {

struct DenseWarpResult {
    double x;      // This lane's solution coordinate; quiet NaN on failure.
    double error;  // Uniform relative residual; +infinity on failure.
    bool ok;       // Uniform over the N participating lanes.
};

// One independent dense system per hardware warp, with row i in lane i.
// Every lane 0..N-1 MUST call together, without missing/exited participants.
// Other lanes may call (they return failure without taking part) or stay out.
// The fixed mask is the low N bits; shuffle width remains 32 even for N=17.
// No arbitrary/subgroup masks are accepted. A partial final warp must still
// contain all N required lanes. Rows are lane-private, full, original FP64 A;
// rhs is the original b_i. Neither input is modified or used as scratch.
//
// residual_tolerance is explicit, finite, nonnegative and identical in all
// participants. Pass 1e-12 to retain the donor's acceptance criterion.
// Returns failure for nonfinite input/intermediates detected by the solve or
// residual, nonpositive pivots, nonfinite residual scaling, or residual excess.
// A mask/participation contract violation cannot be safely detected here.
//
// Compile the including translation unit with --fmad=false, --prec-div=true,
// and without --use_fast_math or reassociation. For N=17, valid finite solves
// retain the donor's forward/back substitution and residual accumulation order.
// A may include any dense mass matrix (e.g. A = (2/dt)*M + K); assembly and
// time integration are the caller's responsibility. No pivoting/fallback.
template <int N>
__device__ __forceinline__ DenseWarpResult dense_warp_solve(
    const double (&original_row)[N],
    const double rhs,
    const double residual_tolerance)
{
    static_assert(N >= 1 && N <= 32, "dense_warp_solve requires 1 <= N <= 32");
    constexpr unsigned mask = 0xffffffffu >> (32 - N);
    const unsigned linear_thread = threadIdx.x + blockDim.x *
        (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned lane = linear_thread & 31u;
    const DenseWarpResult failure = {CUDART_NAN, CUDART_INF, false};
    if (lane >= static_cast<unsigned>(N)) return failure;

    // Keep original_row and rhs intact for the final backward-error check.
    double A[N];
    double b = rhs;
    double x = 0.;
    const double tolerance = __shfl_sync(mask, residual_tolerance, 0);
    bool invalid = !isfinite(rhs) || !isfinite(residual_tolerance) ||
        residual_tolerance < 0. || residual_tolerance != tolerance;
    for (int j = 0; j < N; ++j) {
        A[j] = original_row[j];
        invalid |= !isfinite(A[j]);
    }
    if (__any_sync(mask, invalid)) return failure;

    // Same row elimination order as kc_fused_warp.py (donor17).
    for (int j = 0; j < N; ++j) {
        const double pivot = __shfl_sync(mask, A[j], j);
        const double bj = __shfl_sync(mask, b, j);
        // Both broadcasts are uniform: all participating lanes exit together.
        if (!(pivot > 0.) || !isfinite(pivot)) return failure;
        const double factor = lane > static_cast<unsigned>(j) ? A[j] / pivot : 0.;
        for (int k = j + 1; k < N; ++k) {
            const double ajk = __shfl_sync(mask, A[k], j);
            if (lane > static_cast<unsigned>(j)) A[k] -= factor * ajk;
        }
        if (lane > static_cast<unsigned>(j)) b -= factor * bj;
    }

    for (int j = N - 1; j >= 0; --j) {
        if (lane == static_cast<unsigned>(j)) x = b / A[j];
        const double xj = __shfl_sync(mask, x, j);
        if (lane < static_cast<unsigned>(j)) b -= A[j] * xj;
    }
    if (__any_sync(mask, !isfinite(x))) return failure;

    // Residual of ORIGINAL A*x-rhs, never of the eliminated matrix/RHS.
    double residual = -rhs;
    double row = 0.;
    for (int j = 0; j < N; ++j) {
        const double xj = __shfl_sync(mask, x, j);
        residual += original_row[j] * xj;
        row += fabs(original_row[j]);
    }
    // fmax alone would hide some NaNs. Reject before the max reduction.
    if (__any_sync(mask, !isfinite(residual) || !isfinite(row))) return failure;

    double rmax = fabs(residual);
    double amax = row;
    double bmax = fabs(rhs);
    double xmax = fabs(x);
    // Keep the donor's 16,8,4,2,1 tree. Every source is a participating lane;
    // lanes without a neighbor read themselves and do not apply that value.
    for (int offset = 16; offset > 0; offset /= 2) {
        const unsigned neighbor = lane + static_cast<unsigned>(offset);
        const bool has_neighbor = neighbor < static_cast<unsigned>(N);
        const int source = static_cast<int>(has_neighbor ? neighbor : lane);
        const double rr = __shfl_sync(mask, rmax, source);
        const double aa = __shfl_sync(mask, amax, source);
        const double bb = __shfl_sync(mask, bmax, source);
        const double xx = __shfl_sync(mask, xmax, source);
        if (has_neighbor) {
            rmax = fmax(rmax, rr);
            amax = fmax(amax, aa);
            bmax = fmax(bmax, bb);
            xmax = fmax(xmax, xx);
        }
    }

    double error = CUDART_INF;
    if (lane == 0) {
        const double scale = amax * xmax + bmax;
        // Overflow must not turn a finite residual into a false zero error.
        if (isfinite(scale)) error = rmax / fmax(scale, 1e-300);
    }
    error = __shfl_sync(mask, error, 0);
    if (!isfinite(error) || error > tolerance) return failure;
    return DenseWarpResult{x, error, true};
}

}  // namespace neurocore
