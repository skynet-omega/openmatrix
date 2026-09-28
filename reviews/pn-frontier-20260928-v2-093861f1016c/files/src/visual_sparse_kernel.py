"""Deterministic float64 row-parallel CNS summation, without fast-math.

Each row is accumulated in the canonical CSR order by one thread. Parallel
execution changes which row runs first, never its summation order. Positive
and negative visual conductances are accumulated directly, avoiding two full
matrix passes and cancellation in absolute/signed decomposition.
"""
import math
import numba
import numpy as np


@numba.njit(parallel=True, fastmath=False, cache=True)
def coefficients(indptr, indices, weights, release, caps, visual, tau, gain_over_cap,
                 theta, drive, photo_by_row, scale, connected):
    n = len(release)
    target, rate = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
    for row in numba.prange(n):
        if visual[row]:
            exc, inh = photo_by_row[row], 0.
            for edge in range(indptr[row], indptr[row+1]):
                value = weights[edge]*scale*release[indices[edge]]
                if value >= 0.:
                    exc += value
                else:
                    inh -= value
            total = 1.+exc+inh
            target[row] = (.25+exc)/total
            rate[row] = total/tau[row]
        else:
            current = 0.
            for edge in range(indptr[row], indptr[row+1]):
                col = indices[edge]
                if connected or not visual[col]:
                    current += weights[edge]*(release[col]*caps[col])
            target[row] = max(0., math.tanh(gain_over_cap[row]*(current+drive[row]-theta[row])))
            rate[row] = 1./tau[row]
    return target, rate
