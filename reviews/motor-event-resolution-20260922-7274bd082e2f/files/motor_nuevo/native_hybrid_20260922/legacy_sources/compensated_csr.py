"""Accumulate high-degree CSR rows without changing stored coefficients.

Long port rows can sum hundreds of thousands of cancelling currents. Recompute
only these rows with Neumaier summation; ordinary short rows keep SciPy's CSR
product. This addresses summation error, not assembly error or conditioning.
"""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


@njit(cache=True)
def _rows(indptr, indices, data, rows, vector, result):
    for row in rows:
        total = 0.; correction = 0.
        for k in range(indptr[row], indptr[row+1]):
            term = data[k]*vector[indices[k]]
            value = total+term
            if abs(total) >= abs(term):
                correction += (total-value)+term
            else:
                correction += (term-value)+total
            total = value
        result[row] = total+correction


class CompensatedCSR:
    """Real CSR matvec; complex vectors use two independent real products."""
    def __init__(self, matrix, *, minimum_terms=32):
        if type(minimum_terms) is not int or minimum_terms < 1:
            raise ValueError('Positive integer row threshold required')
        self.matrix = csr_matrix(matrix, dtype=float, copy=False)
        self.rows = np.flatnonzero(np.diff(self.matrix.indptr) >= minimum_terms)

    def _real(self, vector):
        out = self.matrix@vector
        _rows(self.matrix.indptr, self.matrix.indices, self.matrix.data,
              self.rows, vector, out)
        return out

    def __call__(self, vector):
        v = np.asarray(vector)
        if v.shape != (self.matrix.shape[1],):
            raise ValueError('Complete one-dimensional vector required')
        if np.iscomplexobj(v):
            out = self._real(v.real).astype(complex)
            out.imag = self._real(v.imag)
            return out
        return self._real(v)
