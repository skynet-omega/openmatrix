"""External PN experiment: solve the unchanged fine RC system with multigrid.

Only volume unknowns are aggregated. Cable/collar unknowns survive on every
level. Coarse matrices are Galerkin operators used as a PRECONDITIONER, not
substitute anatomy or fitted membrane. Final residuals use the fine matrix.
All electrical units remain nS, nF, mV, pA; shift is in reciprocal seconds.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg
from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.relaxation import jacobi_indexed, gauss_seidel_indexed


def _smoother(indices, iterations, relaxation='jacobi', sweep='forward'):
    def smooth(A, x, b):
        if relaxation == 'jacobi':
            jacobi_indexed(A, x, b, indices, iterations=iterations, omega=2/3)
        else:
            gauss_seidel_indexed(A, x, b, indices, iterations=iterations, sweep=sweep)
    return smooth


def bounded_gmres(A, b, M, *, initial=None, rtol=1e-8, atol=1e-11,
                  restart=12, maxiter=500, callback=None):
    """Left-preconditioned Arnoldi GMRES, with bounded vector accumulation.

    Large complex GEMV in SciPy's restart update crashed the local BLAS at
    18.6 million unknowns. Accumulate vectors separately here. Only the small
    Hessenberg least-squares problem uses dense linear algebra. Two passes
    of modified Gram-Schmidt control loss of orthogonality. The acceptance
    test always uses the original, unpreconditioned residual. maxiter counts
    inner iterations, including across restarts.
    """
    if (not np.isscalar(rtol) or np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol<=0
            or not np.isscalar(atol) or np.iscomplexobj(atol) or not np.isfinite(atol) or atol<0
            or type(maxiter) is not int or maxiter<0 or type(restart) is not int or restart<1):
        raise ValueError('Finite tolerances and bounded iteration budget required')
    x = np.zeros_like(b) if initial is None else np.array(initial, dtype=b.dtype, copy=True)
    bnorm=float(np.linalg.norm(b));threshold = max(atol, rtol*bnorm); done = 0
    if not np.isfinite(bnorm) or not np.isfinite(threshold):raise ValueError('Finite current norm and threshold required')
    while True:
        residual = b-A@x
        if np.linalg.norm(residual) <= threshold:
            return x, 0
        if done >= maxiter:
            return x, done
        r = M@residual; beta = np.linalg.norm(r)
        if not np.isfinite(beta) or beta == 0:
            return x, -1
        count = min(restart, maxiter-done)
        V = np.empty((count+1, len(b)), dtype=b.dtype); V[0] = r/beta
        H = np.zeros((count+1, count), dtype=b.dtype)
        target = np.zeros(count+1, dtype=b.dtype); target[0] = beta
        for k in range(count):
            w = M@(A@V[k]); original_norm = np.linalg.norm(w)
            for _ in range(2):
                for j in range(k+1):
                    projection = np.vdot(V[j], w)
                    H[j, k] += projection; w -= projection*V[j]
            H[k+1, k] = np.linalg.norm(w)
            breakdown = abs(H[k+1, k]) <= np.finfo(float).eps*original_norm
            if not breakdown:
                V[k+1] = w/H[k+1, k]
            y = np.linalg.lstsq(H[:k+2, :k+1], target[:k+2], rcond=None)[0]
            done += 1
            if callback:
                callback(float(np.linalg.norm(target[:k+2]-H[:k+2, :k+1]@y)/max(np.linalg.norm(b), 1e-300)))
            if breakdown:
                break
        for j in range(k+1):
            x += y[j]*V[j]
        del V, w, residual, r
        # Happy breakdown is accepted only through the fine residual above.
        if breakdown and np.linalg.norm(b-A@x) > threshold:
            return x, -2


class JointPNMultigrid:
    """Geometric aggregation of volume, exact retained exterior on coarse grid.

    Aggregation never mixes separate fine-volume components. In a component,
    a box can aggregate disconnected local pieces ONLY for preconditioning;
    the fine operator still has every original face and no extra connection.
    Symmetric volume-only Jacobi avoids treating generalized port couplings
    as an M-matrix. Their complete exterior block is solved on the last level.
    """
    def __init__(self, G_nS, C_nF, volume_xyz, volume_components, *,
                 coarse_volume_limit=800, smoothing_steps=2, relaxation='jacobi', progress=None):
        # Own the coefficients: a caller changing its assembly buffers must
        # not silently invalidate coarse operators or a saved state identity.
        G = csr_matrix(G_nS, dtype=float, copy=True); C = np.array(C_nF, dtype=float, copy=True)
        xyz = np.asarray(volume_xyz); component = np.asarray(volume_components)
        if (G.shape != (len(C), len(C)) or xyz.ndim != 2 or xyz.shape[1] != 3
                or xyz.dtype.kind not in 'iu' or np.any(xyz < 0)
                or component.shape != (len(xyz),) or component.dtype.kind not in 'iu'
                or np.any(component < 0) or not len(xyz) or len(xyz) >= len(C)
                or not np.isfinite(G.data).all() or not np.isfinite(C).all() or np.any(C < 0)
                or type(coarse_volume_limit) is not int or coarse_volume_limit < 1
                or type(smoothing_steps) is not int or smoothing_steps < 1
                or relaxation not in ['jacobi', 'symmetric_gauss_seidel']):
            raise ValueError('Complete fine RC system, volume coordinates/components required')
        G.sort_indices(); asym = G-G.T
        if asym.nnz and np.max(abs(asym.data)) > 1e-8:
            raise ValueError('Reciprocal fine operator required')
        self.levels = []; self.smoothing_steps = smoothing_steps; self.shift = None
        nc = int(component.max())+1; exterior = len(C)-len(xyz)
        while True:
            level = MultilevelSolver.Level(); level.G = G; level.C = C
            level.volume_nodes = len(xyz); level.A = G
            level.volume_parity=((xyz[:,0]+xyz[:,1]+xyz[:,2])%2).astype(np.uint8)
            self.levels.append(level)
            if progress: progress(dict(level=len(self.levels)-1, volume_nodes=len(xyz),
                                       exterior_nodes=exterior, nonzeros=G.nnz))
            if len(xyz) <= coarse_volume_limit:
                break
            coarse_xyz = xyz//2; shape = coarse_xyz.max(0)+1
            key = np.ravel_multi_index(coarse_xyz.T, shape)*nc+component
            unique, aggregate = np.unique(key, return_inverse=True)
            if len(unique) >= len(xyz):
                raise ValueError('Geometric aggregation cannot reduce the volume further')
            aggregate = aggregate.astype(np.int32)
            ncoarse = len(unique); N = G.shape[0]
            columns = np.r_[aggregate, ncoarse+np.arange(exterior, dtype=np.int32)]
            P = csr_matrix((np.ones(N), columns, np.arange(N+1, dtype=np.int32)),
                           shape=(N, ncoarse+exterior))
            level.P = P; level.R = P.T.tocsr()
            indices = np.arange(len(xyz), dtype=np.int32)
            level.presmoother = _smoother(indices, smoothing_steps, relaxation, 'forward')
            level.postsmoother = _smoother(indices, smoothing_steps, relaxation, 'backward')
            G = (level.R@(G@P)).tocsr(); G.sort_indices()
            C = np.r_[np.bincount(aggregate, weights=C[:len(xyz)], minlength=ncoarse), C[len(xyz):]]
            xyz = np.column_stack(np.unravel_index(unique//nc, shape)).astype(np.int32)
            component = (unique % nc).astype(np.int32)
        for level in self.levels:
            for a in [level.G.data,level.G.indices,level.G.indptr,level.C]:a.flags.writeable=False
        self.set_shift(0.)

    def set_shift(self, shift):
        """Real positive shift for the preconditioner; fine complex solve stays exact."""
        if not np.isfinite(shift) or shift < 0:
            raise ValueError('Finite nonnegative real preconditioning shift required')
        if self.shift == shift:
            return
        for level in self.levels:
            if shift == 0:
                level.A = level.G
            else:
                level.A = level.G.copy()
                level.A.setdiag(level.G.diagonal()+shift*level.C)
                level.A.sort_indices()
        self.ml = MultilevelSolver(self.levels, coarse_solver='splu')
        self.preconditioner = self.ml.aspreconditioner(cycle='V')
        self.shift = shift

    def update_conductance(self, delta_G):
        """Change a surface map without changing geometry, C, or prolongation."""
        delta = csr_matrix(delta_G, dtype=float)
        if delta.shape != self.levels[0].G.shape or not np.isfinite(delta.data).all():
            raise ValueError('Fine operator update has wrong shape or nonfinite terms')
        asym = delta-delta.T
        if asym.nnz and np.max(abs(asym.data)) > 1e-8:
            raise ValueError('Nonreciprocal interface update')
        for k, level in enumerate(self.levels):
            level.G = (level.G+delta).tocsr(); level.G.sort_indices()
            for a in [level.G.data,level.G.indices,level.G.indptr]:a.flags.writeable=False
            if k+1 < len(self.levels):
                delta = (level.R@(delta@level.P)).tocsr()
        shift = self.shift; self.shift = None; self.set_shift(shift)

    def solve(self, current_pA, *, shift=0., rtol=1e-8, atol=1e-11,
              maxiter=500, initial=None, progress=None, compensated_rows=False):
        """DC or Laplace/frequency solve; report the independently recomputed residual.

        This is not a time integrator or CNS checkpoint. Complex shifts use
        GMRES and the real positive-shift V cycle on real/imaginary parts.
        """
        if (not np.isscalar(shift) or not np.isfinite(shift) or np.real(shift) < 0
                or not np.isscalar(rtol) or np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol <= 0
                or not np.isscalar(atol) or np.iscomplexobj(atol) or not np.isfinite(atol) or atol < 0
                or type(maxiter) is not int or maxiter < 1):
            raise ValueError('Stable finite shift and positive solver tolerances required')
        self.set_shift(float(abs(shift)))
        level = self.levels[0]; G = level.G; C = level.C
        accurate = None
        if compensated_rows:
            from compensated_csr import CompensatedCSR
            accurate = CompensatedCSR(G)
        dtype = np.complex128 if np.iscomplexobj(shift) or np.iscomplexobj(current_pA) else np.float64
        b = np.asarray(current_pA, dtype=dtype)
        if b.shape != C.shape or not np.isfinite(b).all():
            raise ValueError('Finite current required on every joint unknown')
        bnorm=float(np.linalg.norm(b));threshold=max(atol,rtol*bnorm)
        if not np.isfinite(bnorm) or not np.isfinite(threshold):raise ValueError('Finite current norm and threshold required')
        def action(v):
            # scipy's mixed real-CSR/complex-vector path can cast every CSR
            # coefficient to complex on every product. Avoid that multi-GB
            # temporary; keep the SAME real fine matrix and two real products.
            if accurate is not None:
                value = accurate(v)
            elif np.iscomplexobj(v):
                value = (G@v.real).astype(np.complex128)
                value.imag = G@v.imag
            else:
                value = G@v
            return value+shift*C*v
        A = LinearOperator(G.shape, matvec=action, dtype=dtype)
        start = time.monotonic(); iterations = 0
        def callback(value):
            nonlocal iterations
            iterations += 1
            if progress and (iterations == 1 or iterations % 20 == 0):
                progress(dict(iteration=iterations, seconds=time.monotonic()-start,
                              preconditioned_residual=float(value) if np.isscalar(value) else None,
                              fine_relative_residual=(None if np.isscalar(value) else
                                  float(np.linalg.norm(b-A@value)/max(bnorm,1e-300)))))
        restarts = 0
        if dtype == np.float64:
            x = initial
            threshold = max(atol, rtol*np.linalg.norm(b))
            while True:
                remaining = maxiter-iterations
                x, info = cg(A, b, x0=x, rtol=rtol*(.5**restarts),
                             atol=atol*(.5**restarts), maxiter=remaining,
                             M=self.preconditioner, callback=callback)
                # Recursive CG residuals can drift. Restart from the true
                # residual with a stricter internal target; never relax the
                # requested acceptance target or the total iteration budget.
                if (np.linalg.norm(b-A@x) <= threshold or info != 0
                        or iterations >= maxiter or restarts >= 3):
                    break
                restarts += 1
        else:
            real_M = self.preconditioner
            def complex_precondition(v):
                # Explicit contiguous buffers also isolate the compiled real
                # relaxation kernels from strided views of complex storage.
                real = real_M@np.ascontiguousarray(v.real)
                imag = real_M@np.ascontiguousarray(v.imag)
                return real+1j*imag
            M = LinearOperator(G.shape, matvec=complex_precondition, dtype=dtype)
            x, info = bounded_gmres(A, b, M, initial=initial, rtol=rtol, atol=atol,
                                   restart=12, maxiter=maxiter, callback=callback)
        residual = b-A@x; norm = float(np.linalg.norm(residual)); bnorm = float(np.linalg.norm(b))
        threshold = max(atol, rtol*bnorm)
        report = dict(info=int(info), iterations=iterations, seconds=time.monotonic()-start,
                      residual_l2_pA=norm, current_l2_pA=bnorm, threshold_pA=threshold,
                      relative_residual=norm/max(bnorm, 1e-300),
                      fine_residual_passed=bool(norm <= threshold),
                      compensated_rows=0 if accurate is None else len(accurate.rows),
                      residual_restarts=restarts,
                      shift_real=float(np.real(shift)), shift_imag=float(np.imag(shift)))
        return x, report
