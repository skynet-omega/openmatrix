"""Same full M/G Jacobian, with a reusable passive sparse factor.

An optional numerical backend, not admitted to a CNS branch by its existence.
No prior iterate is reused: the preconditioner depends only on immutable M/G
and the requested shift. The true residual and caller tolerance decide success.
"""
from collections import OrderedDict
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu,gmres,LinearOperator
from pn_mass_backend import FullMassBackend

POLICY='PN_full_mass_passive_LU_GMRES12_true_residual_direct_fallback_v1'

class FullMassKrylovBackend(FullMassBackend):
    @classmethod
    def adopt(cls,parent):
        if type(parent) is not FullMassBackend:raise ValueError('Exact full-mass parent required')
        parent.assert_model()
        obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._passive_factors=OrderedDict();obj._factor=None;obj._factor_key=None
        obj.solver_policy=POLICY
        return obj

    def solve(self,rhs,*,shift,rtol,atol,maxiter,diagonal_update,**unused):
        self.assert_model()
        if self.solver_policy!=POLICY:raise ValueError('Numerical solver policy changed')
        rhs=np.asarray(rhs,dtype=float);nodes,values=diagonal_update;nodes=np.asarray(nodes);values=np.asarray(values,dtype=float)
        if (not np.isscalar(shift) or not np.isfinite(shift) or shift<=0 or rhs.shape!=self.C.shape or not np.isfinite(rhs).all()
            or not np.isscalar(rtol) or not np.isscalar(atol) or not np.isfinite([rtol,atol]).all() or rtol<0 or atol<0
            or rtol+atol<=0 or type(maxiter) is not int or maxiter<1):raise ValueError('Finite positive-tolerance electrical system required')
        if (nodes.dtype.kind not in 'iu' or nodes.shape!=values.shape or nodes.ndim!=1 or not np.isfinite(values).all()
            or np.any(nodes<0) or np.any(nodes>=len(rhs)) or len(np.unique(nodes))!=len(nodes)):
            raise ValueError('Explicit unique signed diagonal correction required')
        diagonal=np.zeros_like(rhs);diagonal[nodes]=values
        A=(self.G+shift*self.M+diags(diagonal)).tocsr()
        key=float(shift);cold=key not in self._passive_factors
        if cold:
            self._passive_factors[key]=splu((self.G+shift*self.M).tocsc())
            if len(self._passive_factors)>4:self._passive_factors.popitem(last=False)
        factor=self._passive_factors[key]
        P=LinearOperator(A.shape,matvec=factor.solve,dtype=float);count=[0]
        def iteration(_):count[0]+=1
        x,info=gmres(A,rhs,M=P,rtol=rtol,atol=atol,restart=12,maxiter=maxiter,callback=iteration,callback_type='legacy')
        limit=max(float(atol),float(rtol)*float(np.linalg.norm(rhs)));norm=float(np.linalg.norm(A@x-rhs))
        passed=bool(np.isfinite(x).all() and np.isfinite(norm) and norm<=limit)
        if not passed:
            x,report=super().solve(rhs,shift=shift,rtol=rtol,atol=atol,maxiter=maxiter,diagonal_update=(nodes,values))
            report.update(solver_policy=POLICY,direct_fallback=True,GMRES_iterations=count[0],GMRES_info=int(info),
                GMRES_true_residual_l2_pA=norm,passive_factor_cold=cold)
            return x,report
        return x,dict(info=0,iterations=count[0],fine_residual_passed=True,residual_l2_pA=norm,threshold_pA=limit,
            linear_solver='GMRES with passive full-mass LU preconditioner',solver_policy=POLICY,direct_fallback=False,
            GMRES_info=int(info),passive_factor_cold=cold,residual_scope='Reduced/generalized equation, not residual in omitted fine volume')
