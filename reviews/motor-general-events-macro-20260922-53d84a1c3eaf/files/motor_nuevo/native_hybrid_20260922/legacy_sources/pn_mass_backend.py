"""Candidate CPU sparse electrical backend with explicit full capacitance action.

Physical membrane C and generalized M have separate APIs. No mass diagonalization,
conductance fitting, or implicit anatomical projection occurs in this backend.
"""
import hashlib
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import splu
from numba import njit
from compensated_csr import CompensatedCSR

class NumpyArrayAPI:
    def __getattr__(self,name):return getattr(np,name)
    def asnumpy(self,value):return np.array(value,copy=True)


def immutable(a):
    x=np.ascontiguousarray(a)
    return np.frombuffer(x.tobytes(),dtype=x.dtype).reshape(x.shape)


def sparse(a):
    m=csr_matrix(a,dtype=float,copy=True);m.sum_duplicates();m.eliminate_zeros();m.sort_indices()
    if m.shape[0]!=m.shape[1] or not np.isfinite(m.data).all():raise ValueError('Finite square electrical matrix required')
    d=m-m.T
    if d.nnz and np.max(abs(d.data))>1e-13*max(1.,np.max(abs(m.data))):raise ValueError('Electrical matrix must be symmetric')
    m.data=immutable(m.data);m.indices=immutable(m.indices);m.indptr=immutable(m.indptr)
    return m

@njit(cache=True)
def _difference_action(indptr,indices,data,sums,v):
    out=np.empty_like(v)
    for i in range(len(v)):
        total=sums[i]*v[i];correction=0.
        for k in range(indptr[i],indptr[i+1]):
            j=indices[k]
            if i==j:continue
            term=data[k]*(v[j]-v[i]);value=total+term
            if abs(total)>=abs(term):correction+=(total-value)+term
            else:correction+=(term-value)+total
            total=value
        out[i]=total+correction
    return out


class FullMassBackend:
    def __init__(self,G_nS,M_nF,physical_C_nF,*,provenance):
        self.G=sparse(G_nS);self.M=sparse(M_nF);c=np.asarray(physical_C_nF,dtype=float)
        if (self.M.shape!=self.G.shape or c.shape!=(self.G.shape[0],) or not np.isfinite(c).all()
            or np.any(c<0) or not isinstance(provenance,str) or not provenance):
            raise ValueError('Explicit physical capacitance, same-size mass and provenance required')
        self.C=immutable(c);self.provenance=provenance
        extra=self.M-diags(self.C);extra.eliminate_zeros();support=np.unique(extra.nonzero()[0])
        # H/Z adds mass only at its20ports and20internal coordinates. Refuse
        # to silently make a giant dense certification matrix for other cases.
        if len(support)>512:raise ValueError('Mass correction needs a different bounded PSD certificate')
        e=np.linalg.eigvalsh(extra[support][:,support].toarray()) if len(support) else np.zeros(1)
        tolerance=128*np.finfo(float).eps*max(float(abs(e).max()),1e-30)
        if e[0]<-tolerance:raise ValueError('Physical C exceeds full mass: invalid membrane lower bound')
        self.mass_certificate=dict(correction_support=len(support),minimum_eigenvalue_nF=float(e[0]),roundoff_tolerance_nF=tolerance)
        self.cp=NumpyArrayAPI();self.cpu=SimpleNamespace(levels=[SimpleNamespace(G=self.G,C=self.C)])
        self.levels=[dict(C=self.C)]
        self._buffers=(self.G.data,self.G.indices,self.G.indptr,self.M.data,self.M.indices,self.M.indptr,self.C)
        h=hashlib.sha256(('PN_full_mass_backend_v1'+provenance).encode())
        for a in self._buffers:h.update(str((a.dtype.str,a.shape)).encode());h.update(memoryview(a).cast('B'))
        self.identity=h.hexdigest();self._factor=None;self._factor_key=None
        self._G_sum=immutable(CompensatedCSR(self.G,minimum_terms=1)(np.ones(len(self.C))))
        self._mass_product=CompensatedCSR(self.M,minimum_terms=16)
    def assert_model(self):
        actual=(self.G.data,self.G.indices,self.G.indptr,self.M.data,self.M.indices,self.M.indptr,self.C)
        if (any(a is not b or a.flags.writeable for a,b in zip(actual,self._buffers))
            or self.cpu.levels[0].G is not self.G or self.cpu.levels[0].C is not self.C
            or self.levels[0]['C'] is not self.C):raise ValueError('Full mass or physical membrane model changed')
    def action(self,v):return _difference_action(self.G.indptr,self.G.indices,self.G.data,self._G_sum,np.asarray(v))
    def mass_action(self,v):return self._mass_product(v)
    def solve(self,rhs,*,shift,rtol,atol,maxiter,diagonal_update,**unused):
        self.assert_model();rhs=np.asarray(rhs,dtype=float);nodes,values=diagonal_update
        if not np.isfinite(shift) or shift<=0 or rhs.shape!=self.C.shape or not np.isfinite(rhs).all():raise ValueError('Finite shifted electrical system required')
        nodes=np.asarray(nodes);values=np.asarray(values,dtype=float)
        if nodes.dtype.kind not in 'iu' or nodes.shape!=values.shape or not np.isfinite(values).all() or np.any(nodes<0) or np.any(nodes>=len(rhs)) or len(np.unique(nodes))!=len(nodes):raise ValueError('Explicit unique diagonal correction required')
        diagonal=np.zeros_like(rhs);diagonal[nodes]=values
        A=self.G+shift*self.M+diags(diagonal)
        key=(float(shift),nodes.tobytes(),values.tobytes())
        if key!=self._factor_key:
            factor=splu(A.tocsc());self._factor=factor;self._factor_key=key
        x=self._factor.solve(rhs);limit=max(float(atol),float(rtol)*float(np.linalg.norm(rhs)))
        norm=float(np.linalg.norm(A@x-rhs));iterations=1
        while norm>limit and iterations<min(int(maxiter),4):
            x+=self._factor.solve(rhs-A@x);norm=float(np.linalg.norm(A@x-rhs));iterations+=1
        passed=bool(np.isfinite(x).all() and np.isfinite(norm) and norm<=limit)
        return x,dict(info=0 if passed else 1,iterations=iterations,fine_residual_passed=passed,
            residual_l2_pA=norm,threshold_pA=limit,linear_solver='CPU sparse LU of signed full Jacobian',
            residual_scope='Reduced/generalized equation, not residual in omitted fine volume')
