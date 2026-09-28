"""Exact passive block handoff with internal membrane history retained.

An exterior cable and a local block share explicitly declared electrical ports.
Schur elimination is performed at EACH implicit-Euler step, including the
internal capacitive RHS. It is not a static conductance or fitted filter.
Anatomical port admission is separate; no automatic CNS promotion is provided.
"""
import hashlib
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix,diags
from scipy.sparse.linalg import splu


def _fingerprint(arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.asarray(value);h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def withdraw_source_block(parent_G,parent_C,block):
    """Remove whole original edges with their original half-edge membrane.

    The caller obtains the block with neck_multiport.extract_block. Selecting
    whole edges adds an explicitly accounted cable collar outside a voxel box;
    subtracting a fractional geometric area at old lumped nodes is avoided.
    Nothing is written back to the parent. A second subtraction fails the
    residual accounting rather than clipping negative capacitance to zero.
    """
    G=csr_matrix(parent_G);C=np.asarray(parent_C,dtype=float);nodes=np.asarray(block['nodes'])
    if G.shape!=(len(C),len(C)) or (nodes<0).any() or (nodes>=len(C)).any():
        raise ValueError('Source dimensions do not match block')
    part=block['G_nS'].tocoo();out=G-coo_matrix((part.data,(nodes[part.row],nodes[part.col])),shape=G.shape).tocsr()
    outC=C.copy();outC[nodes]-=block['C_nF'];inside=block['internal_nodes']
    scale=np.asarray(abs(G).sum(1)).ravel();tol=64*np.finfo(float).eps*np.maximum(scale,1e-30)
    residual=np.asarray(abs(out[inside]).sum(1)).ravel()
    ctol=64*np.finfo(float).eps*np.maximum(C,1e-30)
    if (np.any(residual>tol[inside]) or np.any(abs(outC[inside])>ctol[inside])
            or np.any(outC < -ctol)):
        raise ValueError('Source block already withdrawn or membrane/axial accounting mismatch')
    keep=np.setdiff1d(np.arange(len(C)),inside);mapping=np.full(len(C),-1,np.int64);mapping[keep]=np.arange(len(keep))
    # Only roundoff within the declared local tolerance may be zeroed.
    outC[abs(outC)<=ctol]=0.
    return dict(G_nS=out[keep][:,keep].tocsr(),C_nF=outC[keep],kept_original_nodes=keep,
                original_to_outside=mapping,outside_ports=mapping[block['port_nodes']],
                retired_C_nF=float(np.sum(block['C_nF'])),retired_edges=block['removed_edges'].copy())


class DynamicPortHandoff:
    """Sparse exterior, exact local dynamic elimination, explicit full state."""
    def __init__(self,outside_G,outside_C,patch_G,patch_C,outside_ports,patch_ports,
                 *,patch_congruence=None):
        self.G=csr_matrix(outside_G,dtype=float,copy=True);self.G.sort_indices()
        self.C=np.array(outside_C,dtype=float,copy=True)
        self.P=csr_matrix(patch_G,dtype=float,copy=True);self.P.sort_indices()
        self.Q=np.array(patch_C,dtype=float,copy=True)
        op=np.asarray(outside_ports);pp=np.asarray(patch_ports)
        if (self.G.shape!=(len(self.C),len(self.C)) or self.P.shape!=(len(self.Q),len(self.Q))
                or op.ndim!=1 or pp.shape!=op.shape or not len(op) or op.dtype.kind not in 'iu' or pp.dtype.kind not in 'iu'
                or np.any(op<0) or np.any(op>=len(self.C)) or np.any(pp<0) or np.any(pp>=len(self.Q))
                or len(np.unique(op))!=len(op) or len(np.unique(pp))!=len(pp)
                or any(not np.isfinite(a).all() for a in [self.G.data,self.P.data,self.C,self.Q])
                or np.any(self.C<0) or np.any(self.Q<0)):
            raise ValueError('Invalid passive block/port dimensions')
        for matrix in [self.G,self.P]:
            diff=matrix-matrix.T
            if diff.nnz and np.max(abs(diff.data))>1e-11:raise ValueError('Nonreciprocal passive block')
            off=matrix-diags(matrix.diagonal());off.eliminate_zeros()
            if off.nnz and np.max(off.data)>1e-11:raise ValueError('Nonpassive axial term')
            scale=np.asarray(abs(matrix).sum(1)).ravel()
            tolerance=64*np.finfo(float).eps*(np.diff(matrix.indptr)+1)*np.maximum(scale,1e-30)
            if np.any(np.asarray(matrix.sum(1)).ravel() < -tolerance):
                raise ValueError('Negative membrane leak')
        if patch_congruence is not None:
            # A surface interpolation is an energy-conjugate constraint,
            # not necessarily an M-matrix. Keep strict resistor validation
            # above for both base blocks; add ONLY a certified PSD term.
            # T*1=0 preserves common-mode voltage and creates no membrane.
            T,weight=patch_congruence;T=csr_matrix(T,dtype=float,copy=True)
            weight=np.asarray(weight,dtype=float)
            if (T.shape[1]!=len(self.Q) or T.shape[0]<1 or weight.shape!=(T.shape[0],)
                    or not np.isfinite(T.data).all() or not np.isfinite(weight).all()
                    or np.any(weight<=0) or np.max(abs(np.asarray(T.sum(1)).ravel()))>1e-8):
                raise ValueError('Invalid conservative passive congruence certificate')
            self.P=(self.P+T.T@diags(weight)@T).tocsr();self.P.sort_indices()
        self.op=op.astype(np.int64,copy=True);self.pp=pp.astype(np.int64,copy=True)
        self.inner=np.setdiff1d(np.arange(len(self.Q)),self.pp)
        self.identity=_fingerprint([self.G.data,self.G.indices,self.G.indptr,self.C,
            self.P.data,self.P.indices,self.P.indptr,self.Q,self.op,self.pp])
        self.outside=np.zeros(len(self.C));self.internal=np.zeros(len(self.inner));self.time_ns=0;self._cache=None
        for array in [self.C,self.Q,self.op,self.pp,self.inner,self.G.data,self.G.indices,self.G.indptr,self.P.data,self.P.indices,self.P.indptr]:
            array.flags.writeable=False

    def patch_voltage(self):
        v=np.empty(len(self.Q));v[self.pp]=self.outside[self.op];v[self.inner]=self.internal;return v

    def advance(self,dt_ns,outside_current,patch_current):
        ext=np.asarray(outside_current,dtype=float);local=np.asarray(patch_current,dtype=float)
        if (type(dt_ns) is not int or dt_ns<=0 or ext.shape!=self.C.shape or local.shape!=self.Q.shape
                or not np.isfinite(ext).all() or not np.isfinite(local).all()):raise ValueError('Invalid explicit step/current')
        dt=dt_ns*1e-9
        if self._cache is None or self._cache[0]!=dt_ns:
            A=self.P+diags(self.Q/dt);pp=self.pp;ii=self.inner
            Aip=A[ii][:,pp].toarray();Api=A[pp][:,ii].tocsr()
            factor=splu(A[ii][:,ii].tocsc()) if len(ii) else None
            transfer=factor.solve(Aip) if factor is not None else np.empty((0,len(pp)))
            K=A[pp][:,pp].toarray()-Api@transfer
            # Reciprocal elimination may differ by roundoff; do not fit or
            # discard its common-mode capacitance/leak response.
            K=(K+K.T)/2
            row,col=np.indices(K.shape);addition=coo_matrix((K.ravel(),(self.op[row.ravel()],self.op[col.ravel()])),shape=self.G.shape)
            total=self.G+diags(self.C/dt)+addition
            external_factor=splu(total.tocsc())
            self._cache=(dt_ns,factor,Aip,Api,transfer,external_factor)
        _,factor,Aip,Api,transfer,external_factor=self._cache
        local_rhs=local+self.Q*self.patch_voltage()/dt
        response=factor.solve(local_rhs[self.inner]) if factor is not None else np.empty(0)
        rhs=ext+self.C*self.outside/dt
        rhs[self.op]+=local_rhs[self.pp]-Api@response
        outside=external_factor.solve(rhs);inside=response-transfer@outside[self.op]
        if not np.isfinite(outside).all() or not np.isfinite(inside).all():raise ValueError('Nonfinite handoff state')
        self.outside=outside;self.internal=inside;self.time_ns+=dt_ns
        return outside.copy(),self.patch_voltage()

    def state_dict(self):
        return dict(schema='neck_dynamic_handoff_v1',identity=self.identity,time_ns=self.time_ns,
                    outside_delta_mV=self.outside.copy(),internal_delta_mV=self.internal.copy())

    def load_state_dict(self,state):
        if (set(state)!={'schema','identity','time_ns','outside_delta_mV','internal_delta_mV'}
                or state['schema']!='neck_dynamic_handoff_v1' or state['identity']!=self.identity
                or type(state['time_ns']) is not int or state['time_ns']<0
                or np.asarray(state['outside_delta_mV']).shape!=self.outside.shape
                or np.asarray(state['internal_delta_mV']).shape!=self.internal.shape
                or not np.isfinite(state['outside_delta_mV']).all() or not np.isfinite(state['internal_delta_mV']).all()):
            raise ValueError('Different or incomplete handoff state')
        self.outside=np.array(state['outside_delta_mV'],copy=True);self.internal=np.array(state['internal_delta_mV'],copy=True)
        self.time_ns=state['time_ns'];self._cache=None
