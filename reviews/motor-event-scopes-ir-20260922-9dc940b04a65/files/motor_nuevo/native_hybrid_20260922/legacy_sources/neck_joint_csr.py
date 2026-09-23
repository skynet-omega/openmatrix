"""Assemble the unchanged volume/collar/exterior operator without bulk copies.

Two passes over physical volume edges allocate the fine CSR once. Only the
boundary cross block and retained exterior are assembled separately. This
changes storage, not anatomical admission, face voltages, G, or membrane C.
"""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, coo_matrix, block_diag


@njit(cache=True)
def _count_edges(edges, counts):
    for k in range(len(edges)):
        i,j=edges[k]; counts[i]+=1; counts[j]+=1


@njit(cache=True)
def _fill_volume(edges, g, leak, outer, face_g, indptr, cursor, columns, data):
    for i in range(len(leak)):
        j=cursor[i]; columns[j]=i; data[j]=leak[i]; cursor[i]+=1
    for k in range(len(edges)):
        i,j=edges[k]
        data[indptr[i]]+=g; data[indptr[j]]+=g
        a=cursor[i]; columns[a]=j; data[a]=-g; cursor[i]+=1
        a=cursor[j]; columns[a]=i; data[a]=-g; cursor[j]+=1
    for f in range(len(outer)):
        data[indptr[outer[f]]]+=face_g[f]


@njit(cache=True)
def _fill_cross(boundary, cross, ports, offset, cursor, columns, data):
    for k in range(len(boundary)):
        i=boundary[k]
        for p in range(len(ports)):
            value=cross[k,p]
            if value==0: continue
            j=offset+ports[p]
            a=cursor[i]; columns[a]=j; data[a]=value; cursor[i]+=1
            a=cursor[j]; columns[a]=i; data[a]=value; cursor[j]+=1


@njit(cache=True)
def _fill_exterior(ep, ei, ev, offset, cursor, columns, data):
    for row in range(len(ep)-1):
        out=offset+row
        for k in range(ep[row],ep[row+1]):
            a=cursor[out]; columns[a]=offset+ei[k]; data[a]=ev[k]; cursor[out]+=1


def join_volume_exterior_csr(patch, outside_G, outside_C, outside_ports):
    """Equivalent to join_volume_exterior, with no patch.matrix() or bmat.

    Caller must supply an ALREADY withdrawn exterior. Interface conductance
    is S.T D S, -S.T D P, and P.T D P; no additional Dirichlet clamp. Node maps
    and capacitance use exactly the same alias convention as the reference.
    """
    Gext=csr_matrix(outside_G); Cext=np.asarray(outside_C,dtype=float)
    op=np.asarray(outside_ports); cp=np.asarray(patch.collar['source_port_indices'])
    if (Gext.shape!=(len(Cext),len(Cext)) or op.shape!=cp.shape
            or op.dtype.kind not in 'iu' or len(np.unique(op))!=len(op)
            or np.any(op<0) or np.any(op>=len(Cext))
            or cp.dtype.kind not in 'iu' or np.any(cp<0) or np.any(cp>=patch.nc)
            or len(np.unique(cp))!=len(cp)
            or not np.isfinite(Cext).all() or np.any(Cext<0)
            or not np.isfinite(Gext.data).all()):
        raise ValueError('Withdrawn exterior and unique matching collar aliases required')
    nv=patch.n; extra=np.setdiff1d(np.arange(patch.nc),cp)
    cm=np.empty(patch.nc,np.int64);cm[cp]=op;cm[extra]=len(Cext)+np.arange(len(extra))
    ne=len(Cext)+len(extra); N=nv+ne
    if N>=np.iinfo(np.int32).max:raise ValueError('Joint node count exceeds explicit int32 bound')
    co=csr_matrix(patch.collar['G_nS']).tocoo()
    if co.shape!=(patch.nc,patch.nc) or not np.isfinite(co.data).all():
        raise ValueError('Collar operator mismatch')
    exterior=block_diag([Gext,csr_matrix((len(extra),len(extra)))],format='csr')
    exterior+=coo_matrix((co.data,(cm[co.row],cm[co.col])),shape=(ne,ne)).tocsr()
    surface=cm[patch.cp]; P=patch.interface.P; fg=patch.interface.g
    if len(np.unique(surface))!=len(surface):raise ValueError('Surface aliases are repeated')
    gram=P.T@(fg[:,None]*P);i,j=np.indices(gram.shape)
    exterior+=coo_matrix((gram.ravel(),(surface[i.ravel()],surface[j.ravel()])),shape=(ne,ne)).tocsr()
    exterior.sum_duplicates();exterior.eliminate_zeros();exterior.sort_indices()
    boundary,face_row=np.unique(patch.interface.nodes,return_inverse=True)
    cross=np.zeros((len(boundary),P.shape[1]))
    np.add.at(cross,face_row,-fg[:,None]*P)
    counts=np.ones(N,np.int32);counts[nv:]=np.diff(exterior.indptr)
    _count_edges(patch.t['edges'],counts)
    nonzero=cross!=0
    counts[boundary]+=nonzero.sum(1,dtype=np.int32)
    counts[nv+surface]+=nonzero.sum(0,dtype=np.int32)
    pointer64=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    if pointer64[-1]>=np.iinfo(np.int32).max:
        raise ValueError('Joint CSR nnz exceeds explicit int32 bound')
    indptr=pointer64.astype(np.int32);del pointer64,counts,nonzero,face_row
    columns=np.empty(indptr[-1],np.int32);data=np.empty(indptr[-1],float)
    cursor=indptr[:-1].copy()
    _fill_volume(patch.t['edges'],float(patch.terms['g_nS']),patch.terms['leak_nS'],
                 patch.interface.nodes,fg,indptr,cursor,columns,data)
    _fill_cross(boundary,cross,surface,nv,cursor,columns,data)
    _fill_exterior(exterior.indptr,exterior.indices,exterior.data,nv,cursor,columns,data)
    assert np.array_equal(cursor,indptr[1:]),'Incomplete CSR allocation'
    del cursor,cross,exterior
    G=csr_matrix((data,columns,indptr),shape=(N,N),copy=False)
    assert np.shares_memory(G.data,data) and np.shares_memory(G.indices,columns)
    G.sort_indices()
    C=np.r_[patch.terms['C_nF'],Cext,np.zeros(len(extra))]
    np.add.at(C,nv+cm,patch.collar['C_nF'])
    np.testing.assert_allclose(C.sum(),patch.C_nF.sum()+Cext.sum(),rtol=1e-13)
    return dict(G_nS=G,C_nF=C,volume_nodes=nv,exterior_offset=nv,
                collar_to_joint=nv+cm,outside_to_joint=nv+np.arange(len(Cext)))
