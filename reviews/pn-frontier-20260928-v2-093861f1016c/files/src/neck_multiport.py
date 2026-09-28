"""Conservative local replacement of a passive cable block.

Ports are existing electrical nodes of one neuron, not face electrodes or
new synapses. A candidate must account for the membrane it removes, preserve
all crossing branches and protect every anatomical input/output site. This
module does not infer voxel-to-cable correspondence or active physiology.
"""
import hashlib
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu


def laplacian(node_count, edges, axial):
    e=np.asarray(edges);g=np.asarray(axial,dtype=float)
    if (e.ndim!=2 or e.shape[1]!=2 or e.dtype.kind not in 'iu' or g.shape!=(len(e),)
            or np.any(e<0) or np.any(e>=node_count) or np.any(e[:,0]==e[:,1])
            or not np.isfinite(g).all() or np.any(g<=0)):
        raise ValueError('Invalid axial graph')
    diagonal=np.bincount(e.ravel(),weights=np.repeat(g,2),minlength=node_count)
    return diags(diagonal)-coo_matrix((np.r_[g,g],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(node_count,node_count)).tocsr()


def extract_block(edges, axial_nS, edge_area_um2, internal_nodes, node_count,
                  *, Cm_uF_cm2, Rm_ohm_cm2, protected_nodes):
    """Assign half of each removed edge's membrane to each endpoint, once."""
    e=np.asarray(edges);area=np.asarray(edge_area_um2,dtype=float)
    internal=np.asarray(internal_nodes);protected=np.asarray(protected_nodes)
    if (internal.ndim!=1 or not len(internal) or internal.dtype.kind not in 'iu'
            or len(np.unique(internal))!=len(internal) or np.any(internal<0)
            or np.any(internal>=node_count) or np.intersect1d(internal,protected).size
            or area.shape!=(len(e),) or np.any(area<=0) or not np.isfinite(area).all()
            or not np.isfinite(Cm_uF_cm2) or Cm_uF_cm2<=0
            or not np.isfinite(Rm_ohm_cm2) or Rm_ohm_cm2<=0):
        raise ValueError('Invalid block or attempted removal of a protected anatomical site')
    selected=np.isin(e,internal).any(axis=1);nodes=np.unique(e[selected])
    if not np.isin(internal,nodes).all():raise ValueError('Internal node has no membrane edges')
    ports=np.setdiff1d(nodes,internal)
    if not len(ports):raise ValueError('Replacement requires at least one external connection')
    local_edges=np.searchsorted(nodes,e[selected])
    local_area=np.bincount(local_edges.ravel(),weights=np.repeat(area[selected]/2,2),minlength=len(nodes))
    C=local_area*Cm_uF_cm2*1e-5;leak=local_area*10/Rm_ohm_cm2
    G=laplacian(len(nodes),local_edges,np.asarray(axial_nS)[selected])+diags(leak)
    return dict(nodes=nodes,internal_nodes=np.sort(internal),port_nodes=ports,
        port_indices=np.searchsorted(nodes,ports),removed_edges=np.flatnonzero(selected),
        C_nF=C,G_nS=G.tocsr(),leak_nS=leak,node_area_um2=local_area,
        electrical_units='nF,nS,mV,pA',anatomical_geometry_changed=False)


def require_candidate_ports(block, candidate_ports, *, touches_unmapped_boundary,
                            protected_sites_preserved, membrane_accounted):
    p=np.asarray(candidate_ports)
    if (p.ndim!=1 or p.dtype.kind not in 'iu' or len(np.unique(p))!=len(p)
            or not np.array_equal(np.sort(p),block['port_nodes'])):
        raise ValueError('Candidate changes or omits anatomical crossing ports')
    if touches_unmapped_boundary:
        raise ValueError('Candidate reaches an unaccounted acquisition boundary')
    if not protected_sites_preserved or not membrane_accounted:
        raise ValueError('Candidate loses anatomical sites or membrane accounting')


def port_admittance(G_nS,C_nF,port_indices,frequency_hz=0.):
    """Exact terminal Schur complement of the declared RC block at j*omega."""
    G=csr_matrix(G_nS);C=np.asarray(C_nF,dtype=float);p=np.asarray(port_indices)
    if (G.shape!=(len(C),len(C)) or np.any(C<=0) or not np.isfinite(C).all()
            or p.ndim!=1 or p.dtype.kind not in 'iu' or len(np.unique(p))!=len(p)
            or not len(p) or np.any(p<0) or np.any(p>=len(C))
            or not np.isfinite(frequency_hz) or frequency_hz<0):
        raise ValueError('Invalid RC transfer request')
    A=(G+diags(2j*np.pi*frequency_hz*C)).tocsc()
    inner=np.setdiff1d(np.arange(len(C)),p)
    result=A[p][:,p].toarray()
    if len(inner):
        result-=A[p][:,inner]@splu(A[inner][:,inner]).solve(A[inner][:,p].toarray())
    return result


def assemble_replacement(parent_G,parent_C,block,candidate_G,candidate_C,
                         candidate_ports,candidate_port_indices):
    """Remove old local edges/membrane before adding a port-compatible block.

Returns the original-to-new node map, so callers must check every anatomical
site after assembly. Internal candidate nodes have no invented synapse IDs.
Candidate geometry admission is a separate required check, not inferred here.
"""
    G=csr_matrix(parent_G);C=np.asarray(parent_C,dtype=float)
    newG=csr_matrix(candidate_G);newC=np.asarray(candidate_C,dtype=float)
    ports=np.asarray(candidate_ports);pi=np.asarray(candidate_port_indices)
    require_candidate_ports(block,ports,touches_unmapped_boundary=False,
                            protected_sites_preserved=True,membrane_accounted=True)
    if (G.shape!=(len(C),len(C)) or newG.shape!=(len(newC),len(newC))
            or pi.shape!=ports.shape or pi.dtype.kind not in 'iu' or np.any(pi<0)
            or np.any(pi>=len(newC)) or len(np.unique(pi))!=len(pi)
            or not np.isfinite(newC).all() or np.any(newC<=0)):
        raise ValueError('Invalid replacement dimensions/membrane')
    # Only small boundary blocks belong in this explicit admission interface.
    dense=newG.toarray()
    if (not np.isfinite(dense).all() or not np.allclose(dense,dense.T,rtol=0,atol=1e-11)
            or np.linalg.eigvalsh(dense).min()<=0):
        raise ValueError('Replacement is not a passive positive-definite block')
    if not np.isclose(newC.sum(),block['C_nF'].sum(),rtol=1e-12,atol=0):
        raise ValueError('Replacement changes the declared membrane capacitance')
    old_nodes=block['nodes'];local=block['G_nS'].tocoo()
    outside=G-coo_matrix((local.data,(old_nodes[local.row],old_nodes[local.col])),shape=G.shape).tocsr()
    outside_C=C.copy();outside_C[old_nodes]-=block['C_nF']
    internal=block['internal_nodes'];keep=np.setdiff1d(np.arange(len(C)),internal)
    if (abs(outside[internal]).max()>1e-10 or
            np.max(abs(outside_C[internal]))>1e-14 or np.any(outside_C[keep]<-1e-14)):
        raise ValueError('Local accounting does not match the original membrane/edges')
    mapping=np.full(len(C),-1,dtype=np.int64);mapping[keep]=np.arange(len(keep))
    ni=np.setdiff1d(np.arange(len(newC)),pi)
    replacement_map=np.empty(len(newC),dtype=np.int64)
    replacement_map[pi]=mapping[ports];replacement_map[ni]=len(keep)+np.arange(len(ni))
    size=len(keep)+len(ni);base=outside[keep][:,keep].tocoo();patch=newG.tocoo()
    finalG=coo_matrix((np.r_[base.data,patch.data],
        (np.r_[base.row,replacement_map[patch.row]],np.r_[base.col,replacement_map[patch.col]])),shape=(size,size)).tocsr()
    finalC=np.zeros(size);finalC[:len(keep)]=np.maximum(outside_C[keep],0.)
    np.add.at(finalC,replacement_map,newC)
    if np.any(finalC<=0):raise ValueError('Unassigned membrane state')
    np.testing.assert_allclose(finalC.sum(),C.sum(),rtol=1e-12,atol=0)
    return finalG,finalC,mapping,replacement_map


class PassiveNetwork:
    """General passive RC operator with explicit persistent membrane memory."""
    def __init__(self,G_nS,C_nF,time_ns=0):
        self.G=csr_matrix(G_nS);self.C=np.asarray(C_nF,dtype=float).copy()
        if (self.G.shape!=(len(self.C),len(self.C)) or np.any(self.C<=0)
                or not np.isfinite(self.C).all() or not np.isfinite(self.G.data).all()
                or type(time_ns) is not int or time_ns<0):raise ValueError('Invalid passive network')
        self.G.sort_indices();self.time_ns=time_ns;self.delta=np.zeros(len(self.C));self._cache=None
        h=hashlib.sha256()
        for a in (self.G.data,self.G.indices,self.G.indptr,self.C):
            h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
        self.identity=h.hexdigest()

    def dc(self,current_pA):
        current=np.asarray(current_pA,dtype=float)
        if current.shape!=self.C.shape or not np.isfinite(current).all():raise ValueError('Invalid current')
        return splu(self.G.tocsc()).solve(current)

    def advance(self,dt_ns,current_pA):
        current=np.asarray(current_pA,dtype=float)
        if (type(dt_ns) is not int or dt_ns<=0 or current.shape!=self.C.shape
                or not np.isfinite(current).all()):raise ValueError('Invalid step/current')
        if self._cache is None or self._cache[0]!=dt_ns:
            self._cache=(dt_ns,splu((self.G+diags(self.C/(dt_ns*1e-9))).tocsc()))
        new=self._cache[1].solve(current+self.C*self.delta/(dt_ns*1e-9))
        if not np.isfinite(new).all():raise ValueError('Nonfinite passive solution')
        self.delta=new;self.time_ns+=dt_ns;return self.delta.copy()

    def state_dict(self):
        return dict(schema='neck_passive_network_state_v1',identity=self.identity,
                    time_ns=self.time_ns,delta_mV=self.delta.copy())

    def load_state_dict(self,s):
        if (set(s)!={'schema','identity','time_ns','delta_mV'} or s['schema']!='neck_passive_network_state_v1'
                or s['identity']!=self.identity or type(s['time_ns']) is not int or s['time_ns']<0
                or np.asarray(s['delta_mV']).shape!=self.C.shape or not np.isfinite(s['delta_mV']).all()):
            raise ValueError('Different or incomplete passive state')
        self.time_ns=s['time_ns'];self.delta=np.array(s['delta_mV'],copy=True);self._cache=None
