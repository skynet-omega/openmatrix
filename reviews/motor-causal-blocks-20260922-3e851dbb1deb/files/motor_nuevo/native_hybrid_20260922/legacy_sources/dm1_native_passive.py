"""Passive electrical candidate on a declared native forest, in nF/nS/mV/pA.

Frustum geometry, lumped membrane area and transferred specific electrical
properties are explicit modeling assumptions. No radius fit, healing, spike
boundary, local release law or native conductance identification is implied.
"""
import hashlib
import json
import numpy as np
from numba import njit


def _positive(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
        raise ValueError('Require positive '+name)
    return float(value)


def forest_order(nodes, edges):
    e=np.asarray(edges)
    if (type(nodes) is not int or nodes<2 or e.ndim!=2 or e.shape[1]!=2 or e.dtype.kind not in 'iu'
            or not len(e) or np.any(e<0) or np.any(e>=nodes) or np.any(e[:,0]==e[:,1])):
        raise ValueError('Invalid forest endpoints')
    parent=np.full(nodes,-1,dtype=np.int64)
    if len(np.unique(e[:,1]))!=len(e):raise ValueError('Multiple parents')
    parent[e[:,1]]=e[:,0];roots=np.flatnonzero(parent<0)
    if len(roots)!=nodes-len(e):raise ValueError('Forest count mismatch')
    order=np.empty(nodes,dtype=np.int64);order[:len(roots)]=roots;end=len(roots)
    starts=np.r_[0,np.cumsum(np.bincount(e[:,0],minlength=nodes))]
    children=e[np.argsort(e[:,0],kind='stable'),1]
    cursor=0
    while cursor<end:
        row=int(order[cursor]);kids=children[starts[row]:starts[row+1]]
        order[end:end+len(kids)]=kids;end+=len(kids);cursor+=1
    if end!=nodes:raise ValueError('Cycle or unreachable component')
    return parent,order,roots


def frustum_parameters(xyz_um, radius_um, edges, *, Rm_ohm_cm2, Cm_uF_cm2, Ri_ohm_cm):
    xyz=np.asarray(xyz_um,dtype=float);r=np.asarray(radius_um,dtype=float);e=np.asarray(edges)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or r.shape!=(len(xyz),) or not np.isfinite(xyz).all() or not np.isfinite(r).all() or np.any(r<=0):
        raise ValueError('Invalid physical geometry')
    forest_order(len(xyz),e)
    Rm=_positive(Rm_ohm_cm2,'Rm');Cm=_positive(Cm_uF_cm2,'Cm');Ri=_positive(Ri_ohm_cm,'Ri')
    length=np.linalg.norm(xyz[e[:,1]]-xyz[e[:,0]],axis=1)
    if np.any(length<=0):raise ValueError('Zero-length edge')
    a=r[e[:,0]];b=r[e[:,1]]
    # Lateral frustum area, no end caps or invented spherical soma.
    area=np.pi*(a+b)*np.hypot(length,b-a)
    node_area=np.bincount(e.ravel(),weights=np.repeat(area/2,2),minlength=len(xyz))
    if np.any(node_area<=0):raise ValueError('Isolated node has no declared membrane area')
    axial=1e5*np.pi*a*b/(Ri*length)  # exact axial resistance integral for a linear taper
    return dict(C_nF=node_area*Cm*1e-5,leak_nS=node_area*10/Rm,axial_nS=axial,
                node_area_um2=node_area,edge_area_um2=area,edge_length_um=length)


def insert_sites(xyz_um,radius_um,edges,site_edges,site_fractions,*,merge_distance_um=1e-6):
    """Insert contact nodes without snapping to the original coarse endpoints.

Only positions within the declared numerical tolerance (default .001 nm) are
merged. All contacts retain a node map and their displacement. The continuous
linear taper, cable length, lateral area and axial series resistance survive.
"""
    xyz=np.asarray(xyz_um,dtype=float);r=np.asarray(radius_um,dtype=float);e=np.asarray(edges)
    se=np.asarray(site_edges);sf=np.asarray(site_fractions,dtype=float)
    forest_order(len(xyz),e);tol=_positive(merge_distance_um,'merge distance')
    if (se.ndim!=1 or se.dtype.kind not in 'iu' or sf.shape!=se.shape or np.any(se<0) or np.any(se>=len(e))
            or not np.isfinite(sf).all() or np.any(sf<0) or np.any(sf>1) or r.shape!=(len(xyz),)
            or not np.isfinite(xyz).all() or not np.isfinite(r).all() or np.any(r<=0)):
        raise ValueError('Invalid contact interpolation')
    lengths=np.linalg.norm(xyz[e[:,1]]-xyz[e[:,0]],axis=1)
    if np.any(lengths<=0):raise ValueError('Zero-length edge')
    new_xyz=[];new_r=[];new_edges=[];mapped=np.empty(len(se),dtype=np.int64);movement=np.empty(len(se))
    permutation=np.lexsort((sf,se));sorted_edges=se[permutation]
    touched=np.unique(se);untouched=np.ones(len(e),dtype=bool);untouched[touched]=False
    new_edges.extend(e[untouched].tolist())
    for edge in touched:
        a,b=map(int,e[edge]);rows=permutation[np.searchsorted(sorted_edges,edge,'left'):np.searchsorted(sorted_edges,edge,'right')]
        last=a;position=0.
        for row in rows:
            f=float(sf[row]);length=float(lengths[edge])
            if f*length<=tol:node=a;at=0.
            elif (1-f)*length<=tol:node=b;at=1.
            elif (f-position)*length<=tol:node=last;at=position
            else:
                node=len(xyz)+len(new_xyz);at=f
                new_xyz.append(xyz[a]+f*(xyz[b]-xyz[a]));new_r.append(r[a]+f*(r[b]-r[a]))
                new_edges.append([last,node]);last=node;position=f
            mapped[row]=node;movement[row]=abs(f-at)*length
        new_edges.append([last,b])
    return dict(xyz_um=np.vstack((xyz,np.asarray(new_xyz).reshape(-1,3))),radius_um=np.r_[r,new_r],
                edges=np.asarray(new_edges,dtype=np.int64),site_nodes=mapped,site_movement_um=movement,
                original_node_count=len(xyz),numerical_merge_um=tol)


@njit(cache=False)
def _factor(diagonal,parent,order,axial_to_parent):
    d=diagonal.copy()
    for index in range(len(order)-1,-1,-1):
        node=order[index];p=parent[node]
        if d[node]<=0 or not np.isfinite(d[node]):raise ValueError('Nonpositive forest pivot')
        if p>=0:d[p]-=axial_to_parent[node]**2/d[node]
    return d


@njit(cache=False)
def _solve(factored,rhs,parent,order,axial_to_parent):
    b=rhs.copy()
    for index in range(len(order)-1,-1,-1):
        node=order[index];p=parent[node]
        if p>=0:b[p]+=axial_to_parent[node]*b[node]/factored[node]
    for index in range(len(order)):
        node=order[index];p=parent[node]
        if p>=0:b[node]=(b[node]+axial_to_parent[node]*b[p])/factored[node]
        else:b[node]/=factored[node]
    return b


class NativePassiveCable:
    SCHEMA='native_dm1_passive_forest_v1'

    def __init__(self,body_id,edges,C_nF,leak_nS,axial_nS,*,rest_mV,time_ns=0):
        self.C=np.asarray(C_nF,dtype=float).copy();self.leak=np.asarray(leak_nS,dtype=float).copy()
        self.axial=np.asarray(axial_nS,dtype=float).copy();self.edges=np.asarray(edges).copy()
        if type(body_id) is not int or body_id<=0 or not np.isfinite(rest_mV):raise ValueError('Explicit identity and rest required')
        self.body_id=body_id;self.rest=float(rest_mV);self.parent,self.order,self.roots=forest_order(len(self.C),self.edges)
        if (self.C.ndim!=1 or self.leak.shape!=self.C.shape or self.axial.shape!=(len(self.edges),)
                or any(not np.isfinite(v).all() or np.any(v<=0) for v in [self.C,self.leak,self.axial])):
            raise ValueError('Invalid passive coefficients')
        self.axial_to_parent=np.zeros(len(self.C));self.axial_to_parent[self.edges[:,1]]=self.axial
        self.diagonal=self.leak+np.bincount(self.edges.ravel(),weights=np.repeat(self.axial,2),minlength=len(self.C))
        h=hashlib.sha256(json.dumps(dict(body_id=body_id,rest_mV=self.rest,schema=self.SCHEMA),sort_keys=True).encode())
        for v in [self.edges,self.C,self.leak,self.axial]:h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        self.identity=h.hexdigest();self.initial_time_ns=self._clock(time_ns);self.time_ns=self.initial_time_ns
        self.delta=np.zeros(len(self.C));self._cache=None
        for a in [self.C,self.leak,self.axial,self.edges,self.parent,self.order,self.roots,self.axial_to_parent,self.diagonal]:a.flags.writeable=False

    @staticmethod
    def _clock(value):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0 or value>np.iinfo(np.int64).max:
            raise ValueError('Nonnegative integer nanosecond clock required')
        return int(value)

    def _inputs(self,current_pA,conductance_nS,reversal_mV):
        current=np.asarray(current_pA,dtype=float)
        g=np.zeros_like(self.C) if conductance_nS is None else np.asarray(conductance_nS,dtype=float)
        reversal=np.asarray(reversal_mV,dtype=float)
        if current.shape!=self.C.shape or g.shape!=self.C.shape or reversal.shape not in [(),self.C.shape] or np.any(g<0) or not all(np.isfinite(x).all() for x in [current,g,reversal]):
            raise ValueError('Invalid physical current, conductance or reversal')
        rhs=current+g*(reversal-self.rest)
        if not np.isfinite(rhs).all():raise ValueError('Input overflow')
        return g,rhs

    def dc(self,current_pA,*,conductance_nS=None,reversal_mV=0.):
        g,rhs=self._inputs(current_pA,conductance_nS,reversal_mV)
        d=_factor(self.diagonal+g,self.parent,self.order,self.axial_to_parent)
        return _solve(d,rhs,self.parent,self.order,self.axial_to_parent)

    def advance(self,dt_ns,current_pA,*,conductance_nS=None,reversal_mV=0.):
        """L-stable backward Euler. All times/input units explicit; no reset law."""
        dt=self._clock(dt_ns);end=self._clock(self.time_ns+dt)
        if dt==0:raise ValueError('Positive step required')
        g,rhs=self._inputs(current_pA,conductance_nS,reversal_mV)
        if self._cache is None or dt!=self._cache[0] or not np.array_equal(g,self._cache[1]):
            d=_factor(self.diagonal+g+self.C/(dt*1e-9),self.parent,self.order,self.axial_to_parent)
            self._cache=(dt,g.copy(),d)
        x=_solve(self._cache[2],rhs+self.C/(dt*1e-9)*self.delta,self.parent,self.order,self.axial_to_parent)
        if not np.isfinite(x).all():raise ValueError('Nonfinite membrane state')
        self.delta=x;self.time_ns=end
        return self.rest+self.delta

    def apply_G(self,delta):
        x=np.asarray(delta,dtype=float);e=self.edges
        if x.shape!=self.C.shape or not np.isfinite(x).all():raise ValueError('Invalid voltage vector')
        flow=self.axial*(x[e[:,0]]-x[e[:,1]])
        return self.leak*x+np.bincount(e.ravel(),weights=np.column_stack((flow,-flow)).ravel(),minlength=len(x))

    def energy_fJ(self):return .5*float(np.dot(self.C,self.delta**2))

    def state_dict(self):
        return dict(schema=self.SCHEMA,identity=self.identity,initial_time_ns=self.initial_time_ns,time_ns=self.time_ns,delta_mV=self.delta.copy())

    def load_state_dict(self,state):
        if state.get('schema')!=self.SCHEMA or state.get('identity')!=self.identity or state.get('initial_time_ns')!=self.initial_time_ns:
            raise ValueError('Different native cable or preparation')
        clock=self._clock(state.get('time_ns'));x=np.asarray(state.get('delta_mV'),dtype=float)
        if clock<self.initial_time_ns or x.shape!=self.C.shape or not np.isfinite(x).all():raise ValueError('Invalid native state')
        self.delta=x.copy();self.time_ns=clock;self._cache=None
