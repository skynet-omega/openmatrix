"""Stateful passive DM1 reference with physical input and observation ports.

No normalized firing rate is converted into current. No spikes or release are
inferred from this passive membrane. Conductances and reversals are explicit
inputs; gap coupling conserves equal/opposite current between declared ports.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.linalg import block_diag,eigh

SCHEMA='dm1_projected_passive_state_v1'


def _clock(value,positive=False):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or value<(1 if positive else 0) or value>np.iinfo(np.int64).max):
        raise ValueError('Clock must be an integer number of nanoseconds')
    return int(value)


class Dm1PnMembrane:
    def __init__(self,artifact,cell_ids=(10176,10208),*,time_ns=0):
        artifact=Path(artifact)
        self.artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
        with np.load(artifact,allow_pickle=False) as z:
            self.C=z['C_nF'].copy();self.G=z['G_nS'].copy()
            self.input=z['input_basis'].copy();self.observation=z['observation'].copy()
            self.shunt_G=z['shunt_G'].copy();self.shunt_b=z['shunt_b'].copy()
            self.rest=float(z['rest_mV']);self.port_names=tuple(z['port_names'].tolist())
        self.dimension=len(self.C);m=self.dimension;p=len(self.port_names)
        if p!=3 or self.G.shape!=(m,m) or self.C.shape!=(m,m):
            raise ValueError('Invalid passive matrix dimensions')
        for matrix in [self.C,self.G]:
            if not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T,rtol=1e-11,atol=1e-13):
                raise ValueError('Nonreciprocal passive matrix')
            if np.linalg.eigvalsh(matrix).min()<=0:raise ValueError('Nonpassive matrix')
        if (self.input.shape!=(m,p) or self.observation.shape!=(p,m)
                or self.shunt_G.shape!=(p,m,m) or self.shunt_b.shape!=(p,m)
                or not np.isfinite(self.rest)):
            raise ValueError('Invalid electrical port dimensions')
        for array in [self.input,self.observation,self.shunt_G,self.shunt_b]:
            if not np.isfinite(array).all():raise ValueError('Nonfinite port')
        if (not np.allclose(self.input.T,self.observation,rtol=0,atol=1e-12)
                or not np.allclose(self.shunt_b,self.observation,rtol=0,atol=1e-12)):
            raise ValueError('Input/output port identity differs')
        for matrix in self.shunt_G:
            if not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(matrix).min()<-1e-12:
                raise ValueError('Nonpassive local conductance')
        ids=np.asarray(cell_ids)
        if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
                or np.any(ids<=0) or np.any(ids>np.iinfo(np.int64).max)
                or len(np.unique(ids))!=len(ids)):
            raise ValueError('Require unique positive integer cell IDs')
        self.cell_ids=ids.astype(np.int64,copy=True);self.cells=len(ids)
        self.time_ns=_clock(time_ns);self.delta=np.zeros((self.cells,m))
        self._C=block_diag(*[self.C]*self.cells)
        self._cached=None
        for array in [self.C,self.G,self.input,self.observation,self.shunt_G,self.shunt_b,self.cell_ids,self._C]:
            array.flags.writeable=False

    def voltage_mV(self):
        return self.rest+self.delta@self.observation.T

    def stored_energy_fJ(self):
        """Capacitive energy relative to the leak reference: nF*mV² = fJ."""
        return .5*float(np.einsum('bi,ij,bj->',self.delta,self.C,self.delta))

    def state_dict(self):
        return dict(schema=SCHEMA,artifact_sha256=self.artifact_sha256,
            cell_ids=self.cell_ids.copy(),time_ns=self.time_ns,delta_mV=self.delta.copy())

    def load_state_dict(self,state):
        if state.get('schema')!=SCHEMA or state.get('artifact_sha256')!=self.artifact_sha256:
            raise ValueError('State belongs to another membrane reference')
        ids=np.asarray(state.get('cell_ids'))
        if ids.dtype.kind not in 'iu' or not np.array_equal(ids,self.cell_ids):
            raise ValueError('State belongs to different canonical identities/order')
        clock=_clock(state.get('time_ns'));delta=np.asarray(state.get('delta_mV'),dtype=float)
        if delta.shape!=self.delta.shape or not np.isfinite(delta).all():
            raise ValueError('Invalid membrane coordinates')
        self.delta=delta.copy();self.time_ns=clock;self._cached=None

    def _gap_vectors(self,gaps):
        seen=set();vectors=[]
        for a,pa,b,pb,g in gaps:
            if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in [a,pa,b,pb]):
                raise ValueError('Gap endpoints require integer cell and port indices')
            if not 0<=a<self.cells or not 0<=b<self.cells or not 0<=pa<3 or not 0<=pb<3:
                raise ValueError('Gap endpoint outside declared ports')
            pair=tuple(sorted(((int(a),int(pa)),(int(b),int(pb)))))
            if pair[0]==pair[1] or pair in seen or not np.isfinite(g) or g<0:
                raise ValueError('Duplicate, self or invalid gap conductance')
            seen.add(pair)
            w=np.zeros(self.cells*self.dimension)
            w[a*self.dimension:(a+1)*self.dimension]+=self.observation[pa]
            w[b*self.dimension:(b+1)*self.dimension]-=self.observation[pb]
            vectors.append((float(g),w))
        return vectors

    def advance(self,dt_ns,current_pA,*,conductances=(),gaps=()):
        """Exact evolution for held inputs during dt_ns.

current_pA has shape (cells,3). Each conductance entry is (g_nS, E_mV),
with g of that same shape and E either scalar or that shape. The shunt is
distributed over the original port sites. Gap tuples (cell,port,cell,port,g_nS)
instead connect port voltages by an effective reciprocal conductance; a tuft
gap is a distributed effective port, not a mapped anatomical gap junction.
        """
        dt_ns=_clock(dt_ns,positive=True);next_time=_clock(self.time_ns+dt_ns)
        current=np.asarray(current_pA,dtype=float)
        if current.shape!=(self.cells,3) or not np.isfinite(current).all():
            raise ValueError('Explicit current_pA must have shape (cells,3)')
        A=block_diag(*[self.G]*self.cells);rhs=current@self.input.T
        for conductance,reversal in conductances:
            g=np.asarray(conductance,dtype=float);e=np.asarray(reversal,dtype=float)
            if (g.shape!=(self.cells,3) or e.shape not in [(),g.shape]
                    or not np.isfinite(g).all() or np.any(g<0) or not np.isfinite(e).all()):
                raise ValueError('Invalid explicit local conductance or reversal')
            matrices=np.einsum('bp,pij->bij',g,self.shunt_G)
            A+=block_diag(*matrices)
            rhs+=(g*(e-self.rest))@self.shunt_b
        for g,w in self._gap_vectors(gaps):A+=g*np.outer(w,w)
        if not np.isfinite(A).all() or not np.isfinite(rhs).all():
            raise ValueError('Electrical input exceeds finite range')
        if self._cached is None or self._cached[0]!=dt_ns or not np.array_equal(self._cached[1],A):
            rates,vectors=eigh(A,self._C,check_finite=False)
            if rates.min()<=0:raise ValueError('Held system is not passive')
            dt=dt_ns*1e-9
            self._cached=(dt_ns,A.copy(),vectors,np.exp(-rates*dt),-np.expm1(-rates*dt)/rates)
        _,_,vectors,decay,integral=self._cached
        old=self.delta.ravel()
        new=vectors@(decay*(vectors.T@(self._C@old))+integral*(vectors.T@rhs.ravel()))
        if not np.isfinite(new).all():raise FloatingPointError('Nonfinite passive evolution')
        self.delta=new.reshape(self.cells,self.dimension);self.time_ns=next_time
        return self.voltage_mV()
