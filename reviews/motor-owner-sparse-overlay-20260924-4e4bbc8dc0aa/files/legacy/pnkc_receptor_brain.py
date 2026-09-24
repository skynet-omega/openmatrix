"""Published PN->KC receptor kinetics in the continuing, rate-based CNS.

Only the selected cholinergic ALPN->KCgamma terms bypass the generic release
filter. Other PN projections, all anatomical weights, regional routing and KC
rate coordinates remain intact. The mean-transmitter closure and conversion
back to inherited current units are explicit prostheses, not measured nS.
"""
from pathlib import Path
import copy, json
import numpy as np

from lamina_boundary_brain import LaminaBoundaryBrain, GpuLaminaBoundaryBrain, BOUNDARY_KEYS
from kcgamma_regional_brain import _record_hash, _regional_currents
from session_io import sha256
from synaptic_visual_brain import _hash_array

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SELECTION=ROOT/'data/pnkc_receptor_20260910/selection.npz'
PARENT_STATE_SIZE=359074
POLICY='canonical_alpn_kcgamma_mean_receptor_v1'
KEYS=BOUNDARY_KEYS|{'pnkc_receptor_manifest'}
SELECTION_KEYS={'source_rows','source_ids','target_rows','target_ids','csr_positions',
    'pre_rows','post_rows','contacts','source_ground_truth'}
ALPHA_PER_S=2500.
BETA_PER_S=400.
TRANSMITTER_AMPLITUDE=.5
TRANSMITTER_DURATION_S=.0003
K_PER_HZ=ALPHA_PER_S*TRANSMITTER_AMPLITUDE*TRANSMITTER_DURATION_S


def parameters():
    return dict(alpha_per_s=ALPHA_PER_S,beta_per_s=BETA_PER_S,
        transmitter_amplitude=TRANSMITTER_AMPLITUDE,transmitter_duration_s=TRANSMITTER_DURATION_S,
        mean_activation_per_hz=K_PER_HZ)


def receptor_coefficients(rate_hz):
    activation=K_PER_HZ*rate_hz
    return activation/(activation+BETA_PER_S),activation+BETA_PER_S


def _selection(reference, data):
    b=reference.brain
    if set(data)!=SELECTION_KEYS or b.n_neurons!=166700:
        raise ValueError('PNKC receptor requires the canonical CNS and complete selection')
    for key in SELECTION_KEYS-{'source_ground_truth'}:
        a=data[key]
        if not isinstance(a,np.ndarray) or a.dtype!=np.int64 or a.ndim!=1 or not len(a):
            raise ValueError('Invalid PNKC integer selection: '+key)
    source,target,pos=(data[k] for k in ('source_rows','target_rows','csr_positions'))
    if (any(np.any(np.diff(a)<=0) for a in (source,target,pos))
            or source[0]<0 or target[0]<0 or source[-1]>=b.n_neurons or target[-1]>=b.n_neurons
            or pos[0]<0 or pos[-1]>=b.W.nnz
            or not np.array_equal(b.node_ids[source],data['source_ids'])
            or not np.array_equal(b.node_ids[target],data['target_ids'])
            or not np.array_equal(b.W.indices[pos],data['pre_rows'])
            or not np.array_equal(np.searchsorted(b.W.indptr,pos,side='right')-1,data['post_rows'])
            or not np.array_equal(np.unique(data['pre_rows']),source)
            or not np.array_equal(np.unique(data['post_rows']),target)
            or data['contacts'].shape!=pos.shape or np.any(data['contacts']<=0)
            or data['source_ground_truth'].shape!=source.shape
            or data['source_ground_truth'].dtype.kind!='U'
            or np.any(reference.weights64[pos]<=0)
            or np.any(reference.visual_mask[np.r_[source,target]])
            or not np.isin(target,reference.regional_rows).all()
            or np.intersect1d(source,reference.regional_rows).size):
        raise ValueError('PNKC identities, topology, positive weights or rate-mode mapping differ')
    # Every selected PN->target term must be listed; unselected afferents stay generic.
    actual=[]
    for row in target:
        edges=np.arange(b.W.indptr[row],b.W.indptr[row+1],dtype=np.int64)
        actual.extend(edges[np.isin(b.W.indices[edges],source)])
    if not np.array_equal(np.asarray(actual,dtype=np.int64),pos):
        raise ValueError('Incomplete PN->KC pair mapping')


class PnkcReceptorBrain(LaminaBoundaryBrain):
    SCHEMA='matrix_pnkc_receptor_brain_fp64_v1'
    PNKC_PARENT_CLASS=LaminaBoundaryBrain

    @classmethod
    def adopt(cls,reference,*,enabled=True,selection_path=DEFAULT_SELECTION):
        if type(reference) is not cls.PNKC_PARENT_CLASS or type(enabled) is not bool or len(reference.state)!=PARENT_STATE_SIZE:
            raise ValueError('PNKC adoption requires the exact continuing lamina boundary parent')
        path=Path(selection_path).resolve()
        source=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        if source['selection_sha256']!=sha256(path):raise ValueError('PNKC selection checksum changed')
        with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        _selection(reference,data)
        old=reference.state_dict();record={k:v for k,v in old.items() if k!='schema'}
        rows=data['source_rows']
        # Match the previous branch current at adoption; no fabricated past spikes.
        initial=(K_PER_HZ/BETA_PER_S)*reference.transmission_release()[rows]*reference.caps[rows]
        m=dict(policy=POLICY,enabled=enabled,source_schema=reference.SCHEMA,parent_state_size=PARENT_STATE_SIZE,
            adoption_time_ns=int(reference.time_ns),parameters=parameters(),**data,
            source=dict(selection_path=str(path),manifest=source,selection_arrays_sha256=_record_hash(data)),
            initial_parent_record_sha256=_record_hash(record),initial_new_state_sha256=_hash_array(initial),
            node_ids_sha256=_hash_array(reference.brain.node_ids),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),
            anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            anatomical_weights_sha256=_hash_array(reference.brain.W.data),
            inherited_regional_manifest_sha256=_record_hash(reference.regional_manifest),
            state_order=['mean_open_receptor_fraction'],
            equation='db/dt=k*f_PN*(1-b)-beta*b; f_PN=q_PN*caps_PN; effective_rate=(beta/k)*b; retain original pair weight.',
            initialization='b=k/beta*previous PN filtered rate; match initial branch current, unknown earlier receptor history.',
            compression='One state per PN for identical receptor kinetics and identical initialization at its selected KC terminals. Individual anatomical pair strengths remain stored and applied.',
            source_scope='Turner2008 female population voltage-clamp kinetic estimates transferred to male gamma KC; mean transmitter rather than explicit PN spikes.',
            remaining_prostheses=['Rate-to-mean-transmitter closure; homogeneous kinetics across mapped terminals',
                'Inherited anatomical-contact-to-current scale; low-rate strength normalization is an interface assumption',
                'KC rate transfer and excitability; no KC membrane voltage or full voltage-dependent synaptic conductance',
                'APL remains the inherited neural component, without a localized graded electrical replacement'],
            changed_term='Only mapped ALPN->KCgamma release filter; all other afferents and PN projections retained.',
            new_canonical_neurons=0,new_anatomical_pairs=0,biological_validation=False,parameter_fitting=False,
            external_neuron_model_integrated=False,learning_demonstrated=False)
        m['record_sha256']=_record_hash(m)
        old.update(schema=cls.SCHEMA,state=np.r_[reference.state,initial],pnkc_receptor_manifest=m)
        return cls.from_state(reference.brain,old)

    @property
    def receptor_state(self):return self.state[PARENT_STATE_SIZE:].copy()

    @property
    def boundary_state(self):
        return self.state[self.boundary_parent_state_size:PARENT_STATE_SIZE].reshape(4,-1).copy()

    def _norm_size(self):
        if self.pnkc_receptor_manifest['enabled']:return len(self.state)
        if self.boundary_manifest['enabled']:return PARENT_STATE_SIZE
        return LaminaBoundaryBrain._norm_size(self)

    def _build_pnkc_cache(self):
        m=self.pnkc_receptor_manifest;b=self.brain
        data={k:m[k] for k in SELECTION_KEYS};_selection(self,data)
        if (m['policy']!=POLICY or type(m['enabled']) is not bool
                or m['source_schema']!=self.PNKC_PARENT_CLASS.SCHEMA or m['parent_state_size']!=PARENT_STATE_SIZE
                or m['parameters']!=parameters() or m['record_sha256']!=_record_hash(m)
                or m['source']['selection_arrays_sha256']!=_record_hash(data)
                or m['biological_validation'] is not False or m['parameter_fitting'] is not False
                or m['external_neuron_model_integrated'] is not False
                or m['new_canonical_neurons']!=0 or m['new_anatomical_pairs']!=0
                or type(m['adoption_time_ns']) is not int or not 0<=m['adoption_time_ns']<=self.time_ns
                or self.state.dtype!=np.float64 or self.state.shape!=(PARENT_STATE_SIZE+len(m['source_rows']),)
                or not np.isfinite(self.state).all() or np.any((self.state<0)|(self.state>1))):
            raise ValueError('Changed PNKC receptor mechanism, layout or history')
        for key,a in [('node_ids',b.node_ids),('anatomical_indptr',b.W.indptr),
                      ('anatomical_indices',b.W.indices),('anatomical_weights',b.W.data)]:
            if m[key+'_sha256']!=_hash_array(a):raise ValueError('PNKC graph changed: '+key)
        if m['inherited_regional_manifest_sha256']!=_record_hash(self.regional_manifest):
            raise ValueError('PNKC adoption changed regional routing')
        if self.time_ns==m['adoption_time_ns']:
            parent={k:v for k,v in self.state_dict().items() if k not in ('schema','pnkc_receptor_manifest')}
            parent['state']=self.state[:PARENT_STATE_SIZE].copy()
            if (_record_hash(parent)!=m['initial_parent_record_sha256']
                    or _hash_array(self.receptor_state)!=m['initial_new_state_sha256']):
                raise ValueError('PNKC adoption changed inherited state or initial receptor history')
        # Reuse the existing regional current operator on these rows only.
        # All their incoming terms stay present, including APL and KC axonal routing.
        rows=m['target_rows'];r=self._regional_cache
        ptr=np.r_[0,np.cumsum(b.W.indptr[rows+1]-b.W.indptr[rows])].astype(np.int64)
        c=dict(rows=rows.copy(),local_ptr=ptr,mode=np.zeros(ptr[-1],dtype=np.int8),
               fraction=np.zeros(ptr[-1]),slot=np.full(ptr[-1],-1,dtype=np.int32),source_rows=m['source_rows'].copy())
        if self.regional_manifest['enabled']:
            for j,row in enumerate(rows):
                k=np.searchsorted(r['rows'],row)
                if k<len(r['rows']) and r['rows'][k]==row:
                    for field in ('mode','fraction','slot'):
                        c[field][ptr[j]:ptr[j+1]]=r[field][r['local_ptr'][k]:r['local_ptr'][k+1]]
        self._pnkc_cache=c

    def _coefficients(self,state,drive,light):
        target,rate=LaminaBoundaryBrain._coefficients(self,state[:PARENT_STATE_SIZE],drive,light)
        if not self.pnkc_receptor_manifest['enabled']:
            return np.r_[target,state[PARENT_STATE_SIZE:]],np.r_[rate,np.zeros(len(state)-PARENT_STATE_SIZE)]
        c=self._pnkc_cache;rows=c['source_rows'];gate=state[PARENT_STATE_SIZE:]
        rt,rr=receptor_coefficients(state[rows]*self.caps[rows])
        release=state[self.transmission_start:self.inherited_state_size].copy()
        release[rows]=(BETA_PER_S/K_PER_HZ)*gate/self.caps[rows]
        sax=state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        _regional_currents(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],
            self.brain.W.indptr,self.brain.W.indices,self.weights64,release,sax,self.caps,self.visual_mask,
            self.rate_gain,self.rate_theta,drive,self.visual_output_connected,target)
        return np.r_[target,rt],np.r_[rate,rr]

    def state_dict(self):
        saved=super().state_dict();saved['pnkc_receptor_manifest']=copy.deepcopy(self.pnkc_receptor_manifest);return saved

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved.get('schema')!=cls.SCHEMA:
            raise ValueError('Incomplete PNKC receptor state or wrong backend')
        state=saved['state']
        parent={k:v for k,v in saved.items() if k!='pnkc_receptor_manifest'}
        parent.update(schema=cls.PNKC_PARENT_CLASS.SCHEMA,state=state[:PARENT_STATE_SIZE].copy())
        base=cls.PNKC_PARENT_CLASS.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.state=state.copy();obj.pnkc_receptor_manifest=copy.deepcopy(saved['pnkc_receptor_manifest'])
        obj._build_pnkc_cache();return obj


class GpuPnkcReceptorBrain(PnkcReceptorBrain,GpuLaminaBoundaryBrain):
    SCHEMA='matrix_pnkc_receptor_brain_fp64_cuda_v1'
    PNKC_PARENT_CLASS=GpuLaminaBoundaryBrain

    def _build_pnkc_cache(self):
        super()._build_pnkc_cache()
        import cupy as cp
        self._pnkc_cuda={k:cp.asarray(v) for k,v in self._pnkc_cache.items()}

    def coefficients_gpu(self,state,drive,light):
        import cupy as cp
        target,rate=GpuLaminaBoundaryBrain.coefficients_gpu(self,state[:PARENT_STATE_SIZE],drive,light)
        if not self.pnkc_receptor_manifest['enabled']:
            return cp.concatenate((target,state[PARENT_STATE_SIZE:])),cp.concatenate((rate,cp.zeros(len(state)-PARENT_STATE_SIZE)))
        r=self._pnkc_cuda;c=self.cuda;rows=r['source_rows'];gate=state[PARENT_STATE_SIZE:]
        rt,rr=receptor_coefficients(state[rows]*c['caps'][rows])
        release=state[self.transmission_start:self.inherited_state_size].copy()
        release[rows]=(BETA_PER_S/K_PER_HZ)*gate/c['caps'][rows]
        sax=state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        n=len(self._pnkc_cache['rows'])
        args=(np.int32(n),r['rows'],r['local_ptr'],r['mode'],r['fraction'],r['slot'],c['indptr'],c['indices'],
            c['weights'],release,sax,c['caps'],c['visual'],c['gain'],c['theta'],drive,
            np.bool_(self.visual_output_connected),target)
        self._regional_current_kernel(((n*32+255)//256,),(256,),args)
        return cp.concatenate((target,rt)),cp.concatenate((rate,rr))

    @staticmethod
    def backend_identity():
        out=GpuLaminaBoundaryBrain.backend_identity()
        out['pnkc_receptor']='Turner2008 mean receptor kinetics on actual ALPN->KCgamma terms; inherited rate/current interface; FP64 stage-consistent state, existing regional warp reduction.'
        return out
