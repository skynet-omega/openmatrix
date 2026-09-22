"""Continue the canonical CNS with regional gamma release at remaining mapped outputs.

No new dynamics, gains, neurons or edges. Reuse the persisted axonal state on
known gamma-lobe fractions only. Preserve every other afferent and the complete
PN->KC/retinal/body state. This completes routing of an existing hypothesis;
it does not establish its unmeasured kinetics or replace APL physiology.
"""
from pathlib import Path
import copy, json
import numpy as np
from pnkc_receptor_brain import PnkcReceptorBrain, GpuPnkcReceptorBrain, KEYS as PARENT_KEYS
from kcgamma_regional_brain import _record_hash, _regional_currents
from session_io import sha256
from synaptic_visual_brain import _hash_array

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SELECTION=ROOT/'data/apl_mbon_system_20260910/gamma_output_selection.npz'
POLICY='canonical_kcgamma_remaining_output_routes_v1'
PARENT_STATE_SIZE=359373
KEYS=PARENT_KEYS|{'kcgamma_output_manifest'}
SELECTION_KEYS={'csr_positions','pre_rows','post_rows','pre_ids','post_ids',
    'total_contacts','gamma_contacts','calyx_contacts','other_contacts','target_rows','source_rows'}


def _selection(h,a):
    b=h.brain
    if set(a)!=SELECTION_KEYS or b.n_neurons!=166700 or len(h.state)!=PARENT_STATE_SIZE:
        raise ValueError('Output routing requires the complete continuing canonical CNS')
    if any(not isinstance(v,np.ndarray) or v.ndim!=1 or v.dtype!=np.int64 or not len(v) for v in a.values()):
        raise ValueError('Invalid canonical output selection arrays')
    pos,pre,post=(a[k] for k in ('csr_positions','pre_rows','post_rows'))
    if (any(len(a[k])!=len(pos) for k in SELECTION_KEYS-{'target_rows','source_rows'})
            or np.any(np.diff(pos)<=0) or pos[0]<0 or pos[-1]>=b.W.nnz
            or np.any(pre<0) or np.any(post<0) or np.any(pre>=b.n_neurons) or np.any(post>=b.n_neurons)
            or not np.array_equal(b.W.indices[pos],pre)
            or not np.array_equal(np.searchsorted(b.W.indptr,pos,side='right')-1,post)
            or not np.array_equal(b.node_ids[pre],a['pre_ids']) or not np.array_equal(b.node_ids[post],a['post_ids'])
            or not np.array_equal(np.unique(pre),a['source_rows']) or not np.array_equal(np.unique(post),a['target_rows'])
            or np.any(a['total_contacts']<=0) or np.any(a['gamma_contacts']<=0)
            or np.any(a['calyx_contacts']<0) or np.any(a['other_contacts']<0)
            or not np.array_equal(a['total_contacts'],a['gamma_contacts']+a['calyx_contacts']+a['other_contacts'])
            or not np.isin(pre,h.regional_rows).all() or np.isin(post,h.regional_rows).any()
            or np.any(h.visual_mask[np.r_[pre,post]]) or np.any(h.weights64[pos]<0)
            or not np.isfinite(h.weights64[pos]).all()):
        raise ValueError('Changed output identities, regional partition or stored pair mapping')
    blocked=np.unique(np.r_[h._regional_cache['rows'],h.pvlp_adaptation_manifest['target_rows'],
        h.orn_pn_synaptic_manifest['post_indices'],h._cvn7_rows,h.pnkc_receptor_manifest['target_rows']])
    if np.intersect1d(post,blocked).size:
        raise ValueError('Output routing overlaps an existing specialized transfer')


class KcGammaOutputBrain(PnkcReceptorBrain):
    SCHEMA='matrix_kcgamma_output_brain_fp64_v1'
    OUTPUT_PARENT_CLASS=PnkcReceptorBrain

    @classmethod
    def adopt(cls,reference,*,enabled=True,selection_path=DEFAULT_SELECTION):
        if type(reference) is not cls.OUTPUT_PARENT_CLASS or type(enabled) is not bool:
            raise ValueError('Output routing requires its exact PN->KC parent backend')
        path=Path(selection_path).resolve();source=json.loads((path.parent/'gamma_output_manifest.json').read_text(encoding='utf-8'))
        if source['selection_sha256']!=sha256(path):raise ValueError('Output anatomical selection checksum changed')
        with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        _selection(reference,a)
        saved=reference.state_dict();old={k:v for k,v in saved.items() if k!='schema'}
        m=dict(policy=POLICY,enabled=enabled,source_schema=reference.SCHEMA,parent_state_size=PARENT_STATE_SIZE,
            adoption_time_ns=int(reference.time_ns),initial_parent_record_sha256=_record_hash(old),**a,
            source=dict(path=str(path),manifest=source,arrays_sha256=_record_hash(a)),
            inherited_regional_manifest_sha256=_record_hash(reference.regional_manifest),
            node_ids_sha256=_hash_array(reference.brain.node_ids),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),
            anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            anatomical_weights_sha256=_hash_array(reference.brain.W.data),
            equation='release_eff=(1-f_gL)*old_filtered_soma + f_gL*persisted_filtered_axon; apply original signed pair weight and cap.',
            region_scope='Only additional mapped non-KCgamma ordinary-rate targets. Existing specialized transfers remain authoritative; unassigned and non-gamma fractions remain inherited.',
            retained_prostheses=['Gamma axonal kinetics and contact-to-current scales are inherited hypotheses, without new fitting.',
                'Anatomical ROI does not identify receptor subtype or prove functional strength.',
                'APL/KC/MBON rate transfer, local APL release and complete neuromodulation remain incomplete.'],
            new_canonical_neurons=0,new_anatomical_pairs=0,new_dynamic_states=0,new_parameters=0,
            parameter_fitting=False,biological_validation=False,learning_demonstrated=False)
        m['record_sha256']=_record_hash(m)
        saved.update(schema=cls.SCHEMA,kcgamma_output_manifest=m)
        return cls.from_state(reference.brain,saved)

    def _build_output_cache(self):
        m=self.kcgamma_output_manifest;b=self.brain;a={k:m[k] for k in SELECTION_KEYS};_selection(self,a)
        if (m['policy']!=POLICY or type(m['enabled']) is not bool or m['source_schema']!=self.OUTPUT_PARENT_CLASS.SCHEMA
                or m['parent_state_size']!=PARENT_STATE_SIZE or m['record_sha256']!=_record_hash(m)
                or m['source']['arrays_sha256']!=_record_hash(a)
                or m['inherited_regional_manifest_sha256']!=_record_hash(self.regional_manifest)
                or type(m['adoption_time_ns']) is not int or not 0<=m['adoption_time_ns']<=self.time_ns
                or any(m[k]!=0 for k in ('new_canonical_neurons','new_anatomical_pairs','new_dynamic_states','new_parameters'))
                or any(m[k] is not False for k in ('parameter_fitting','biological_validation','learning_demonstrated'))):
            raise ValueError('Changed gamma output history or hypothesis')
        for key,value in [('node_ids',b.node_ids),('anatomical_indptr',b.W.indptr),
                          ('anatomical_indices',b.W.indices),('anatomical_weights',b.W.data)]:
            if m[key+'_sha256']!=_hash_array(value):raise ValueError('Changed gamma output graph: '+key)
        if self.time_ns==m['adoption_time_ns']:
            old={k:v for k,v in self.state_dict().items() if k not in ('schema','kcgamma_output_manifest')}
            if _record_hash(old)!=m['initial_parent_record_sha256']:raise ValueError('Output adoption changed inherited state')
        rows=a['target_rows'];ptr=np.r_[0,np.cumsum(b.W.indptr[rows+1]-b.W.indptr[rows])].astype(np.int64)
        c=dict(rows=rows.copy(),local_ptr=ptr,mode=np.zeros(ptr[-1],dtype=np.int8),
            fraction=np.zeros(ptr[-1]),slot=np.full(ptr[-1],-1,dtype=np.int32))
        local=ptr[np.searchsorted(rows,a['post_rows'])]+a['csr_positions']-b.W.indptr[a['post_rows']]
        c['mode'][local]=2;c['fraction'][local]=a['gamma_contacts']/a['total_contacts']
        c['slot'][local]=np.searchsorted(self.regional_rows,a['pre_rows']).astype(np.int32)
        self._output_cache=c

    def _coefficients(self,state,drive,light):
        target,rate=PnkcReceptorBrain._coefficients(self,state,drive,light)
        if not (self.kcgamma_output_manifest['enabled'] and self.regional_manifest['enabled']):return target,rate
        c=self._output_cache;sax=state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        _regional_currents(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],
            self.brain.W.indptr,self.brain.W.indices,self.weights64,
            state[self.transmission_start:self.inherited_state_size],sax,self.caps,self.visual_mask,
            self.rate_gain,self.rate_theta,drive,self.visual_output_connected,target)
        return target,rate

    def state_dict(self):
        saved=super().state_dict();saved['kcgamma_output_manifest']=copy.deepcopy(self.kcgamma_output_manifest);return saved

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved.get('schema')!=cls.SCHEMA:raise ValueError('Incomplete gamma output checkpoint')
        parent={k:v for k,v in saved.items() if k!='kcgamma_output_manifest'};parent['schema']=cls.OUTPUT_PARENT_CLASS.SCHEMA
        base=cls.OUTPUT_PARENT_CLASS.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.kcgamma_output_manifest=copy.deepcopy(saved['kcgamma_output_manifest']);obj._build_output_cache();return obj


class GpuKcGammaOutputBrain(KcGammaOutputBrain,GpuPnkcReceptorBrain):
    SCHEMA='matrix_kcgamma_output_brain_fp64_cuda_v1'
    OUTPUT_PARENT_CLASS=GpuPnkcReceptorBrain

    def _build_output_cache(self):
        super()._build_output_cache()
        import cupy as cp
        self._output_cuda={k:cp.asarray(v) for k,v in self._output_cache.items()}

    def coefficients_gpu(self,state,drive,light):
        target,rate=GpuPnkcReceptorBrain.coefficients_gpu(self,state,drive,light)
        if not (self.kcgamma_output_manifest['enabled'] and self.regional_manifest['enabled']):return target,rate
        r=self._output_cuda;c=self.cuda;n=len(self._output_cache['rows'])
        sax=state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        args=(np.int32(n),r['rows'],r['local_ptr'],r['mode'],r['fraction'],r['slot'],c['indptr'],c['indices'],
            c['weights'],state[self.transmission_start:self.inherited_state_size],sax,c['caps'],c['visual'],
            c['gain'],c['theta'],drive,np.bool_(self.visual_output_connected),target)
        self._regional_current_kernel(((n*32+255)//256,),(256,),args)
        return target,rate

    @staticmethod
    def backend_identity():
        out=GpuPnkcReceptorBrain.backend_identity()
        out['kcgamma_output']='Existing persistent gamma axonal release on additional anatomically mapped outputs; fixed parameters, original warp reduction.'
        return out
