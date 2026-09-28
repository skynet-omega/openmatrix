"""Persistent event-KC and coupled graded-APL candidate in the complete CNS.

Operator splitting holds the selected outputs for at most 250 us while the
unchanged adaptive CNS solver advances. KC threshold crossings are exact for
the held afferents. The splitting error is distinct from the CNS solver error.
Regional APL release is evaluated on its actual outgoing CSR edges. A temporary
device coefficient buffer represents weight*release; archived weights never
change. This backend intentionally requires CUDA, as does the active organism.
"""
from pathlib import Path
import copy, json
import numpy as np
from kcgamma_output_brain import GpuKcGammaOutputBrain, KEYS as PARENT_KEYS
from kcgamma_regional_brain import _record_hash
from synaptic_visual_brain import _hash_array
from session_io import sha256
from apl_region_routes import AplRegionRoutes
from kc_apl_dynamics import lif_events, cascade, apl_step, afferent_conductances

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/kc_apl_dynamic_20260910'
PARENT_STATE_SIZE = 359373
RECEPTOR_START = 359074
KEYS = PARENT_KEYS | {'kc_apl_dynamic_manifest', 'kc_apl_dynamic_state'}


class GpuKcAplDynamicBrain(GpuKcGammaOutputBrain):
    SCHEMA = 'matrix_kc_apl_dynamic_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls, reference, *, enabled=True):
        if type(reference) is not GpuKcGammaOutputBrain or type(enabled) is not bool:
            raise ValueError('Requires exact continuing CUDA gamma-output parent')
        source = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        with np.load(ROOT/source['physiology']['selection_path'], allow_pickle=False) as z:
            a = {k:z[k].copy() for k in z.files}
        saved = reference.state_dict(); parent = copy.deepcopy(saved); parent.pop('schema')
        m = dict(source=source, source_manifest_sha256=sha256(DATA/'manifest.json'),
                 enabled=enabled, rows=a['rows'], ids=a['ids'], group=a['group'],
                 adoption_time_ns=reference.time_ns, parent_record_sha256=_record_hash(parent),
                 graph_hashes=[_hash_array(x) for x in (reference.brain.node_ids,
                     reference.brain.W.indptr, reference.brain.W.indices, reference.brain.W.data)],
                 parent_schema=reference.SCHEMA, new_canonical_neurons=0, new_anatomical_pairs=0)
        kc = a['group'] < 3; ar = a['rows'][~kc]; groups = source['physiology']['groups']
        d = dict(voltage_mV=np.array([groups[g]['rest_mV'] for g in a['group'][kc]]),
                 refractory_left_s=np.zeros(kc.sum()), spike_count=np.zeros(kc.sum(), dtype=np.int64),
                 apl_voltage_mV=np.full((2,15), source['physiology']['APL']['rest_mV']),
                 apl_q=np.repeat(reference.state[ar,None],15,axis=1),
                 apl_transmission=np.repeat(reference.state[reference.transmission_start+ar,None],15,axis=1),
                 clipped_spike_events=0, coupling_steps=0, time_ns=reference.time_ns)
        m['initial_dynamic_sha256'] = _record_hash(d); m['record_sha256'] = _record_hash(m)
        saved.update(schema=cls.SCHEMA, kc_apl_dynamic_manifest=m, kc_apl_dynamic_state=d)
        return cls.from_state(reference.brain, saved)

    def _build_dynamic_cache(self):
        import cupy as cp
        m = self.kc_apl_dynamic_manifest; b = self.brain
        if (m['record_sha256'] != _record_hash(m) or type(m['enabled']) is not bool
                or m['parent_schema'] != GpuKcGammaOutputBrain.SCHEMA
                or m['graph_hashes'] != [_hash_array(x) for x in (b.node_ids,b.W.indptr,b.W.indices,b.W.data)]
                or b.n_neurons != 166700 or len(self.state) != PARENT_STATE_SIZE):
            raise ValueError('Changed dynamic policy or canonical parent')
        src = m['source']; physiology = src['physiology']; rows = m['rows']; group = m['group']
        scales = np.asarray(src['conductance_scale_nS_per_weight_Hz'])
        if (src['schema']!='matrix_kc_apl_dynamic_source_v1' or scales.shape!=(4,2)
                or not np.isfinite(scales).all() or np.any(scales<=0)
                or type(src['coupling_step_ns']) is not int or not 100<=src['coupling_step_ns']<=250000
                or not np.isfinite(src['coupling_ratio']) or src['coupling_ratio']<0
                or sha256(ROOT/physiology['selection_path'])!=physiology['selection_sha256']):
            raise ValueError('Invalid dynamic units, time step or physiological selection')
        types = self.measured_t4_manifest['canonical_node_types']
        expected = np.sort(np.concatenate([np.flatnonzero(np.char.startswith(types,p))
                                         for p in ('KCab', "KCa'b'", 'KCg', 'APL')]))
        if not np.array_equal(rows,expected) or not np.array_equal(m['ids'],b.node_ids[rows]):
            raise ValueError('Dynamic subtype selection mismatch')
        for g,prefix in enumerate(('KCab', "KCa'b'", 'KCg', 'APL')):
            if not np.array_equal(rows[group==g],np.flatnonzero(np.char.startswith(types,prefix))):
                raise ValueError('Changed subtype groups')
        self.routes = AplRegionRoutes.load(ROOT/src['routes_path'], node_ids=b.node_ids,
                                          indptr=b.W.indptr, indices=b.W.indices)
        if sha256(ROOT/src['routes_path']/'manifest.json') != src['routes_manifest_sha256']:
            raise ValueError('APL regional route source changed')
        a = self.routes.arrays; self._apl_rows = rows[group==3]; self._kc_rows = rows[group<3]
        if not np.array_equal(a['apl_rows'],self._apl_rows): raise ValueError('APL ordering changed')
        ptr = np.r_[0,np.cumsum(b.W.indptr[rows+1]-b.W.indptr[rows])].astype(np.int64)
        c = dict(rows=rows, local_ptr=ptr, mode=np.zeros(ptr[-1],dtype=np.int8),
                 fraction=np.zeros(ptr[-1]), slot=np.full(ptr[-1],-1,dtype=np.int32),
                 pn_slot=np.full(ptr[-1],-1,dtype=np.int32),
                 apl_edge_slot=np.full(ptr[-1],-1,dtype=np.int32),
                 pair_route_slot=np.full(ptr[-1],-1,dtype=np.int32))
        for parent in (self._regional_cache,self._output_cache):
            for j,row in enumerate(rows):
                k = np.searchsorted(parent['rows'],row)
                if k < len(parent['rows']) and parent['rows'][k] == row:
                    for field in ('mode','fraction','slot'):
                        c[field][ptr[j]:ptr[j+1]] = parent[field][parent['local_ptr'][k]:parent['local_ptr'][k+1]]
        pn = self.pnkc_receptor_manifest
        local = ptr[np.searchsorted(rows,pn['post_rows'])]+pn['csr_positions']-b.W.indptr[pn['post_rows']]
        c['pn_slot'][local] = np.searchsorted(pn['source_rows'],pn['pre_rows']).astype(np.int32)
        outgoing_slot = np.full(len(a['csr_positions']),-1,dtype=np.int32)
        outgoing_slot[self.routes.outgoing] = np.arange(len(self.routes.outgoing))
        eligible = np.isin(a['post_rows'],rows); ix = np.flatnonzero(eligible)
        local = ptr[np.searchsorted(rows,a['post_rows'][ix])]+a['csr_positions'][ix]-b.W.indptr[a['post_rows'][ix]]
        c['apl_edge_slot'][local] = outgoing_slot[ix]; c['pair_route_slot'][local] = ix
        c['route_apl'] = np.searchsorted(self._apl_rows,a['post_rows']).astype(np.int64)
        for row in self._apl_rows:
            j = np.searchsorted(rows,row)
            if np.any(c['pair_route_slot'][ptr[j]:ptr[j+1]]<0): raise ValueError('Unrouted APL afferent')
        lookup = physiology['groups']+[physiology['APL']]
        for target,key in [('rest','rest_mV'),('rin','Rin_GOhm'),('tau','membrane_tau_s')]:
            c[target] = np.array([g[key] for g in lookup])[group]
        c['threshold'] = np.array([g['threshold_mV'] for g in physiology['groups']])[group[group<3]]
        area = np.zeros(self.routes.shape)
        np.add.at(area,self.routes.incoming_apl,a['roi_counts'][self.routes.incoming])
        np.add.at(area,self.routes.outgoing_apl,a['roi_counts'][self.routes.outgoing])
        self._apl_area = area/area.sum(axis=1)[:,None]
        self._dynamic_cache = c; self._dynamic_scales = np.array(src['conductance_scale_nS_per_weight_Hz'])
        self._apl_positions = a['csr_positions'][self.routes.outgoing].copy()
        self._apl_gpu_positions = cp.asarray(self._apl_positions)
        self._apl_gpu_rows = cp.asarray(self._apl_rows); self._dynamic_gpu_rows = cp.asarray(rows)
        self._apl_base_weights = cp.asarray(self.weights64[self._apl_positions])
        self._edge_buffer_active = False
        self.validate_dynamic()
        if self.time_ns == m['adoption_time_ns']:
            old = self.state_dict(); old.pop('schema'); old.pop('kc_apl_dynamic_manifest'); old.pop('kc_apl_dynamic_state')
            if _record_hash(old)!=m['parent_record_sha256'] or _record_hash(self.kc_apl_dynamic_state)!=m['initial_dynamic_sha256']:
                raise ValueError('Adoption changed inherited state or invented dynamic history')

    def validate_dynamic(self):
        d = self.kc_apl_dynamic_state; n = len(self._kc_rows)
        shapes = dict(voltage_mV=(n,),refractory_left_s=(n,),spike_count=(n,),
                      apl_voltage_mV=(2,15),apl_q=(2,15),apl_transmission=(2,15))
        if set(d) != set(shapes)|{'clipped_spike_events','coupling_steps','time_ns'}:
            raise ValueError('Incomplete dynamic state')
        for k,shape in shapes.items():
            if (not isinstance(d[k],np.ndarray) or d[k].shape!=shape or not np.isfinite(d[k]).all()
                    or (k!='spike_count' and d[k].dtype!=np.float64)):
                raise ValueError('Invalid dynamic field: '+k)
        if (d['time_ns']!=self.time_ns or d['spike_count'].dtype!=np.int64 or np.any(d['spike_count']<0)
                or any(type(d[k]) is not int or d[k]<0 for k in ('time_ns','clipped_spike_events','coupling_steps'))
                or np.any(d['voltage_mV'] < -80.) or np.any(d['voltage_mV']>self._dynamic_cache['threshold']+1e-10)
                or np.any((d['refractory_left_s']<0)|(d['refractory_left_s']>.0022+1e-12))
                or np.any((d['apl_voltage_mV'] < -80.)|(d['apl_voltage_mV']>0.))
                or any(np.any((d[k]<0)|(d[k]>1)) for k in ('apl_q','apl_transmission'))):
            raise ValueError('Dynamic state domain or clock mismatch')

    def conductances(self):
        c = self._dynamic_cache; y = self.state; a = self.routes.arrays; b = self.brain
        _,edge_release = self.routes.outgoing_release(self.kc_apl_dynamic_state['apl_transmission'])
        sax = y[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        return afferent_conductances(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],c['pn_slot'],
            c['apl_edge_slot'],b.W.indptr,b.W.indices,self.weights64,y[self.transmission_start:self.inherited_state_size],
            sax,self.caps,self.visual_mask,self.visual_output_connected,y[RECEPTOR_START:PARENT_STATE_SIZE],edge_release,
            self._dynamic_scales,self.kc_apl_dynamic_manifest['group'],c['pair_route_slot'],a['region_fractions'],c['route_apl'],self.routes.shape[1])

    def coefficients_gpu(self, state, drive, light):
        if not self.kc_apl_dynamic_manifest['enabled']:
            return GpuKcGammaOutputBrain.coefficients_gpu(self,state,drive,light)
        if not self._edge_buffer_active:
            raise RuntimeError('Regional release buffer must be installed during coefficient evaluation')
        view = state.copy(); view[self.transmission_start+self._apl_gpu_rows] = 1.
        target,rate = GpuKcGammaOutputBrain.coefficients_gpu(self,view,drive,light)
        target[self._dynamic_gpu_rows] = state[self._dynamic_gpu_rows]; rate[self._dynamic_gpu_rows] = 0.
        # These scalar histories are output summaries of the regional histories.
        target[self.transmission_start+self._apl_gpu_rows] = state[self.transmission_start+self._apl_gpu_rows]
        rate[self.transmission_start+self._apl_gpu_rows] = 0.
        return target,rate

    def advance(self, dt_ns, drive, light):
        import cupy as cp
        drive,light = self._validated_inputs(dt_ns,drive,light)
        if np.any(drive[self._dynamic_cache['rows']]!=0.):
            raise ValueError('Direct KC/APL drive has no declared pA conversion')
        m = self.kc_apl_dynamic_manifest; d = self.kc_apl_dynamic_state
        if not m['enabled']:
            GpuKcGammaOutputBrain.advance(self,dt_ns,drive,light); d['time_ns']=self.time_ns; return
        remaining = dt_ns; c = self._dynamic_cache; kc = m['group']<3
        apl = m['source']['physiology']['APL']
        while remaining:
            ns = min(remaining,m['source']['coupling_step_ns']); dt = ns*1e-9
            ge,gi,age,agi = self.conductances()
            gl = 1./c['rin'][kc]; total = gl+ge[kc]+gi[kc]
            vinf = (gl*c['rest'][kc]-68.*gi[kc])/total
            relax = total/gl/c['tau'][kc]
            q = self.state[self._kc_rows].copy()
            clipped = lif_events(d['voltage_mV'],d['refractory_left_s'],d['spike_count'],q,
                vinf,relax,c['rest'][kc],c['threshold'],self.caps[self._kc_rows],self.tau[self._kc_rows],dt)
            av = apl_step(d['apl_voltage_mV'],age,agi,self._apl_area,apl['rest_mV'],apl['Rin_GOhm'],
                          apl['membrane_tau_s'],m['source']['coupling_ratio'],dt)
            # Midpoint voltage gives the held target of the graded release filters.
            vmean = .5*(av+d['apl_voltage_mV'])
            qt = np.clip((vmean-apl['rest_mV'])/(-apl['rest_mV']),0.,1.)
            aq = np.empty_like(qt); ass = np.empty_like(qt)
            for j,row in enumerate(self._apl_rows):
                aq[j],ass[j] = cascade(d['apl_q'][j],d['apl_transmission'][j],qt[j],
                                      self.tau[row],self.parameters['synaptic_tau_s'],dt)
            _,edge_release = self.routes.outgoing_release(d['apl_transmission'])
            self.cuda['weights'][self._apl_gpu_positions] = self._apl_base_weights*cp.asarray(edge_release)
            self._edge_buffer_active = True
            try:
                GpuKcGammaOutputBrain.advance(self,ns,drive,light)
            finally:
                self.cuda['weights'][self._apl_gpu_positions] = self._apl_base_weights
                self._edge_buffer_active = False
            self.state[self._kc_rows] = q
            self.state[self._apl_rows] = (aq*self._apl_area).sum(axis=1)
            self.state[self.transmission_start+self._apl_rows] = (ass*self._apl_area).sum(axis=1)
            d.update(apl_voltage_mV=av,apl_q=aq,apl_transmission=ass,time_ns=self.time_ns,
                     clipped_spike_events=d['clipped_spike_events']+int(clipped),coupling_steps=d['coupling_steps']+1)
            self.publish_rates(); remaining -= ns
        self.validate_dynamic()

    def state_dict(self):
        saved = super().state_dict()
        saved['kc_apl_dynamic_manifest'] = copy.deepcopy(self.kc_apl_dynamic_manifest)
        saved['kc_apl_dynamic_state'] = copy.deepcopy(self.kc_apl_dynamic_state)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved)!=KEYS or saved.get('schema')!=cls.SCHEMA: raise ValueError('Incomplete dynamic brain')
        parent = {k:v for k,v in saved.items() if k not in ('kc_apl_dynamic_manifest','kc_apl_dynamic_state')}
        parent['schema'] = GpuKcGammaOutputBrain.SCHEMA
        base = GpuKcGammaOutputBrain.from_state(brain,parent)
        obj = cls.__new__(cls); obj.__dict__.update(base.__dict__)
        obj.kc_apl_dynamic_manifest = copy.deepcopy(saved['kc_apl_dynamic_manifest'])
        obj.kc_apl_dynamic_state = copy.deepcopy(saved['kc_apl_dynamic_state'])
        obj._build_dynamic_cache(); return obj

    @staticmethod
    def backend_identity():
        info = GpuKcGammaOutputBrain.backend_identity()
        info['kc_apl_dynamic'] = 'Event LIF KC, conservative passive regional APL, 250us operator splitting, CPU FP64 local dynamics and inherited CUDA whole CNS; AHP absent.'
        return info
