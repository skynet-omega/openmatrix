"""Continuing dynamic CNS with separately persisted PN and APL conductances."""
from pathlib import Path
import copy,json
import numpy as np
from kc_apl_dynamic_brain import GpuKcAplDynamicBrain,KEYS as PARENT_KEYS,RECEPTOR_START,PARENT_STATE_SIZE
from kc_apl_dynamics import cascade
from kc_electrical_scales import projection_conductances
from kcgamma_regional_brain import _record_hash
from session_io import sha256

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/kc_electrical_scales_20260910'
KEYS=PARENT_KEYS|{'kc_electrical_scales_manifest','kc_electrical_scales_state'}


class GpuKcElectricalScalesBrain(GpuKcAplDynamicBrain):
    SCHEMA='matrix_kc_electrical_scales_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls,reference,*,enabled=True):
        if type(reference) is not GpuKcAplDynamicBrain or type(enabled) is not bool:
            raise ValueError('Requires the exact saved dynamic CNS parent')
        source=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'));p=source['PN_KCgamma']
        pn=reference.pnkc_receptor_manifest;a=reference.routes.arrays;mask=a['apl_pre']&np.isin(a['post_rows'],reference._kc_rows)
        group=reference.kc_apl_dynamic_manifest['group'];rows=reference.kc_apl_dynamic_manifest['rows']
        mask &= np.isin(a['post_rows'],rows[group==2]);ix=np.flatnonzero(mask)
        apl_positions=a['csr_positions'][ix]
        if np.any(reference.weights64[apl_positions]>=0.):raise ValueError('Noninhibitory APL-gamma edge')
        np.testing.assert_allclose(reference.weights64[pn['csr_positions']],.03*pn['contacts'],rtol=1e-6,atol=1e-8)
        pn_gain=reference.weights64[pn['csr_positions']]/(.03*p['reference_contacts'])*p['conductance_event_area_nS_s']*reference.caps[pn['pre_rows']]
        apl_gain=np.abs(reference.weights64[apl_positions])*reference.caps[a['pre_rows'][ix]]*reference._dynamic_scales[2,1]
        srcrows=pn['source_rows'];b=reference.state[RECEPTOR_START:PARENT_STATE_SIZE]
        # Match the inherited PN conductance on every selected pair at adoption.
        y=(400./.375)*b*reference._dynamic_scales[2,0]*(.03*p['reference_contacts'])/(p['conductance_event_area_nS_s']*reference.caps[srcrows])
        initial=dict(filters=np.vstack((reference.state[srcrows],y)),time_ns=reference.time_ns)
        old=reference.state_dict();parent=copy.deepcopy(old);parent.pop('schema')
        m=dict(source=source,source_manifest_sha256=sha256(DATA/'manifest.json'),enabled=enabled,
            rows=rows.copy(),group=group.copy(),source_rows=srcrows.copy(),source_ids=pn['source_ids'].copy(),
            pn_positions=pn['csr_positions'].copy(),pn_gain_nS=pn_gain,apl_positions=apl_positions.copy(),apl_gbar_nS=apl_gain,
            parent_schema=reference.SCHEMA,adoption_time_ns=reference.time_ns,parent_record_sha256=_record_hash(parent),
            initial_state_sha256=_record_hash(initial),new_canonical_neurons=0,new_anatomical_pairs=0,
            initialization='PN slow filter matches previous conductance per pair; fast filter initialized from current PN rate. Older waveform history is unknown. All existing neural/body coordinates preserved.',
            APL_absolute_scale_identified=False)
        m['record_sha256']=_record_hash(m)
        old.update(schema=cls.SCHEMA,kc_electrical_scales_manifest=m,kc_electrical_scales_state=initial)
        return cls.from_state(reference.brain,old)

    def _build_scales_cache(self):
        m=self.kc_electrical_scales_manifest;src=m['source'];pn=self.pnkc_receptor_manifest;c=self._dynamic_cache;b=self.brain
        if (m['record_sha256']!=_record_hash(m) or type(m['enabled']) is not bool
                or m['parent_schema']!=GpuKcAplDynamicBrain.SCHEMA or m['APL_absolute_scale_identified'] is not False
                or src['schema']!='matrix_kc_electrical_scales_source_v1' or src['coupling_step_ns']!=125000
                or sha256(ROOT/src['primary_source_path'])!=src['primary_source_sha256']
                or not np.array_equal(m['rows'],self.kc_apl_dynamic_manifest['rows'])
                or not np.array_equal(m['group'],self.kc_apl_dynamic_manifest['group'])
                or not np.array_equal(m['source_rows'],pn['source_rows']) or not np.array_equal(m['source_ids'],pn['source_ids'])
                or not np.array_equal(m['pn_positions'],pn['csr_positions'])):
            raise ValueError('Changed projection-scale policy or source mapping')
        a=self.routes.arrays;expected=a['csr_positions'][a['apl_pre']&np.isin(a['post_rows'],m['rows'][m['group']==2])]
        if not np.array_equal(expected,m['apl_positions']):raise ValueError('Incomplete APL-gamma selection')
        self._pn_gain_local=np.zeros(c['local_ptr'][-1]);self._apl_gain_local=np.zeros(c['local_ptr'][-1])
        for positions,values,target in [(m['pn_positions'],m['pn_gain_nS'],self._pn_gain_local),
                                         (m['apl_positions'],m['apl_gbar_nS'],self._apl_gain_local)]:
            if values.shape!=positions.shape or values.dtype!=np.float64 or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError('Invalid projection conductance units')
            post=np.searchsorted(b.W.indptr,positions,side='right')-1
            local=c['local_ptr'][np.searchsorted(c['rows'],post)]+positions-b.W.indptr[post]
            target[local]=values
        self.validate_scales()
        if self.time_ns==m['adoption_time_ns']:
            saved=self.state_dict();saved.pop('schema');saved.pop('kc_electrical_scales_manifest');saved.pop('kc_electrical_scales_state')
            if _record_hash(saved)!=m['parent_record_sha256'] or _record_hash(self.kc_electrical_scales_state)!=m['initial_state_sha256']:
                raise ValueError('Scale adoption changed prior history')

    def validate_scales(self):
        d=self.kc_electrical_scales_state;m=self.kc_electrical_scales_manifest
        if (set(d)!={'filters','time_ns'} or d['time_ns']!=self.time_ns or type(d['time_ns']) is not int
                or d['filters'].dtype!=np.float64 or d['filters'].shape!=(2,len(m['source_rows']))
                or not np.isfinite(d['filters']).all() or np.any((d['filters']<0)|(d['filters']>1))):
            raise ValueError('Invalid PN conductance-filter state or clock')

    def conductance_components(self,caps=None):
        c=self._dynamic_cache;y=self.state;a=self.routes.arrays;b=self.brain
        _,edge=self.routes.outgoing_release(self.kc_apl_dynamic_state['apl_transmission'])
        sax=y[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        return projection_conductances(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],c['pn_slot'],
            c['apl_edge_slot'],b.W.indptr,b.W.indices,self.weights64,y[self.transmission_start:self.inherited_state_size],
            sax,self.caps if caps is None else caps,self.visual_mask,self.visual_output_connected,y[RECEPTOR_START:PARENT_STATE_SIZE],edge,
            self._dynamic_scales,self.kc_apl_dynamic_manifest['group'],c['pair_route_slot'],a['region_fractions'],c['route_apl'],
            self.routes.shape[1],self._pn_gain_local,self.kc_electrical_scales_state['filters'][1],self._apl_gain_local)

    def conductances(self):
        if not self.kc_electrical_scales_manifest['enabled']:return GpuKcAplDynamicBrain.conductances(self)
        return self.conductance_components()[:4]

    def advance(self,dt_ns,drive,light):
        drive,light=self._validated_inputs(dt_ns,drive,light)
        m=self.kc_electrical_scales_manifest;d=self.kc_electrical_scales_state;p=m['source']['PN_KCgamma']
        # The same125us coupling schedule is used in the bypass comparator.
        remaining=dt_ns
        while remaining:
            ns=min(remaining,m['source']['coupling_step_ns']);before=self.state[m['source_rows']].copy()
            GpuKcAplDynamicBrain.advance(self,ns,drive,light)
            if m['enabled']:
                x,y=cascade(d['filters'][0],d['filters'][1],before,p['cascade_fast_tau_s'],p['cascade_slow_tau_s'],ns*1e-9)
                d['filters']=np.vstack((x,y))
            d['time_ns']=self.time_ns;remaining-=ns
        self.validate_scales()

    def state_dict(self):
        saved=super().state_dict()
        saved['kc_electrical_scales_manifest']=copy.deepcopy(self.kc_electrical_scales_manifest)
        saved['kc_electrical_scales_state']=copy.deepcopy(self.kc_electrical_scales_state)
        return saved

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved.get('schema')!=cls.SCHEMA:raise ValueError('Incomplete projection-scale state')
        parent={k:v for k,v in saved.items() if k not in ('kc_electrical_scales_manifest','kc_electrical_scales_state')}
        parent['schema']=GpuKcAplDynamicBrain.SCHEMA
        base=GpuKcAplDynamicBrain.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.kc_electrical_scales_manifest=copy.deepcopy(saved['kc_electrical_scales_manifest'])
        obj.kc_electrical_scales_state=copy.deepcopy(saved['kc_electrical_scales_state'])
        obj._build_scales_cache();return obj

    @staticmethod
    def backend_identity():
        info=GpuKcAplDynamicBrain.backend_identity()
        info['kc_electrical_scales']='PN-gamma somatic-equivalent quantal conductance filters; explicit APL-gamma gbar in nS, unresolved absolute calibration;125us coupling.'
        return info
