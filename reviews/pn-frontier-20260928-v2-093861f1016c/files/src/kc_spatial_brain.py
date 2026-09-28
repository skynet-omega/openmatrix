"""Canonical KCgamma PN/APL coupling to a declared WT9 spatial reference."""
from pathlib import Path
import copy
import numpy as np
from kc_electrical_scales_brain import GpuKcElectricalScalesBrain,KEYS as PARENT_KEYS
from kc_apl_dynamic_brain import GpuKcGammaOutputBrain
from kc_apl_dynamics import lif_events,apl_step,cascade
from kc_projected_batch import ProjectedKcBatch,cell_parameters
from kc_spatial_inputs import CanonicalKcSpatialInputs
from kcgamma_regional_brain import _record_hash

ROOT=Path(__file__).resolve().parents[1]
KEYS=PARENT_KEYS|{'kc_spatial_manifest','kc_spatial_state'}


class GpuKcSpatialBrain(GpuKcElectricalScalesBrain):
    SCHEMA='matrix_kc_spatial_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls,reference,*,enabled=True,pn_enabled=True,apl_enabled=True,inner_step_ns=25000):
        if type(reference) is not GpuKcElectricalScalesBrain:raise ValueError('Require exact continuing electrical-scale parent')
        for x in [enabled,pn_enabled,apl_enabled]:
            if type(x) is not bool:raise ValueError('Explicit boolean intervention required')
        if inner_step_ns not in [12500,25000]:raise ValueError('Untested internal step')
        obj=cls.__new__(cls);obj.__dict__.update(reference.__dict__)
        inputs=CanonicalKcSpatialInputs(obj);rows=inputs.rows
        saved=reference.state_dict();parent=copy.deepcopy(saved);parent.pop('schema')
        params=cell_parameters(ROOT/'data/kc_transfer_20260910/WT9_seventeen_port_active.npz')
        batch=ProjectedKcBatch(params,obj.kc_apl_dynamic_state['voltage_mV'][inputs.kc_index],
            obj.state[rows],obj.caps[rows],obj.tau[rows])
        state=batch.state_dict();state['time_ns']=obj.time_ns
        manifest=dict(enabled=enabled,pn_enabled=pn_enabled,apl_enabled=apl_enabled,inner_step_ns=inner_step_ns,
            coupling_step_ns=125000,cell=params,inputs=inputs.manifest(),adoption_time_ns=obj.time_ns,
            parent_schema=reference.SCHEMA,parent_record_sha256=_record_hash(parent),initial_state_sha256=_record_hash(state),
            initial_gamma_counts=obj.kc_apl_dynamic_state['spike_count'][inputs.kc_index].copy(),
            initialization='Every inherited coordinate preserved. New hidden voltage ports start at source WT9 leak rest; soma coordinate matches inherited midpoint voltage. Gates start at steady state of these declared initial voltages. No older spatial history is inferred.',
            legacy_gamma_voltage='Inactive historical LIF voltage/refractory fields retained; authoritative gamma voltage and gates are kc_spatial_state. Cumulative event counts continue in both ledgers.',
            output='Causal SIZ local maxima above-40mV with past-trough prominence20mV drive inherited q filter. No voltage reset or membrane clipping. Previous regional output/release filters remain downstream proxies.',
            new_canonical_neurons=0,new_anatomical_pairs=0,biological_validation=False)
        manifest['record_sha256']=_record_hash(manifest)
        obj.kc_spatial_manifest=manifest;obj.kc_spatial_state=state;obj._spatial_inputs=inputs;obj._spatial_batch=batch
        obj._spatial_clipped_previous=0
        obj._spatial_other=np.flatnonzero(~np.isin(obj._kc_rows,rows));obj.validate_spatial()
        check=obj.state_dict();check.pop('schema');check.pop('kc_spatial_manifest');check.pop('kc_spatial_state')
        if _record_hash(check)!=manifest['parent_record_sha256']:raise ValueError('Adoption changed inherited fields')
        return obj

    def validate_spatial(self):
        m=self.kc_spatial_manifest
        if m['record_sha256']!=_record_hash(m) or self.brain.n_neurons!=166700:raise ValueError('Changed spatial policy or identity')
        if len(m['inputs']['rows'])!=1557 or not np.array_equal(m['inputs']['ids'],self.brain.node_ids[m['inputs']['rows']]):raise ValueError('Changed canonical gamma selection')
        if m['inner_step_ns'] not in [12500,25000] or m['coupling_step_ns']!=125000:raise ValueError('Invalid solver schedule')
        b=self._spatial_batch
        if m['enabled']:
            if b.elapsed_ns!=self.time_ns-m['adoption_time_ns']:raise ValueError('Spatial clock mismatch')
            state=b.state_dict()
            self.kc_spatial_state=dict(state,time_ns=self.time_ns)
            for k in ['delta','gates','q']:
                if not np.isfinite(state[k]).all():raise FloatingPointError('Nonfinite spatial state')
            if np.any((state['gates']<0)|(state['gates']>1)) or np.any((state['q']<0)|(state['q']>1)):raise ValueError('Invalid spatial gates or output')
            np.testing.assert_array_equal(state['q'],self.state[self._spatial_inputs.rows])
            np.testing.assert_array_equal(m['initial_gamma_counts']+state['counts'],self.kc_apl_dynamic_state['spike_count'][self._spatial_inputs.kc_index])

    def advance(self,dt_ns,drive,light):
        import cupy as cp
        drive,light=self._validated_inputs(dt_ns,drive,light);m=self.kc_spatial_manifest
        if not m['enabled']:
            GpuKcElectricalScalesBrain.advance(self,dt_ns,drive,light);return
        if np.any(drive[self._dynamic_cache['rows']]!=0.):raise ValueError('Direct KC/APL drive lacks physical units')
        c=self._dynamic_cache;d=self.kc_apl_dynamic_state;kc=self.kc_apl_dynamic_manifest['group']<3
        other=self._spatial_other;other_rows=self._kc_rows[other];gidx=self._spatial_inputs.kc_index
        apl=self.kc_apl_dynamic_manifest['source']['physiology']['APL'];remaining=dt_ns
        while remaining:
            ns=min(remaining,m['coupling_step_ns']);dt=ns*1e-9
            before=self.state[self.kc_electrical_scales_manifest['source_rows']].copy()
            components=self.conductance_components();ge,gi,age,agi=components[:4]
            eg,ig,regional=self._spatial_inputs.split(components,pn_enabled=m['pn_enabled'],apl_enabled=m['apl_enabled'])
            qgamma=self._spatial_batch.advance(ns,eg,ig,inner_step_ns=m['inner_step_ns'])
            spatial_clipped=int(self._spatial_batch.clipped.sum().item())
            additional_spatial_clips=spatial_clipped-self._spatial_clipped_previous
            self._spatial_clipped_previous=spatial_clipped
            gl=1./c['rin'][kc][other];total=gl+ge[kc][other]+gi[kc][other]
            vinf=(gl*c['rest'][kc][other]-68.*gi[kc][other])/total;relax=total/gl/c['tau'][kc][other]
            q=self.state[other_rows].copy();v=d['voltage_mV'][other].copy();refrac=d['refractory_left_s'][other].copy();counts=d['spike_count'][other].copy()
            clipped=lif_events(v,refrac,counts,q,vinf,relax,c['rest'][kc][other],c['threshold'][other],self.caps[other_rows],self.tau[other_rows],dt)
            av=apl_step(d['apl_voltage_mV'],age,agi,self._apl_area,apl['rest_mV'],apl['Rin_GOhm'],apl['membrane_tau_s'],self.kc_apl_dynamic_manifest['source']['coupling_ratio'],dt)
            qt=np.clip((.5*(av+d['apl_voltage_mV'])-apl['rest_mV'])/(-apl['rest_mV']),0.,1.)
            aq=np.empty_like(qt);ass=np.empty_like(qt)
            for j,row in enumerate(self._apl_rows):aq[j],ass[j]=cascade(d['apl_q'][j],d['apl_transmission'][j],qt[j],self.tau[row],self.parameters['synaptic_tau_s'],dt)
            _,edge_release=self.routes.outgoing_release(d['apl_transmission'])
            self.cuda['weights'][self._apl_gpu_positions]=self._apl_base_weights*cp.asarray(edge_release);self._edge_buffer_active=True
            try:GpuKcGammaOutputBrain.advance(self,ns,drive,light)
            finally:
                self.cuda['weights'][self._apl_gpu_positions]=self._apl_base_weights;self._edge_buffer_active=False
            self.state[other_rows]=q;self.state[self._spatial_inputs.rows]=qgamma
            d['voltage_mV'][other]=v;d['refractory_left_s'][other]=refrac;d['spike_count'][other]=counts
            d['spike_count'][gidx]=m['initial_gamma_counts']+self._spatial_batch.host(self._spatial_batch.counts)
            self.state[self._apl_rows]=(aq*self._apl_area).sum(axis=1)
            self.state[self.transmission_start+self._apl_rows]=(ass*self._apl_area).sum(axis=1)
            d.update(apl_voltage_mV=av,apl_q=aq,apl_transmission=ass,time_ns=self.time_ns,
                clipped_spike_events=d['clipped_spike_events']+int(clipped)+additional_spatial_clips,coupling_steps=d['coupling_steps']+1)
            pm=self.kc_electrical_scales_manifest;pd=self.kc_electrical_scales_state;p=pm['source']['PN_KCgamma']
            x,y=cascade(pd['filters'][0],pd['filters'][1],before,p['cascade_fast_tau_s'],p['cascade_slow_tau_s'],dt)
            pd.update(filters=np.vstack((x,y)),time_ns=self.time_ns)
            self.publish_rates();remaining-=ns
        self.validate_dynamic();self.validate_scales();self.validate_spatial()

    def state_dict(self):
        result=super().state_dict()
        result['kc_spatial_manifest']=copy.deepcopy(self.kc_spatial_manifest)
        result['kc_spatial_state']=self._spatial_batch.state_dict();result['kc_spatial_state']['time_ns']=self.time_ns
        return result

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved.get('schema')!=cls.SCHEMA:raise ValueError('Incomplete spatial brain state')
        parent={k:v for k,v in saved.items() if k not in ['kc_spatial_manifest','kc_spatial_state']};parent['schema']=GpuKcElectricalScalesBrain.SCHEMA
        base=GpuKcElectricalScalesBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.kc_spatial_manifest=copy.deepcopy(saved['kc_spatial_manifest']);obj.kc_spatial_state=copy.deepcopy(saved['kc_spatial_state'])
        inputs=CanonicalKcSpatialInputs(obj);obj._spatial_inputs=inputs;m=obj.kc_spatial_manifest
        if _record_hash(inputs.manifest())!=_record_hash(m['inputs']):raise ValueError('Changed electrical input routing')
        state=copy.deepcopy(obj.kc_spatial_state)
        if state.pop('time_ns')!=obj.time_ns:raise ValueError('Wrong spatial state clock')
        obj._spatial_batch=ProjectedKcBatch(m['cell'],obj.kc_apl_dynamic_state['voltage_mV'][inputs.kc_index],
            obj.state[inputs.rows],obj.caps[inputs.rows],obj.tau[inputs.rows],state=state)
        obj._spatial_clipped_previous=int(state['clipped'].sum())
        obj._spatial_other=np.flatnonzero(~np.isin(obj._kc_rows,inputs.rows));obj.validate_spatial();return obj

    @staticmethod
    def backend_identity():
        info=GpuKcElectricalScalesBrain.backend_identity();info['kc_spatial']='FP64 CUDA batched17-port WT9 template; native gates; explicit PN/APL domains and causal peak output;125us CNS coupling.';return info
