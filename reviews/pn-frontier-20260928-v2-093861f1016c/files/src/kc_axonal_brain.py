"""Wire WT9 axonal-port events to existing canonical gamma-lobe outputs."""
import copy
import numpy as np
from kc_spatial_brain import GpuKcSpatialBrain,KEYS as PARENT_KEYS,ROOT
from kc_axonal_release import AxonalRelease,AxonalSamplingBatch
from kcgamma_regional_brain import _record_hash

KEYS=PARENT_KEYS|{'kc_axonal_manifest','kc_axonal_state'}


class GpuKcAxonalBrain(GpuKcSpatialBrain):
    SCHEMA='matrix_kc_axonal_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls,reference,*,enabled=True):
        if type(reference) is not GpuKcSpatialBrain or type(enabled) is not bool:
            raise ValueError('Requires saved spatial parent and explicit activation')
        obj=cls.__new__(cls);obj.__dict__.update(reference.__dict__)
        previous=reference.state_dict();previous.pop('schema')
        with np.load(ROOT/'data/kc_transfer_20260910/WT9_seventeen_port_active.npz',allow_pickle=False) as z:
            cap=z['full_capacitance_nF'];mask=z['port_weights'][:,5:]>0
            weights=cap@mask;weights/=weights.sum()
        rows=obj._spatial_inputs.rows
        np.testing.assert_array_equal(rows,obj.regional_rows)
        x,b,s=obj.state[obj.regional_parent_state_size:obj.retinal_parent_state_size].reshape(3,-1)
        base=obj._spatial_batch
        pub=AxonalRelease(base.host(base.delta[:,5:])+base.rest,x,s,obj.caps[rows],obj.tau[rows],
            obj.parameters['synaptic_tau_s'],weights)
        m=dict(enabled=enabled,adoption_time_ns=obj.time_ns,parent_schema=reference.SCHEMA,
            parent_record_sha256=_record_hash(previous),rows=rows.copy(),ids=obj.brain.node_ids[rows].copy(),
            weights=weights,source_artifact=base.parameters['artifact_sha256'],
            event_criterion='Past local maximum above -40mV with prominence >=20mV; same declared criterion as spatial SIZ ledger; no voltage reset.',
            release_equation='Local event q decays with inherited neuronal tau; each event adds 1/(cap*tau). Exact synaptic cascade of q/(1+eta*b) uses inherited synaptic tau. Events enter at the right endpoint.',
            routing='Area-weighted twelve-band axonal release replaces x/s_ax summaries only. Existing gamma-lobe contact fractions and other synapses are preserved. WT9 bands are not assigned to canonical gamma1-5.',
            initialization='Local q/s begin at inherited regional x/s_ax in every band; peak detectors begin at saved axonal voltages with no invented preceding slope or spikes.',
            calcium_and_vesicles_identified=False,absolute_output_strength_identified=False,
            new_canonical_neurons=0,new_anatomical_pairs=0,parameter_fitting=False,biological_validation=False)
        m['record_sha256']=_record_hash(m);obj.kc_axonal_manifest=m;obj._axonal_release=pub
        obj._install_axonal_batch();obj.validate_axonal()
        saved=obj.state_dict();saved.pop('schema');saved.pop('kc_axonal_manifest');saved.pop('kc_axonal_state')
        if _record_hash(saved)!=m['parent_record_sha256']:raise ValueError('Axonal adoption altered parent state')
        return obj

    def _install_axonal_batch(self):
        m=self.kc_axonal_manifest
        np.testing.assert_array_equal(m['rows'],self.regional_rows)
        if m['enabled']:
            def gain():
                b=self.state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[1]
                return 1./(1.+self.regional_manifest['parameters']['eta']*b)
            self._spatial_batch=AxonalSamplingBatch(self._spatial_batch,self._axonal_release,gain)

    def coefficients_gpu(self,state,drive,light):
        target,rate=super().coefficients_gpu(state,drive,light)
        if self.kc_axonal_manifest['enabled']:
            start=self.regional_parent_state_size;n=len(self.regional_rows)
            # x and s now follow actual sampled axonal events. Receptor b keeps
            # its declared inherited kinetics and reads the held x/s summaries.
            for sl in [slice(start,start+n),slice(start+2*n,start+3*n)]:
                target[sl]=state[sl];rate[sl]=0.
        return target,rate

    def advance(self,dt_ns,drive,light):
        if not self.kc_axonal_manifest['enabled']:
            super().advance(dt_ns,drive,light);return
        remaining=dt_ns
        while remaining:
            ns=min(remaining,125000)
            super().advance(ns,drive,light)
            x,s=self._axonal_release.summaries();start=self.regional_parent_state_size;n=len(x)
            self.state[start:start+n]=x;self.state[start+2*n:start+3*n]=s
            remaining-=ns
        self.validate_axonal()

    def validate_axonal(self):
        m=self.kc_axonal_manifest;self._axonal_release.validate()
        if m['record_sha256']!=_record_hash(m) or type(m['enabled']) is not bool:raise ValueError('Changed axonal policy')
        np.testing.assert_array_equal(m['ids'],self.brain.node_ids[m['rows']])
        if m['enabled'] and self._axonal_release.state['elapsed_ns']!=self.time_ns-m['adoption_time_ns']:
            raise ValueError('Axonal/CNS clocks diverged')

    def state_dict(self):
        saved=super().state_dict();saved['kc_axonal_manifest']=copy.deepcopy(self.kc_axonal_manifest)
        saved['kc_axonal_state']=self._axonal_release.state_dict();return saved

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete axonal brain state')
        parent={k:v for k,v in saved.items() if k not in ['kc_axonal_manifest','kc_axonal_state']};parent['schema']=GpuKcSpatialBrain.SCHEMA
        base=GpuKcSpatialBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.kc_axonal_manifest=copy.deepcopy(saved['kc_axonal_manifest']);m=obj.kc_axonal_manifest;rows=m['rows']
        x,b,s=obj.state[obj.regional_parent_state_size:obj.retinal_parent_state_size].reshape(3,-1)
        batch=obj._spatial_batch
        obj._axonal_release=AxonalRelease(batch.host(batch.delta[:,5:])+batch.rest,x,s,obj.caps[rows],obj.tau[rows],
            obj.parameters['synaptic_tau_s'],m['weights'],state=saved['kc_axonal_state'])
        obj._install_axonal_batch();obj.validate_axonal();return obj

    @staticmethod
    def backend_identity():
        info=GpuKcSpatialBrain.backend_identity();info['kc_axonal_release']='Twelve WT9 axonal-port event ledgers; inherited normalized filtering; area-weighted gamma-lobe release proxy;125us causal coupling.';return info
