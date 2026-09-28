"""Deliver canonical VPN/LVIN input to KCgamma-d dendrites without new gain."""
import copy
import numpy as np
from kc_axonal_brain import GpuKcAxonalBrain,KEYS as PARENT_KEYS
from canonical_pathway_probe import CanonicalPathways
from kc_visual_ports import VisualComponents,VisualKcInputs
from kc_axonal_release import AxonalSamplingBatch
from kcgamma_regional_brain import _record_hash

KEYS=PARENT_KEYS|{'kc_visual_manifest'}


class GpuKcVisualBrain(GpuKcAxonalBrain):
    SCHEMA='matrix_kc_visual_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls,parent,*,enabled=True):
        if type(parent) is not GpuKcAxonalBrain or type(enabled) is not bool:raise ValueError('Require saved axonal parent')
        obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        before=parent.state_dict();before.pop('schema')
        obj._visual_pathways=CanonicalPathways(obj)
        names=('VPN_to_KCgd','LVIN_to_KCgd');groups=[obj._visual_pathways.groups[n] for n in names]
        positions=np.concatenate([g['positions'] for g in groups])
        if len(np.unique(positions))!=len(positions):raise ValueError('Overlapping visual routes')
        rows=np.unique(np.concatenate([g['post'] for g in groups]))
        if not np.isin(rows,obj._spatial_inputs.rows).all():raise ValueError('Visual target lacks spatial model')
        m=dict(enabled=enabled,adoption_time_ns=obj.time_ns,parent_record_sha256=_record_hash(before),
            groups=names,rows=rows,ids=obj.brain.node_ids[rows],pairs=len(positions),
            source='Ganguly et al.2024 doi:10.1038/s41467-024-49616-z, Figure1 and PDFpage3; canonical route identity from saved route manifest.',
            route_manifest=obj._visual_pathways.manifest,
            routing='Move the existing VPN/LVIN->KCgamma-d ge/gi from soma proxy to the area-uniform WT9 dendritic port. Preserve totals, signs, canonical pairs, and all PN/APL terms.',
            limits='Calyx compartment supported anatomically; uniform distribution and shared WT9 template are hypotheses. No individual visual claw coordinates or paired electrical calibration. Alpha/beta-p remains its inherited LIF interface.',
            conductance_fitting=False,individual_synapse_locations_identified=False,biological_validation=False)
        m['record_sha256']=_record_hash(m);obj.kc_visual_manifest=m
        obj._install_visual_inputs();obj.validate_visual_ports()
        after=obj.state_dict();after.pop('schema');after.pop('kc_visual_manifest')
        if _record_hash(after)!=m['parent_record_sha256']:raise ValueError('Visual adoption altered inherited state')
        return obj

    def _install_visual_inputs(self):
        # Adoption/restoration copies runtime attributes, including closures.
        # The inherited sampler must read THIS continuing object's state: its
        # neural state array is replaced by the CUDA integrator on each step.
        if isinstance(self._spatial_batch,AxonalSamplingBatch):
            self._spatial_batch=self._spatial_batch.base
        self._install_axonal_batch()
        self._visual_pathways=CanonicalPathways(self)
        self._visual_gamma_mask=np.isin(self._dynamic_cache['rows'],self._spatial_inputs.rows)
        self._spatial_inputs=VisualKcInputs(self._spatial_inputs,self.kc_visual_manifest['enabled'])

    def conductance_components(self):
        values=super().conductance_components();ge=np.zeros_like(values[0]);gi=ge.copy()
        for name in self.kc_visual_manifest['groups']:
            a,b=self._visual_pathways.visual_conductances(name);ge+=a;gi+=b
        return VisualComponents(values,ge,gi,self._visual_gamma_mask)

    def validate_visual_ports(self):
        m=self.kc_visual_manifest
        if m['record_sha256']!=_record_hash(m) or type(m['enabled']) is not bool:raise ValueError('Changed visual-port policy')
        np.testing.assert_array_equal(m['ids'],self.brain.node_ids[m['rows']])

    def state_dict(self):
        out=super().state_dict();out['kc_visual_manifest']=copy.deepcopy(self.kc_visual_manifest);return out

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete visual-port brain')
        parent={k:v for k,v in saved.items() if k!='kc_visual_manifest'};parent['schema']=GpuKcAxonalBrain.SCHEMA
        base=GpuKcAxonalBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.kc_visual_manifest=copy.deepcopy(saved['kc_visual_manifest']);obj._install_visual_inputs();obj.validate_visual_ports();return obj

    @staticmethod
    def backend_identity():
        out=GpuKcAxonalBrain.backend_identity();out['kc_visual_ports']='Canonical VPN/LVIN gamma-d conductances relocated to dendritic port; total ge/gi conserved.';return out
