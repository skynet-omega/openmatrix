"""Restore canonical ORN activity to the continuing ORN->PN rate synapse.

Only the input of the four inherited resource/filter coordinates changes.
Their stored values, tails, graph weights, PN membrane proxy and all KC/APL
mechanisms are retained. This does not identify native PN cable geometry,
per-contact release, odor transduction or absolute synaptic conductances.
"""
import copy
import numpy as np
from kc_visual_brain import GpuKcVisualBrain, KEYS as PARENT_KEYS
from kc_visual_ports import VisualKcInputs
from kcgamma_regional_brain import _record_hash
from olfactory_synapse_candidate import FAST, SLOW

KEYS=PARENT_KEYS|{'olfactory_endogenous_manifest'}


def endogenous_coefficients(state, rows, caps, begin, end, xp=np):
    """Same population kinetics, with q_ORN*rmax sampled at each ODE stage.

No mean across identities, external rate, additional filter, reset or RNG.
The source-dependent resource history already exists in the parent state.
"""
    af, ass, _, _=state[begin:end].reshape(4,-1)
    q=state[rows]
    hz=q*caps
    rf=1./FAST.tau_A_s+FAST.r_per_spike*hz
    rs=1./SLOW.tau_A_s+SLOW.r_per_spike*hz
    return (xp.concatenate(((1./FAST.tau_A_s)/rf,(1./SLOW.tau_A_s)/rs,q*af,q*ass)),
            xp.concatenate((rf,rs,xp.full(len(q),1./FAST.tau_g_s,dtype=xp.float64),
                            xp.full(len(q),1./SLOW.tau_g_s,dtype=xp.float64))))


class GpuOlfactoryEndogenousBrain(GpuKcVisualBrain):
    SCHEMA='matrix_olfactory_endogenous_brain_fp64_cuda_v1'

    @classmethod
    def adopt(cls,parent,*,enabled=True):
        if type(parent) is not GpuKcVisualBrain or type(enabled) is not bool:
            raise ValueError('Require the complete saved visual/KC parent and explicit activation')
        if not parent.orn_synaptic_enabled:
            raise ValueError('Inherited ORN synapse must be enabled')
        before=parent.state_dict();before.pop('schema')
        obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        m=dict(enabled=enabled,adoption_time_ns=obj.time_ns,
            parent_record_sha256=_record_hash(before),source_rows=obj._orn_rows.copy(),
            source_ids=obj.brain.node_ids[obj._orn_rows].copy(),target_rows=obj._pn_rows.copy(),
            target_ids=obj.brain.node_ids[obj._pn_rows].copy(),
            selected_pairs=len(obj.orn_pn_synaptic_manifest['w_csr_data_positions']),
            source_kinetics='Nagel et al.2015 doi:10.1038/nn.3895; inherited fast/slow population approximation.',
            source_rate='state[canonical_ORN_row]*inherited_rmax, recomputed at every adaptive ODE evaluation',
            migration='All inherited resource/filter coordinates and downstream tails retained exactly. Only future source dependence changes; no added current or duplicate ORN->PN path.',
            legacy_held_afferent_rate_active=not enabled,
            recurrent_ORN_inputs_retained=True,PN_LN_inputs_retained=True,APL_dynamic=True,
            native_PN_geometry_replaced=False,local_terminal_release_identified=False,
            native_odor_transduction_identified=False,absolute_conductance_identified=False,
            synthetic_part='Bounded rate/rmax proxy, shared per-ORN terminal resources, published population kinetics and inherited signed-weight/current conversion. Native PN membrane and presynaptic LN receptor mechanisms remain unidentified.',
            new_canonical_neurons=0,new_anatomical_pairs=0,new_dynamic_coordinates=0,
            parameter_fitting=False,biological_validation=False)
        m['record_sha256']=_record_hash(m);obj.olfactory_endogenous_manifest=m
        obj._rebind_continuing_inputs();obj.validate_endogenous()
        after=obj.state_dict();after.pop('schema');after.pop('olfactory_endogenous_manifest')
        if _record_hash(after)!=m['parent_record_sha256']:
            raise ValueError('Source migration changed inherited neural state')
        return obj

    def _rebind_continuing_inputs(self):
        # Rebind the inherited axonal sampler closure to this object, once.
        # A second visual wrapper would subtract/relocate the same input twice.
        while isinstance(self._spatial_inputs,VisualKcInputs):
            self._spatial_inputs=self._spatial_inputs.base
        self._install_visual_inputs()

    def coefficients_gpu(self,state,drive,light):
        import cupy as cp
        target,rate=super().coefficients_gpu(state,drive,light)
        if self.olfactory_endogenous_manifest['enabled']:
            start,end=self.parent_state_size,self.regional_parent_state_size
            o=self._orn_pn_cuda
            t,r=endogenous_coefficients(state,o['rows'],o['caps'],start,end,xp=cp)
            target[start:end]=t;rate[start:end]=r
        return target,rate

    def validate_endogenous(self):
        m=self.olfactory_endogenous_manifest
        if m['record_sha256']!=_record_hash(m) or type(m['enabled']) is not bool:
            raise ValueError('Changed endogenous olfactory source contract')
        if self.brain.n_neurons!=166700 or m['selected_pairs']!=145:
            raise ValueError('Unexpected canonical olfactory graph')
        np.testing.assert_array_equal(m['source_rows'],self._orn_rows)
        np.testing.assert_array_equal(m['source_ids'],self.brain.node_ids[self._orn_rows])
        np.testing.assert_array_equal(m['target_ids'],self.brain.node_ids[self._pn_rows])
        if len(self._orn_rows)!=74 or self.regional_parent_state_size-self.parent_state_size!=296:
            raise ValueError('Unexpected inherited terminal state layout')
        if not self.orn_synaptic_enabled or m['legacy_held_afferent_rate_active'] is m['enabled']:
            raise ValueError('Contradictory source activation')

    def state_dict(self):
        out=super().state_dict()
        out['olfactory_endogenous_manifest']=copy.deepcopy(self.olfactory_endogenous_manifest)
        return out

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:
            raise ValueError('Incomplete endogenous olfactory brain')
        parent={k:v for k,v in saved.items() if k!='olfactory_endogenous_manifest'}
        parent['schema']=GpuKcVisualBrain.SCHEMA
        base=GpuKcVisualBrain.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.olfactory_endogenous_manifest=copy.deepcopy(saved['olfactory_endogenous_manifest'])
        obj._rebind_continuing_inputs();obj.validate_endogenous();return obj

    @staticmethod
    def backend_identity():
        out=GpuKcVisualBrain.backend_identity()
        out['olfactory_endogenous']='Canonical ORN q*rmax drives inherited terminal resources at each ODE stage; external held-rate branch bypass retired when enabled; no new gains or coordinates.'
        return out
