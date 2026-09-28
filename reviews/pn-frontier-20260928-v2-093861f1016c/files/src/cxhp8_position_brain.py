"""Seven anatomical afferents with an explicitly provisional external current."""
import copy
import numpy as np
from pn_graph_solver_brain import GpuPnGraphSolverBrain
from kcgamma_regional_brain import _record_hash
from cxhp8_position_field import IDS,POLICY,PRIOR_SHA256,AMPLITUDE


class GpuCxHP8PositionBrain(GpuPnGraphSolverBrain):
    SCHEMA='matrix_cxhp8_position_brain_v1'

    @classmethod
    def adopt(cls,parent,sample,*,mode):
        if type(parent) is not GpuPnGraphSolverBrain or mode not in ('position','held_initial'):
            raise ValueError('Exact parent and declared CxHP8 mode required')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj.cxhp8_manifest=dict(policy=POLICY,prior_sha256=PRIOR_SHA256,ids=IDS.copy(),mode=mode,origin_ns=parent.time_ns,amplitude_model_units=AMPLITUDE,biological_electrical_calibration=False,new_canonical_neurons=0,neural_parameters_changed=False,initial_sample=copy.deepcopy(sample))
        obj.cxhp8_manifest['record_sha256']=_record_hash(obj.cxhp8_manifest)
        obj.cxhp8_pending=dict(sample_time_ns=parent.time_ns,**copy.deepcopy(sample),applied_index=float(sample['index']))
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_cxhp8()
        after=obj.state_dict()
        for k in ['cxhp8_manifest','cxhp8_pending']:after.pop(k)
        after['schema']=before['schema']
        if _record_hash(before)!=_record_hash(after):raise ValueError('CxHP8 adoption changed neural history')
        return obj

    def validate_cxhp8(self):
        m=self.cxhp8_manifest;p=self.cxhp8_pending
        if (m.get('record_sha256')!=_record_hash({k:v for k,v in m.items() if k!='record_sha256'}) or m['ids']!=IDS or m['policy']!=POLICY or m['prior_sha256']!=PRIOR_SHA256 or m['mode'] not in ('position','held_initial') or m['amplitude_model_units']!=AMPLITUDE or m['biological_electrical_calibration'] is not False or m['neural_parameters_changed'] is not False):
            raise ValueError('Invalid CxHP8 manifest')
        if type(m['origin_ns']) is not int or type(p['sample_time_ns']) is not int or not 0<=m['origin_ns']<=p['sample_time_ns']<=self.time_ns:
            raise ValueError('Invalid CxHP8 sample clock')
        for s in [p,m['initial_sample']]:
            angles=np.asarray(s['angles_deg'])
            if angles.shape!=(3,) or not np.isfinite(angles).all() or not np.isfinite(s['index']) or not 0<=s['index']<=1:raise ValueError('Invalid position index')
        expected=p['index'] if m['mode']=='position' else m['initial_sample']['index']
        if p['applied_index']!=expected:raise ValueError('Pending input differs from declared mode')
        rows=np.searchsorted(self.brain.node_ids,IDS)
        if np.any(rows>=self.brain.n_neurons) or not np.array_equal(self.brain.node_ids[rows],IDS):raise ValueError('Missing CxHP8 canonical afferent')
        protected=np.unique(np.concatenate([self.vi,self._orn_rows,self._pn_rows,self.regional_rows,self._dynamic_cache['rows'],self._cvn7_rows]))
        if len(np.intersect1d(rows,protected)):raise ValueError('CxHP8 overlaps electrical replacement')
        self._cxhp8_rows=rows

    def set_cxhp8_sample(self,sample):
        self.cxhp8_pending=dict(sample_time_ns=self.time_ns,**copy.deepcopy(sample),applied_index=float(sample['index'] if self.cxhp8_manifest['mode']=='position' else self.cxhp8_manifest['initial_sample']['index']))
        self.validate_cxhp8()

    def advance(self,dt_ns,drive,light):
        self.validate_cxhp8()
        if self.cxhp8_pending['sample_time_ns']!=self.time_ns:raise ValueError('Stale CxHP8 input')
        drive=np.asarray(drive,dtype=float).copy()
        if drive.shape!=(self.brain.n_neurons,):raise ValueError('Invalid drive shape')
        drive[self._cxhp8_rows]+=AMPLITUDE*self.cxhp8_pending['applied_index']
        return super().advance(dt_ns,drive,light)

    def _restore_joint(self,parent_state,source_state,published):
        manifest=copy.deepcopy(self.cxhp8_manifest);pending=copy.deepcopy(self.cxhp8_pending)
        super()._restore_joint(parent_state,source_state,published)
        self.cxhp8_manifest=manifest;self.cxhp8_pending=pending;self.validate_cxhp8()

    def state_dict(self):
        self.validate_cxhp8();out=super().state_dict();out.update(schema=self.SCHEMA,cxhp8_manifest=copy.deepcopy(self.cxhp8_manifest),cxhp8_pending=copy.deepcopy(self.cxhp8_pending));return out

    @classmethod
    def from_state(cls,brain,saved):
        if saved.get('schema')!=cls.SCHEMA:raise ValueError('Wrong CxHP8 neural schema')
        parent={k:v for k,v in saved.items() if k not in ('cxhp8_manifest','cxhp8_pending')};parent['schema']=GpuPnGraphSolverBrain.SCHEMA
        base=GpuPnGraphSolverBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.cxhp8_manifest=copy.deepcopy(saved['cxhp8_manifest']);obj.cxhp8_pending=copy.deepcopy(saved['cxhp8_pending'])
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_cxhp8();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnGraphSolverBrain.backend_identity();out['CxHP8_position_input']=POLICY;return out
