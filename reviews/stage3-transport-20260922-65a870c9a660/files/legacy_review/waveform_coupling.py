"""Midpoint predictor with continuous PN receptor forcing; NOT Strang.

The CNS predictor is rolled back. PN driver history follows initial -> predicted
midpoint -> final CNS observations. Only those two physical half intervals commit.
The retained midpoint output defines a numerical coupling approximation. Its
error must converge independently of nonlinear residual acceptance.
"""
import copy
import numpy as np
from kc_apl_dynamics import cascade
from cxhp8_position_field import AMPLITUDE
from execution_parent_snapshot import ParentSnapshotFrames, DYNAMIC, copy_values
from projection_parallel_brain import GpuProjectionParallelBrain
from pn_online_cns_brain import GpuPnOnlineBrain
from snapshot_execution_brain import GpuSnapshotExecutionBrain
from kc_visual_ports import VisualComponents

def restore_parent(h,frame,published):
    held={'held_afferent_rate_hz','held_boundary_light'}
    for k in held:
        if not np.array_equal(getattr(h,k),frame[k]):raise ValueError('Predictor modified held input '+k)
    for k in DYNAMIC-{'kc_spatial_state','kc_axonal_state'}-held:
        setattr(h,k,copy_values(getattr(h,k),frame[k]))
    s=dict(frame['kc_spatial_state']);s.pop('time_ns')
    h._spatial_batch.load_state(s)
    h.kc_spatial_state=copy.deepcopy(frame['kc_spatial_state'])
    h._axonal_release.state=copy_values(h._axonal_release.state,frame['kc_axonal_state'])
    h._spatial_clipped_previous=int(frame['kc_spatial_state']['clipped'].sum())
    h.brain.time_ns=published[0];np.copyto(h.brain.rates,published[1])
    h.validate_dynamic();h.validate_spatial();h.validate_axonal();h.validate_scales()

def install(brain,step_ns):
    if step_ns not in (31250,62500,125000,250000,500000):raise ValueError('Unregistered waveform step')
    original=GpuSnapshotExecutionBrain.advance;old_components=GpuPnOnlineBrain.conductance_components
    active={};reports={'steps':0,'step_ns':step_ns,'predictor_restore_exact':None,'pn_rejections':0}
    def components(self):
        if self is not brain or not active:return old_components(self)
        if not active['start']<=self.time_ns<=active['end'] or self._online_source.time_ns!=active['source_clock']:
            raise ValueError('Waveform stage clock mismatch')
        old=GpuProjectionParallelBrain.conductance_components(self)
        g=self._online_source.output_nS();values=list(old);values[0]=values[0].copy();values[4]=values[4].copy()
        values[0][self._online_dynamic_slot]+=g;values[4][self._online_dynamic_slot]+=g
        return VisualComponents(values,old.visual_ge,old.visual_gi,old.gamma_mask)
    def cns(self,ns,drive,light):
        active.update(start=self.time_ns,end=self.time_ns+ns,source_clock=self._online_source.time_ns)
        try:GpuProjectionParallelBrain.advance(self,ns,drive,light)
        finally:active.clear()
    def pn(self,ns,first,last):
        r=self._online_source.advance(ns,first,last,connected=self.pn_online_manifest['orn_connected'],
            rtol=1e-9,atol=5e-6,maxiter=220,max_newton=8,gate_atol=1e-12,stage_predictor='linear')
        if not r['accepted']:
            reports['pn_rejections']+=1;raise RuntimeError('Waveform PN rejection: '+str(r.get('reason')))
    def advance(self,dt_ns,drive,light):
        if self is not brain:return original(self,dt_ns,drive,light)
        self.validate_execution();self.validate_cxhp8()
        if self.cxhp8_pending['sample_time_ns']!=self.time_ns:raise ValueError('Stale CxHP8 input')
        drive=np.asarray(drive,dtype=float).copy()
        if drive.shape!=(self.brain.n_neurons,):raise ValueError('Invalid drive shape')
        drive[self._cxhp8_rows]+=AMPLITUDE*self.cxhp8_pending['applied_index']
        self._validated_inputs(dt_ns,drive,light);self.validate_online()
        frames=ParentSnapshotFrames(self);left=dt_ns;p=self.kc_electrical_scales_manifest['source']['PN_KCgamma']
        while left:
            ns=min(left,step_ns)
            if ns%2:raise ValueError('Waveform halves require exact integer clocks')
            half=ns//2;before,published=frames.capture(self);source_before=self._online_source.state_dict()
            first=self._online_ports.observe();old=self.kc_electrical_scales_state['filters'][:,self._online_slot].copy()
            try:
                cns(self,half,drive,light);mid=self._online_ports.observe()
                restore_parent(self,before,published)
                if reports['predictor_restore_exact'] is None:
                    from kcgamma_regional_brain import _record_hash
                    if _record_hash(GpuProjectionParallelBrain.state_dict(self))!=_record_hash(before):
                        raise ValueError('Predictor restore changed declared parent state')
                    reports['predictor_restore_exact']=True
                pn(self,half,first,mid)
                cns(self,ns,drive,light)
                x,y=cascade(old[0],old[1],0.,p['cascade_fast_tau_s'],p['cascade_slow_tau_s'],ns*1e-9)
                self.kc_electrical_scales_state['filters'][:,self._online_slot]=[x,y]
                last=self._online_ports.observe();pn(self,half,mid,last)
                self.validate_online();self.validate_scales();reports['steps']+=1
            except BaseException:
                active.clear();self._restore_joint(before,source_before,published);raise
            finally:frames.release(before)
            left-=ns
    GpuSnapshotExecutionBrain.advance=advance;GpuPnOnlineBrain.conductance_components=components
    def restore():
        GpuSnapshotExecutionBrain.advance=original;GpuPnOnlineBrain.conductance_components=old_components
    return reports,restore
