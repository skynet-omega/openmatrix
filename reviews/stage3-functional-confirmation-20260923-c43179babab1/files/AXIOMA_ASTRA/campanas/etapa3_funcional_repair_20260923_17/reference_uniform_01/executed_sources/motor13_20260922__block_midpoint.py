"""Complete midpoint exchange of PN, KC/APL and rate-network interfaces.

Predictor is discarded; accepted PN histories remain continuous. Rate-network
stages use predicted midpoint KC/APL outputs; physical KC/APL updates use
midpoint conductances. This is an experimental numerical scheme, not a proof of
order for the hybrid event system. Whole-state convergence is mandatory.
"""
import copy
from types import SimpleNamespace
import numpy as np
import cupy as cp
from kc_apl_dynamics import cascade
from cxhp8_position_field import AMPLITUDE
from execution_parent_snapshot import ParentSnapshotFrames
from projection_parallel_brain import GpuProjectionParallelBrain
from pn_online_cns_brain import GpuPnOnlineBrain
from snapshot_execution_brain import GpuSnapshotExecutionBrain
from prosthetic_olfactory_brain import GpuProstheticOlfactoryBrain
from kc_visual_ports import VisualComponents
from waveform_coupling import restore_parent
import kc_spatial_brain
from kc_spatial_inputs import CanonicalKcSpatialInputs
from pn_electrical_output_brain import GpuPnElectricalOutputBrain

def install(brain,step_ns):
    if step_ns not in (31250,62500,125000,250000,500000,1000000):raise ValueError('Unregistered block midpoint')
    original=GpuSnapshotExecutionBrain.advance;old_components=GpuPnOnlineBrain.conductance_components
    old_gpu=GpuProstheticOlfactoryBrain.advance;old_cascade=kc_spatial_brain.cascade
    old_split=CanonicalKcSpatialInputs.split
    old_extra=GpuPnElectricalOutputBrain.conductance_components
    wrapper=brain._spatial_batch;old_gain=wrapper.gain
    canonical=brain._spatial_inputs
    while hasattr(canonical,'base'):canonical=canonical.base
    active={};mid={};reports={'steps':0,'step_ns':step_ns,'predictor_restore_exact':None,'pn_rejections':0,'predictor_evaluations':0}
    frozen_rows=cp.asnumpy(brain._dynamic_gpu_rows)
    def spatial_split(self,values,**kwargs):
        if self is not canonical or not active.get('accepted'):return old_split(self,values,**kwargs)
        # Only the pure port distributor receives a stage view. Never replace
        # the physical APL state merely to satisfy its conservation check.
        view=copy.copy(self)
        view.brain=SimpleNamespace(kc_apl_dynamic_state={'apl_transmission':mid['apl_transmission']})
        return old_split(view,values,**kwargs)
    def extra_components(self):
        if self is not brain or not active.get('accepted'):return old_extra(self)
        old=components(self)
        if not hasattr(self,'_extra_output_slots') or not self.pn_online_manifest['electrical_outputs']['enabled']:return old
        port=self._online_source.extra_output
        inherited=mid['state'][self.transmission_start+self._online_ports.pn_row]*port.legacy_gain
        delta=self._online_source.additional_output_nS()-inherited
        values=list(old);values[0]=values[0].copy();values[2]=values[2].copy()
        values[0][self._extra_output_slots]+=delta
        values[2][self._extra_apl_cell]+=delta[self._extra_apl_local]*self._extra_apl_fraction
        if np.any(values[0]<0) or np.any(values[2]<0):raise ValueError('Negative receiver conductance in midpoint replacement')
        return VisualComponents(values,old.visual_ge,old.visual_gi,old.gamma_mask)
    def components(self):
        if self is not brain or not active:return old_components(self)
        if not active['start']<=self.time_ns<=active['end'] or self._online_source.time_ns!=active['source_clock']:raise ValueError('Midpoint stage clock mismatch')
        if active['accepted']:
            old=mid['components'];v=[x.copy() for x in old]
            change=self._online_source.output_nS()-mid['pn_output']
            v[0][self._online_dynamic_slot]+=change;v[4][self._online_dynamic_slot]+=change
            return VisualComponents(v,old.visual_ge,old.visual_gi,old.gamma_mask)
        old=GpuProjectionParallelBrain.conductance_components(self);g=self._online_source.output_nS()
        values=list(old);values[0]=values[0].copy();values[4]=values[4].copy()
        values[0][self._online_dynamic_slot]+=g;values[4][self._online_dynamic_slot]+=g
        return VisualComponents(values,old.visual_ge,old.visual_gi,old.gamma_mask)
    def gpu(self,ns,drive,light):
        if self is not brain or not active.get('accepted'):return old_gpu(self,ns,drive,light)
        self.state=self.state.copy();m=mid['state'];self.state[frozen_rows]=m[frozen_rows]
        self.state[self.transmission_start+self._apl_rows]=m[self.transmission_start+self._apl_rows]
        start=self.regional_parent_state_size;n=len(self.regional_rows)
        self.state[start:start+n]=m[start:start+n];self.state[start+2*n:start+3*n]=m[start+2*n:start+3*n]
        positions=self._apl_gpu_positions;saved=self.cuda['weights'][positions].copy()
        self.cuda['weights'][positions]=self._apl_base_weights*mid['apl_release_gpu']
        try:return old_gpu(self,ns,drive,light)
        finally:self.cuda['weights'][positions]=saved
    def filters(x,y,rate,tf,ts,dt):
        if active.get('accepted') and isinstance(x,np.ndarray) and np.shares_memory(x,brain.kc_electrical_scales_state['filters']):
            rate=mid['state'][brain.kc_electrical_scales_manifest['source_rows']]
        return old_cascade(x,y,rate,tf,ts,dt)
    def gain():
        if not active.get('accepted'):return old_gain()
        b=mid['state'][brain.regional_parent_state_size:brain.retinal_parent_state_size].reshape(3,-1)[1]
        return 1./(1.+brain.regional_manifest['parameters']['eta']*b)
    def cns(self,ns,drive,light,accepted):
        active.update(start=self.time_ns,end=self.time_ns+ns,source_clock=self._online_source.time_ns,accepted=accepted)
        try:GpuProjectionParallelBrain.advance(self,ns,drive,light)
        finally:active.clear()
    def pn(self,ns,first,last):
        r=self._online_source.advance(ns,first,last,connected=self.pn_online_manifest['orn_connected'],
            rtol=1e-9,atol=5e-6,maxiter=220,max_newton=8,gate_atol=1e-12,stage_predictor='linear')
        if not r['accepted']:
            reports['pn_rejections']+=1;raise RuntimeError('Midpoint PN rejection: '+str(r.get('reason')))
    def advance(self,dt_ns,drive,light):
        if self is not brain:return original(self,dt_ns,drive,light)
        self.validate_execution();self.validate_cxhp8()
        if self.cxhp8_pending['sample_time_ns']!=self.time_ns:raise ValueError('Stale CxHP8 input')
        drive=np.asarray(drive,dtype=float).copy()
        if drive.shape!=(self.brain.n_neurons,):raise ValueError('Invalid drive shape')
        drive[self._cxhp8_rows]+=AMPLITUDE*self.cxhp8_pending['applied_index'];self._validated_inputs(dt_ns,drive,light);self.validate_online()
        frames=ParentSnapshotFrames(self);left=dt_ns;p=self.kc_electrical_scales_manifest['source']['PN_KCgamma']
        while left:
            ns=min(left,step_ns)
            if ns%2:raise ValueError('Exact midpoint clock required')
            half=ns//2;before,published=frames.capture(self);source_before=self._online_source.state_dict()
            first=copy.deepcopy(self._online_ports.observe());old=self.kc_electrical_scales_state['filters'][:,self._online_slot].copy()
            try:
                cns(self,half,drive,light,False);middle=copy.deepcopy(self._online_ports.observe())
                reports['predictor_evaluations']+=self.statistics['evaluations']-before['statistics']['evaluations']
                active.update(start=self.time_ns,end=self.time_ns,source_clock=self._online_source.time_ns,accepted=False)
                try:
                    values=components(self)
                    mid['components']=VisualComponents([x.copy() for x in values],values.visual_ge.copy(),values.visual_gi.copy(),values.gamma_mask.copy())
                finally:active.clear()
                mid['state']=self.state.copy();mid['pn_output']=self._online_source.output_nS().copy()
                mid['apl_transmission']=self.kc_apl_dynamic_state['apl_transmission'].copy();mid['apl_transmission'].flags.writeable=False
                mid['apl_release_gpu']=cp.asarray(self.routes.outgoing_release(mid['apl_transmission'])[1])
                restore_parent(self,before,published)
                if reports['predictor_restore_exact'] is None:
                    from kcgamma_regional_brain import _record_hash
                    if _record_hash(GpuProjectionParallelBrain.state_dict(self))!=_record_hash(before):raise ValueError('Midpoint predictor restore failed')
                    if _record_hash(self._online_source.state_dict())!=_record_hash(source_before):raise ValueError('Predictor modified PN source/history')
                    reports['predictor_restore_exact']=True
                pn(self,half,first,middle);cns(self,ns,drive,light,True)
                x,y=cascade(old[0],old[1],0.,p['cascade_fast_tau_s'],p['cascade_slow_tau_s'],ns*1e-9)
                self.kc_electrical_scales_state['filters'][:,self._online_slot]=[x,y]
                # Physical owners must replace every frozen fast interface.
                start=self.regional_parent_state_size;n=len(self.regional_rows)
                xx,ss=self._axonal_release.summaries()
                np.testing.assert_array_equal(self.state[start:start+n],xx)
                np.testing.assert_array_equal(self.state[start+2*n:start+3*n],ss)
                np.testing.assert_array_equal(self.state[self._spatial_inputs.rows],self._spatial_batch.host(self._spatial_batch.q))
                np.testing.assert_array_equal(self.state[self._apl_rows],(self.kc_apl_dynamic_state['apl_q']*self._apl_area).sum(axis=1))
                np.testing.assert_array_equal(self.state[self.transmission_start+self._apl_rows],(self.kc_apl_dynamic_state['apl_transmission']*self._apl_area).sum(axis=1))
                last=copy.deepcopy(self._online_ports.observe());pn(self,half,middle,last)
                self.validate_online();self.validate_scales();reports['steps']+=1
            except BaseException:
                active.clear();self._restore_joint(before,source_before,published);raise
            finally:frames.release(before)
            left-=ns
    GpuSnapshotExecutionBrain.advance=advance;GpuPnOnlineBrain.conductance_components=components
    GpuProstheticOlfactoryBrain.advance=gpu;kc_spatial_brain.cascade=filters;wrapper.gain=gain
    CanonicalKcSpatialInputs.split=spatial_split
    GpuPnElectricalOutputBrain.conductance_components=extra_components
    def restore():
        GpuSnapshotExecutionBrain.advance=original;GpuPnOnlineBrain.conductance_components=old_components
        GpuProstheticOlfactoryBrain.advance=old_gpu;kc_spatial_brain.cascade=old_cascade;wrapper.gain=old_gain
        CanonicalKcSpatialInputs.split=old_split
        GpuPnElectricalOutputBrain.conductance_components=old_extra
    return reports,restore
