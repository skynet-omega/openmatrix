"""Independent sensory and motor owners; one committed millisecond of latency."""
import math
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def decode(release, baseline):
    delta = np.asarray(release, float)-np.asarray(baseline, float)
    need(delta.shape==(4,) and np.isfinite(delta).all(), 'Invalid DN decoder input')
    raw_forward = float(np.mean(delta[:2]))
    return raw_forward, float(np.clip(raw_forward,0.,.5)), float(np.tanh(250.*(delta[2]-delta[3]))*math.radians(5.))


class UniformOdor:
    def __init__(self, base, world, arm, plan):
        need(arm in ('sham','odor'), 'Unknown paired arm')
        self.base,self.world,self.arm,self.plan = base,world,arm,plan
        self.origin_ns = int(world.time_ns)

    def concentration(self, elapsed_ms):
        return self.plan['odor_amplitude'] if self.arm=='odor' and self.plan['odor_on_ms']<=elapsed_ms<self.plan['odor_off_ms'] else 0.

    def sample(self, data, source_mm, sigma_mm):
        elapsed = self.world.time_ns-self.origin_ns
        need(elapsed%1_000_000==0 and 0<=elapsed<=self.plan['duration_ms']*1_000_000,'Invalid odor sample time')
        geometry = dict(self.base.sample(data,source_mm,sigma_mm))
        geometry['concentration'] = np.full(2,self.concentration(elapsed//1_000_000),np.float64)
        geometry['contact_reinforcement'] = 0.
        return geometry


class NeuralPropulsion:
    """Read only last committed DNg100. Never read odor, time schedule or position."""
    def __init__(self,obj):
        need(obj.command_mode=='neural' and obj.controller.active,'Missing prepared motor owner')
        need(np.array_equal(obj.dn_ids,[10045,10056,10118,10065]),'Effective DN identities differ')
        self.obj = obj
        self.original_advance = obj.body_advance
        self.substeps = round(.001/obj.body.dt)
        need(self.substeps==40,'Physical grid changed')
        self.body_calls = self.trial_step = 0
        self.raw_forward = self.forward = self.raw_yaw = 0.
        obj.body.advance = self.advance

    def advance(self, torque_native, nsteps=1):
        need(nsteps==1 and self.obj.command_mode=='neural','Unexpected body schedule/owner')
        if self.body_calls%self.substeps==0:
            self.trial_step += 1
            self.raw_forward,self.forward,self.raw_yaw = decode(self.obj.last_dn,self.obj.dn_baseline)
        self.body_calls += 1
        requested = self.obj.requested.copy()
        self.obj.command_mode = 'device'
        self.obj.requested = np.array([self.forward,0.])
        try:
            return self.original_advance(torque_native,nsteps)
        finally:
            self.obj.command_mode = 'neural'
            self.obj.requested = requested

    def audit(self):
        return dict(trial_steps=self.trial_step,body_calls=self.body_calls,
            forward_offset_mm_s=0.,gain_mm_s_per_model_unit=1.,forward_limit_mm_s=.5,
            applied_yaw_rad_s=0.,wind_substeps=0,neural_baseline_recentered=False,
            support_and_zero_command_braking='Inherited contact prosthesis; not passive free body',
            native_muscles='Recorded in shadow; not physical torque owner',
            physical_orientation_locked=False,six_leg_walking_claimed=False)


class CausalIntervals:
    def __init__(self,obj,field,motor,plan):
        self.field,self.motor,self.plan = field,motor,plan
        self.origin_ns = int(obj.core.time_ns)
        self.pending = obj.core.pending_sensors.copy()
        self.previous_dn = obj.core.hybrid.release()[obj.dn_ix].copy()
        self.baseline = obj.dn_baseline.copy()
        need(np.array_equal(self.pending,np.zeros(3)),'Inherited prepared pending is not the declared zero input')

    def before(self,obj,k):
        need(obj.core.world.boundary is self.field,'Sensory owner changed')
        need(obj.core.time_ns==self.origin_ns+(k-1)*1_000_000,'Interval start time')
        need(np.array_equal(obj.core.pending_sensors,self.pending),'Pending input changed outside its owner')
        return self.pending.copy()

    def after(self,obj,row,k,used):
        end = self.origin_ns+k*1_000_000
        need(all(int(row[name])==end for name in ('CNS_time_ns','PN_time_ns','body_time_ns')),'Owner clocks differ')
        need(np.array_equal(row['sensores_usados'],used),'Committed sensory latency changed')
        need(np.array_equal(row['DN_q_usada'],self.previous_dn),'Committed motor latency changed')
        need(np.array_equal(row['DN_baseline'],self.baseline),'Baseline was recentered')
        pending = np.r_[np.full(2,self.field.concentration(k)),0.]
        need(np.array_equal(row['sensores_pendientes'],pending),'Sensory schedule not published')
        raw,forward,yaw = decode(self.previous_dn,self.baseline)
        need(row['command_forward_mm_s']==forward and row['command_yaw_rate_rad_s']==0.,'Motor input bypass')
        need(row['forward_unclipped_mm_s']==raw and row['neural_yaw_unapplied_rad_s']==yaw,'Decoder trace differs')
        need(self.motor.trial_step==k and self.motor.body_calls==40*k,'Physical step count')
        self.pending = pending
        self.previous_dn = row['DN_q_actual'].copy()
