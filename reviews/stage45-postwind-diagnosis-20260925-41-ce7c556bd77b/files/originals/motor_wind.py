"""Prospective motor interface and physical wind pulse for one coupled trial.

The decoder uses only the released DN signal. The source position and bearing
are never inputs to the filter. This is an engineered output intervention, not
evidence that the unmodified connectome already navigates.
"""
from __future__ import annotations

import math

import mujoco as mj
import numpy as np


class MotorWind:
    TAU_S = 0.200
    THRESHOLD_RAD_S = 0.0005
    OUTPUT_RAD_S = math.radians(5.0)
    WIND_FIRST_STEP = 1001
    WIND_LAST_STEP = 1020
    WIND_TORQUE_NATIVE = -0.004672697857153467

    def __init__(self, obj):
        if obj.command_mode != 'neural' or obj.controller is None or not obj.controller.active:
            raise ValueError('Expected active neural contact prosthesis after preparation')
        self.obj = obj
        self.original_advance = obj.body_advance
        self.original_torque = obj.controller.torque
        self.alpha = -math.expm1(-0.001 / self.TAU_S)
        self.filtered = 0.0
        self.raw = 0.0
        self.applied = 0.0
        self.forward = 0.0
        self.trial_step = 0
        self.body_calls = 0
        self.substeps_per_ms = round(0.001 / obj.body.dt)
        if self.substeps_per_ms != 40 or abs(self.substeps_per_ms*obj.body.dt-0.001)>1e-12:
            raise ValueError('Expected 25-us physical grid')
        self.wind_substeps = 0
        self.wind_active = False
        self.wind_generalized_peak_native = 0.0
        obj.body.advance = self.advance
        obj.controller.torque = self.torque

    def advance(self, torque_native, nsteps=1):
        if nsteps != 1 or self.obj.command_mode != 'neural':
            raise ValueError('Unexpected body integration schedule or motor owner')
        if self.body_calls % self.substeps_per_ms == 0:
            self.trial_step += 1
            dq = np.asarray(self.obj.last_dn - self.obj.dn_baseline, dtype=float)
            if dq.shape != (4,) or not np.isfinite(dq).all():
                raise FloatingPointError('Invalid DN release')
            self.raw = float(np.tanh(250.0 * (dq[2] - dq[3])) * self.OUTPUT_RAD_S)
            self.forward = float(np.clip(0.2 + np.mean(dq[:2]), 0.0, 0.5))
            self.filtered += self.alpha * (self.raw - self.filtered)
            self.applied = (math.copysign(self.OUTPUT_RAD_S, self.filtered)
                            if abs(self.filtered) >= self.THRESHOLD_RAD_S else 0.0)
            self.wind_active = self.WIND_FIRST_STEP <= self.trial_step <= self.WIND_LAST_STEP
        self.body_calls += 1
        old_requested = self.obj.requested.copy()
        self.obj.command_mode = 'device'
        self.obj.requested = np.array([self.forward, self.applied / self.OUTPUT_RAD_S], dtype=float)
        try:
            return self.original_advance(torque_native, nsteps)
        finally:
            self.obj.command_mode = 'neural'
            self.obj.requested = old_requested
            if self.body_calls % self.substeps_per_ms == 0:
                self.wind_active = False

    def torque(self, data, dt):
        force = self.original_torque(data, dt)
        if self.wind_active:
            extra = np.zeros_like(force)
            body_id = self.obj.controller.thorax
            mj.mj_applyFT(self.obj.body.model, data, np.zeros(3),
                          np.array([0.0, 0.0, self.WIND_TORQUE_NATIVE]),
                          data.xpos[body_id].copy(), body_id, extra)
            if not np.isfinite(extra).all():
                raise FloatingPointError('Wind generalized force nonfinite')
            force = force + extra
            self.wind_substeps += 1
            self.wind_generalized_peak_native = max(self.wind_generalized_peak_native,
                                                     float(np.max(np.abs(extra))))
        return force

    def audit(self):
        return dict(trial_steps=self.trial_step, body_calls=self.body_calls,
                    wind_substeps=self.wind_substeps,
                    expected_wind_substeps=(self.WIND_LAST_STEP-self.WIND_FIRST_STEP+1)*self.substeps_per_ms,
                    tau_s=self.TAU_S, threshold_rad_s=self.THRESHOLD_RAD_S,
                    output_rad_s=self.OUTPUT_RAD_S,
                    wind_first_step=self.WIND_FIRST_STEP, wind_last_step=self.WIND_LAST_STEP,
                    wind_torque_native=self.WIND_TORQUE_NATIVE,
                    wind_torque_Nm=self.WIND_TORQUE_NATIVE*1e-7,
                    wind_generalized_peak_native=self.wind_generalized_peak_native)
