"""Evaluator-only input and body tapes; no change to neural equations/state."""
from __future__ import annotations

import math
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


class TapeBoundary:
    def __init__(self, base, world, arm, tapes):
        need(arm in ('sham', 'common', 'virtual'), 'Unknown tape arm')
        self.base, self.world, self.arm = base, world, arm
        self.steps = np.asarray(tapes['input_trial_ms'])
        key = {'sham':'sham_used', 'common':'common_control_used',
               'virtual':'virtual_intervention_used'}[arm]
        self.values = np.asarray(tapes[key])
        need(np.array_equal(self.steps, np.arange(1021, 1121)), 'Changed input clock')
        if self.values is not None:
            need(self.values.shape == (100, 3) and np.isfinite(self.values).all()
                 and np.all((self.values >= 0) & (self.values <= 1))
                 and np.array_equal(self.values[:, 2], np.zeros(100)), 'Invalid input tape')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def sample(self, data, *args):
        result = self.base.sample(data, *args)
        dt = self.world.time_ns - self.base.origin_ns
        need(dt >= 0 and dt % 1_000_000 == 0, 'Nonintegral sampling clock')
        next_input = dt // 1_000_000 + 1
        if self.values is not None and 1021 <= next_input <= 1120:
            result = dict(result)
            result['concentration'] = self.values[next_input - 1021, :2].copy()
        return result


def make_frozen_motor(parent_class):
    class FrozenMotor(parent_class):
        def __init__(self, obj, donor):
            super().__init__(obj)
            self.donor = donor
            self.delivered_forward = self.delivered_yaw = 0.0
            self.substep_forces = []
            self.substep_commands = []
            self.substep_clock = []

        def torque(self, data, dt):
            force = super().torque(data, dt)
            self.substep_forces.append(force.copy())
            self.substep_commands.append([self.obj.controller.forward_mm_s,
                                          self.obj.controller.yaw_rate_rad_s])
            self.substep_clock.append(int(self.obj.body.steps))
            return force

        def advance(self, torque_native, nsteps=1):
            need(nsteps == 1 and self.obj.command_mode == 'neural', 'Body schedule changed')
            if self.body_calls % self.substeps_per_ms == 0:
                self.trial_step += 1
                dq = np.asarray(self.obj.last_dn - self.obj.dn_baseline, dtype=float)
                need(dq.shape == (4,) and np.isfinite(dq).all(), 'Invalid DN release')
                self.raw = float(np.tanh(250.0 * (dq[2] - dq[3])) * self.OUTPUT_RAD_S)
                self.forward = float(np.clip(0.2 + np.mean(dq[:2]), 0.0, 0.5))
                self.filtered += self.alpha * (self.raw - self.filtered)
                self.applied = (math.copysign(self.OUTPUT_RAD_S, self.filtered)
                                if abs(self.filtered) >= self.THRESHOLD_RAD_S else 0.0)
                self.wind_active = self.WIND_FIRST_STEP <= self.trial_step <= self.WIND_LAST_STEP
                index = self.donor['row_by_step'][self.trial_step]
                self.delivered_forward = float(self.donor['command_forward_mm_s'][index])
                self.delivered_yaw = float(self.donor['command_yaw_rate_rad_s'][index])
            self.body_calls += 1
            requested = self.obj.requested.copy()
            self.obj.command_mode = 'device'
            self.obj.requested = np.array([self.delivered_forward,
                                          self.delivered_yaw / self.OUTPUT_RAD_S], dtype=float)
            try:
                return self.original_advance(torque_native, nsteps)
            finally:
                self.obj.command_mode = 'neural'
                self.obj.requested = requested
                if self.body_calls % self.substeps_per_ms == 0:
                    self.wind_active = False
    return FrozenMotor
