"""Keep the campaign40 motor interface; fix only physical torque mapping."""
from pathlib import Path
import importlib.util
import numpy as np
from wind_mapping import WorldTorque

_source = Path(__file__).resolve().parents[2] / 'campanas/etapa45_navigation_wind_20260925_40/motor_wind.py'
_spec = importlib.util.spec_from_file_location('_wind_protocol40', _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class MotorWind(_module.MotorWind):
    def __init__(self, obj):
        super().__init__(obj)
        self.mapper = WorldTorque(obj.body.model, obj.controller.thorax)
        self.wind_log = []

    def torque(self, data, dt):
        force = self.original_torque(data, dt)
        if self.wind_active:
            world = np.array([0., 0., self.WIND_TORQUE_NATIVE])
            extra = self.mapper.map(data, world)
            if np.linalg.norm(extra) == 0.:
                raise FloatingPointError('Scheduled wind mapped to zero force')
            force = force + extra
            self.wind_substeps += 1
            self.wind_generalized_peak_native = max(self.wind_generalized_peak_native,
                                                     float(np.max(np.abs(extra))))
            self.wind_log.append(dict(trial_step=self.trial_step, body_call=self.body_calls,
                                      body_time_s=float(data.time), world_torque=world,
                                      generalized=extra, qpos=data.qpos.copy()))
        return force

    def audit(self):
        result = super().audit()
        result.update(mapping='current_pose_scratch_kinematics_comPos',
                      nonzero_substeps=int(sum(np.linalg.norm(r['generalized']) > 0 for r in self.wind_log)),
                      physical_restart_during_trial=False)
        return result
