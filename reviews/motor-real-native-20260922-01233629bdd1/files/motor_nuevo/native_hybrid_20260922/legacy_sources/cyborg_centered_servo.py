"""Fixed neutral-calibrated motor-to-position prosthesis, without visual input.

The mechanical servo follows a position supplied by the same eight neural ports.
Its local position regulation is engineering; no visual error or scene is read.
The critically damped mechanical ODE has an exact constant-command solution.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cyborg_rotor import CyborgRotor, _clock, _scalar
from cyborg_bidirectional_rotor import (CyborgBidirectionalRotor, MOTOR_IDS,
                                        VL1_MOTOR_IDS, VL2_MOTOR_IDS)

SCHEMA = "matrix_cyborg_centered_servo_v1"
MECHANICAL_TAU_S = .020
INERTIA_KG_M2 = 1.e-12
NEUTRAL_ENVELOPE_DEG = 15.


def _record(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Calibration manifest must be a JSON record")
    return json.loads(json.dumps(value, allow_nan=False))


def calibrate_neutral_trace(path):
    """Return a reproducible record from an explicitly static 400 ms trace."""
    path = Path(path).resolve()
    protocol = json.loads((path.parent / "protocol.json").read_text())
    if protocol.get("condition") != "static":
        raise ValueError("Device calibration requires a neutral static stimulus")
    with np.load(path, allow_pickle=False) as trace:
        command, time = trace["neural_command"], trace["time_ns"]
        if (command.shape != (401,) or time.shape != command.shape
                or time.dtype.kind not in "iu" or np.any(np.diff(time) != 1_000_000)
                or int(time[-1]) - int(time[0]) != 400_000_000
                or not np.isfinite(command).all() or np.any(np.abs(command) > 1.)):
            raise ValueError("Need 401 finite bounded samples spanning 400 ms at 1 ms")
        bias = float(np.mean(command))
        d95 = float(np.percentile(np.abs(command - bias), 95., method="linear"))
        if d95 <= 0.:
            raise ValueError("A constant neutral command cannot identify an excursion scale")
        gain = math.radians(NEUTRAL_ENVELOPE_DEG) / d95
        return dict(schema="matrix_centered_servo_neutral_calibration_v1",
            trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            start_ns=int(time[0]), end_ns=int(time[-1]), sample_count=len(command),
            neutral_bias=bias, d95=d95, gain_rad_per_raw_unit=gain,
            neutral_envelope_deg=NEUTRAL_ENVELOPE_DEG,
            estimator="Arithmetic mean of all 401 samples; linear 95th percentile of abs(u-mean(u)).",
            design_rule="G=radians(15)/d95; fixed before visual testing. No visual-response fitting.",
            biological_calibration=False)


class CyborgCenteredServo(CyborgBidirectionalRotor):
    @classmethod
    def from_pose(cls, *args, **kwargs):
        raise ValueError("Adopt an existing signed rotor with explicit neutral calibration")

    @classmethod
    def from_predecessor(cls, predecessor, neutral_bias, gain_rad_per_raw_unit,
                         calibration_manifest=None):
        if type(predecessor) is not CyborgBidirectionalRotor:
            raise ValueError("Migration requires the explicit bidirectional predecessor")
        prior = CyborgBidirectionalRotor.from_state(predecessor.state_dict())
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(prior.__dict__))
        obj.neutral_bias = _scalar(neutral_bias, "neutral bias")
        obj.gain_rad_per_raw_unit = _scalar(gain_rad_per_raw_unit, "position gain")
        obj.mount_angle_rad = prior.angle_rad
        obj.adoption_time_ns = prior.time_ns
        obj.calibration_manifest = _record(calibration_manifest)
        obj.adoption = dict(predecessor_schema=prior.state_dict()["schema"],
            predecessor_last_command=prior.last_command, predecessor_last_torque_nm=prior.last_torque_nm,
            command_initialization="New servo initially requests its mounting angle (command=0); pending neural command must be sampled by the session.",
            preserved="Angle, angular velocity, optical geometry and both clocks; new torque follows the new position device.")
        obj.last_command = 0.
        obj.last_torque_nm = obj._torque(obj.angle_rad, obj.omega_rad_s, obj.mount_angle_rad)
        obj._validate()
        return obj

    def parameters(self):
        tau, inertia = MECHANICAL_TAU_S, INERTIA_KG_M2
        return dict(inertia_kg_m2=inertia, mechanical_tau_s=tau,
            damping_nm_s=2. * inertia / tau, position_stiffness_nm_per_rad=inertia / tau**2,
            neutral_bias=self.neutral_bias, gain_rad_per_raw_unit=self.gain_rad_per_raw_unit,
            command_normalization=1. + abs(self.neutral_bias),
            command_rule="raw=mean(VL1)-mean(VL2); command=(raw-neutral_bias)/(1+abs(neutral_bias))",
            position_rule="theta_target=mount_angle_rad+gain_rad_per_raw_unit*(1+abs(neutral_bias))*command",
            mechanical_equation="J*theta_ddot=J*((theta_target-theta)/tau^2-2*theta_dot/tau)",
            torque_observation="Instantaneous net mechanical torque at the end of the last interval",
            mount_angle_rad=self.mount_angle_rad, command_range=[-1., 1.],
            positive_motor_ids=list(VL1_MOTOR_IDS), negative_motor_ids=list(VL2_MOTOR_IDS),
            biological_antagonism=False, baseline_subtraction=True,
            anatomical_neck=False, restoring_spring=False, internal_position_servo=True,
            target_orientation=True, visual_target_orientation=False, visual_feedback_controller=False,
            positive_direction="initial_rotation @ Ry(angle_rad); engineered wiring",
            rotation_limit_rad=None, actuator_lag_s=0., max_torque_nm=None,
            torque_limit="None; ideal powered prosthesis with fixed declared physical parameters",
            zero_command="Return to or maintain mounting angle; neural input is neutralized, servo remains powered",
            calibration_manifest=copy.deepcopy(self.calibration_manifest))

    def map_release(self, node_ids, release):
        ports = CyborgBidirectionalRotor.map_release(node_ids, release)
        raw = ports["command"]
        ports["raw_command"] = raw
        ports["command"] = (raw - self.neutral_bias) / (1. + abs(self.neutral_bias))
        return ports

    def target_angle(self, command):
        value = _scalar(command, "servo command")
        if not -1. <= value <= 1.:
            raise ValueError("Servo command must be in [-1,1]")
        return self.mount_angle_rad + self.gain_rad_per_raw_unit * (1. + abs(self.neutral_bias)) * value

    @staticmethod
    def _torque(angle, velocity, target):
        tau = MECHANICAL_TAU_S
        return INERTIA_KG_M2 * ((target - angle) / tau**2 - 2. * velocity / tau)

    def _validate(self):
        if not -1. <= self.neutral_bias <= 1. or self.gain_rad_per_raw_unit <= 0.:
            raise ValueError("Finite normalized neutral bias and strictly positive gain required")
        for name in ("neutral_bias", "gain_rad_per_raw_unit", "mount_angle_rad",
                     "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            _scalar(getattr(self, name), name)
        if not _clock(self.start_ns) <= _clock(self.adoption_time_ns) <= _clock(self.time_ns):
            raise ValueError("Inconsistent servo clocks")
        expected = self._torque(self.angle_rad, self.omega_rad_s, self.target_angle(self.last_command))
        if not math.isfinite(expected) or self.last_torque_nm != expected:
            raise ValueError("Servo torque does not match its saved physical state")
        if self.time_ns == self.adoption_time_ns and self.angle_rad != self.mount_angle_rad:
            raise ValueError("Servo mounting angle must preserve its adoption angle")
        for key in ("neutral_bias", "gain_rad_per_raw_unit"):
            if key in self.calibration_manifest and self.calibration_manifest[key] != getattr(self, key):
                raise ValueError("Servo parameters differ from their calibration record")

    def advance(self, command, dt_ns):
        self._validate()
        value = _scalar(command, "servo command")
        target = self.target_angle(value)
        if type(dt_ns) is not int or dt_ns <= 0:
            raise ValueError("Duration must be a positive Python integer in ns")
        dt, tau = dt_ns * 1.e-9, MECHANICAL_TAU_S
        decay = math.exp(-dt / tau)
        error = self.angle_rad - target
        coefficient = self.omega_rad_s + error / tau
        angle = target + (error + coefficient * dt) * decay
        velocity = (self.omega_rad_s - coefficient * dt / tau) * decay
        torque = self._torque(angle, velocity, target)
        if not all(map(math.isfinite, (angle, velocity, torque))):
            raise FloatingPointError("Servo integration overflow; no clipping or rescue")
        self.angle_rad, self.omega_rad_s = angle, velocity
        self.last_command, self.last_torque_nm = value, torque
        self.time_ns += dt_ns

    def state_dict(self):
        self._validate()
        state = CyborgBidirectionalRotor.state_dict(self)
        state.update(schema=SCHEMA, neutral_bias=self.neutral_bias,
            gain_rad_per_raw_unit=self.gain_rad_per_raw_unit, mount_angle_rad=self.mount_angle_rad,
            adoption_time_ns=self.adoption_time_ns, calibration_manifest=copy.deepcopy(self.calibration_manifest),
            adoption=copy.deepcopy(self.adoption))
        return state

    @classmethod
    def from_state(cls, state):
        base_keys = {"schema", "parameters", "motor_ids", "start_ns", "time_ns", "initial_rotation",
                     "initial_centers", "pivot_mm", "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"}
        extra = {"neutral_bias", "gain_rad_per_raw_unit", "mount_angle_rad", "adoption_time_ns", "calibration_manifest", "adoption"}
        if not isinstance(state, dict) or set(state) != base_keys | extra or state["schema"] != SCHEMA:
            raise ValueError("Unknown or incomplete centered servo state")
        prior = CyborgRotor.from_pose(state["start_ns"], state["initial_rotation"], state["initial_centers"])
        obj = cls.__new__(cls)
        obj.__dict__.update(prior.__dict__)
        for name in ("time_ns", "adoption_time_ns"):
            setattr(obj, name, _clock(state[name]))
        for name in ("neutral_bias", "gain_rad_per_raw_unit", "mount_angle_rad",
                     "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            setattr(obj, name, _scalar(state[name], name))
        obj.calibration_manifest, obj.adoption = _record(state["calibration_manifest"]), _record(state["adoption"])
        if (state["parameters"] != obj.parameters() or state["motor_ids"] != list(MOTOR_IDS)
                or not np.array_equal(state["pivot_mm"], obj.pivot_mm)):
            raise ValueError("Modified centered servo physical contract")
        obj._validate()
        return obj
