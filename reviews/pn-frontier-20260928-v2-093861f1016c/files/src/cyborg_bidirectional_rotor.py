"""Signed prosthetic wiring of named VL1/VL2 ports; no biological antagonism.

The complete VL1 family drives the positive device terminal, VL2 the negative
terminal. This anatomical-label partition was fixed before inspecting CvN4
responses. It changes neither neural signs nor the predecessor torque gain.
"""
import math

import numpy as np

from cyborg_rotor import (CyborgRotor, MOTOR_IDS as VL1_MOTOR_IDS,
    MECHANICAL_TAU_S, MAX_TORQUE_NM, _clock, _scalar)


VL2_MOTOR_IDS = (11316, 12141)
MOTOR_IDS = tuple(sorted(VL1_MOTOR_IDS + VL2_MOTOR_IDS))
SCHEMA = "matrix_cyborg_bidirectional_rotor_v1"
DESIGN_SHA256 = "e61248a72af82bbfc91456f0a672e4d819d7ea362c1221c5df248289ea8c8b17"


class CyborgBidirectionalRotor(CyborgRotor):
    @staticmethod
    def parameters():
        result = CyborgRotor.parameters()
        result.update(command_rule="mean(VL1_release) - mean(VL2_release)",
                      command_range=[-1., 1.], positive_motor_ids=list(VL1_MOTOR_IDS),
                      negative_motor_ids=list(VL2_MOTOR_IDS),
                      biological_antagonism=False, baseline_subtraction=False,
                      device_design_sha256=DESIGN_SHA256)
        return result

    @classmethod
    def from_predecessor(cls, predecessor):
        if type(predecessor) is not CyborgRotor:
            raise ValueError("Migration requires the explicit unidirectional predecessor")
        # Validate its full native state before preserving its mechanical past.
        prior = CyborgRotor.from_state(predecessor.state_dict())
        obj = cls.from_pose(prior.start_ns, prior.initial_rotation, prior.initial_centers)
        for name in ("time_ns", "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            setattr(obj, name, getattr(prior, name))
        return obj

    @staticmethod
    def map_release(node_ids, release):
        nodes, values = np.asarray(node_ids), np.asarray(release)
        if (nodes.ndim != 1 or nodes.dtype != np.int64 or len(nodes) == 0
                or np.any(np.diff(nodes) <= 0) or values.shape != nodes.shape
                or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any((values < 0.) | (values > 1.))):
            raise ValueError("Canonical sorted int64 IDs and finite normalized releases are required")
        ids = np.asarray(MOTOR_IDS, dtype=np.int64)
        indices = np.searchsorted(nodes, ids)
        if np.any(indices >= len(nodes)) or not np.array_equal(nodes[indices], ids):
            raise ValueError("All eight anatomically named motor ports must exist individually")
        individual = values[indices].astype(np.float64, copy=True)
        vl1 = float(np.mean(individual[np.searchsorted(ids, VL1_MOTOR_IDS)]))
        vl2 = float(np.mean(individual[np.searchsorted(ids, VL2_MOTOR_IDS)]))
        return dict(motor_ids=ids, individual_normalized_release=individual,
                    group_commands={"VL1": vl1, "VL2": vl2}, command=vl1-vl2)

    def advance(self, command, dt_ns):
        u = _scalar(command, "signed motor command")
        if not -1. <= u <= 1. or type(dt_ns) is not int or dt_ns <= 0:
            raise ValueError("Command must be in [-1,1] and duration a positive integer ns")
        duration = dt_ns * 1.e-9
        one_minus_decay = -math.expm1(-duration / MECHANICAL_TAU_S)
        terminal_velocity = math.pi * u
        old_velocity = self.omega_rad_s
        increment = (terminal_velocity * duration
                     + (old_velocity-terminal_velocity) * MECHANICAL_TAU_S * one_minus_decay)
        angle = self.angle_rad + increment
        velocity = (old_velocity * math.exp(-duration / MECHANICAL_TAU_S)
                    + terminal_velocity * one_minus_decay)
        if not math.isfinite(angle) or not math.isfinite(velocity):
            raise FloatingPointError("Signed rotor integration overflow; no clipping or rescue")
        self.angle_rad, self.omega_rad_s = angle, velocity
        self.last_command, self.last_torque_nm = u, MAX_TORQUE_NM * u
        self.time_ns += dt_ns

    def state_dict(self):
        result = super().state_dict()
        result.update(schema=SCHEMA, motor_ids=list(MOTOR_IDS))
        return result

    @classmethod
    def from_state(cls, state):
        keys = {"schema", "parameters", "motor_ids", "start_ns", "time_ns", "initial_rotation",
                "initial_centers", "pivot_mm", "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"}
        if (not isinstance(state, dict) or set(state) != keys or state["schema"] != SCHEMA
                or state["parameters"] != cls.parameters() or state["motor_ids"] != list(MOTOR_IDS)):
            raise ValueError("Unknown or modified signed rotor device contract")
        obj = cls.from_pose(state["start_ns"], state["initial_rotation"], state["initial_centers"])
        obj.time_ns = _clock(state["time_ns"])
        if obj.time_ns < obj.start_ns or not np.array_equal(np.asarray(state["pivot_mm"]), obj.pivot_mm):
            raise ValueError("Signed rotor clock or fixed pivot is inconsistent")
        for key in ("angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            setattr(obj, key, _scalar(state[key], key))
        if (not -1. <= obj.last_command <= 1.
                or obj.last_torque_nm != MAX_TORQUE_NM * obj.last_command):
            raise ValueError("Signed rotor last command and torque are inconsistent")
        if obj.time_ns == obj.start_ns and any(getattr(obj, k) != 0. for k in
                ("angle_rad", "omega_rad_s", "last_command", "last_torque_nm")):
            raise ValueError("A newly attached rotor must be at rest")
        return obj
