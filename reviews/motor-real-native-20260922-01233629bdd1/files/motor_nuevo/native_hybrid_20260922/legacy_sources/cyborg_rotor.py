"""A fixed optical bench with a one-axis motor, not an anatomical fly neck.

Six anatomical VL1 motor ports supply an arithmetic-mean device command. The
positive direction, torque gain, inertia and damping are engineering choices.
No visual error, target orientation, spring, neural sign or behavior policy is
computed here. Both eyes rotate rigidly about their initial midpoint.
"""
import copy
import math

import numpy as np

from retinal_world import CompoundEye


MOTOR_IDS = (10156, 10627, 10754, 174810, 192306, 556449)
SCHEMA = "matrix_cyborg_rotor_v1"
INERTIA_KG_M2 = 1.e-12
MECHANICAL_TAU_S = .020
DAMPING_NM_S = INERTIA_KG_M2 / MECHANICAL_TAU_S
MAX_TORQUE_NM = DAMPING_NM_S * math.pi


def _clock(value):
    if type(value) is not int or value < 0:
        raise ValueError("Rotor clock must be a nonnegative Python integer in ns")
    return value


def _scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"Invalid {label}")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def _rotation(value):
    rotation = np.asarray(value, dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1.e-10)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1.e-10)):
        raise ValueError("A proper finite orthonormal optical frame is required")
    return rotation.copy()


def _centers(value):
    if not isinstance(value, dict) or set(value) != {"L", "R"}:
        raise ValueError("Exactly two named eye centers L/R are required")
    centers = {side: np.asarray(value[side], dtype=np.float64).copy() for side in ("L", "R")}
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in centers.values()):
        raise ValueError("Finite three-dimensional eye centers are required")
    if np.array_equal(centers["L"], centers["R"]):
        raise ValueError("Eye centers must be distinct")
    return centers


class CyborgRotor:
    """Continuous 360-degree gimbal; positive pitch means frame @ Ry(angle)."""

    @classmethod
    def from_pose(cls, time_ns, rotation, centers):
        obj = cls()
        obj.start_ns = obj.time_ns = _clock(time_ns)
        obj.initial_rotation = _rotation(rotation)
        obj.initial_centers = _centers(centers)
        obj.pivot_mm = (obj.initial_centers["L"] + obj.initial_centers["R"]) * .5
        obj.angle_rad = obj.omega_rad_s = obj.last_command = obj.last_torque_nm = 0.
        return obj

    @staticmethod
    def parameters():
        return dict(inertia_kg_m2=INERTIA_KG_M2, damping_nm_s=DAMPING_NM_S,
                    max_torque_nm=MAX_TORQUE_NM, mechanical_tau_s=MECHANICAL_TAU_S,
                    command_rule="arithmetic_mean_of_six_individual_VL1_motor_releases",
                    positive_direction="initial_rotation @ Ry(angle_rad); engineered wiring",
                    anatomical_neck=False, restoring_spring=False,
                    target_orientation=False, visual_feedback_controller=False,
                    rotation_limit_rad=None, actuator_lag_s=0.)

    @staticmethod
    def map_release(node_ids, release):
        nodes, values = np.asarray(node_ids), np.asarray(release)
        if (nodes.ndim != 1 or nodes.dtype != np.int64 or len(nodes) == 0
                or np.any(np.diff(nodes) <= 0) or values.shape != nodes.shape
                or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any((values < 0.) | (values > 1.))):
            raise ValueError("Canonical sorted int64 IDs and finite normalized releases are required")
        motor_ids = np.asarray(MOTOR_IDS, dtype=np.int64)
        indices = np.searchsorted(nodes, motor_ids)
        if np.any(indices >= len(nodes)) or not np.array_equal(nodes[indices], motor_ids):
            raise ValueError("Every specified VL1 motor port must exist individually")
        individual = values[indices].astype(np.float64, copy=True)
        return dict(motor_ids=motor_ids, individual_normalized_release=individual,
                    command=float(np.mean(individual)))

    def advance(self, command, dt_ns):
        """Exact solution for a piecewise constant command; no numerical step size fit."""
        u = _scalar(command, "motor command")
        if not 0. <= u <= 1. or type(dt_ns) is not int or dt_ns <= 0:
            raise ValueError("Command must be in [0,1] and duration a positive integer ns")
        duration = dt_ns * 1.e-9
        decay_difference = -math.expm1(-duration / MECHANICAL_TAU_S)
        terminal_velocity = math.pi * u
        old_velocity = self.omega_rad_s
        increment = (terminal_velocity * duration
                     + (old_velocity - terminal_velocity) * MECHANICAL_TAU_S * decay_difference)
        angle = self.angle_rad + increment
        velocity = (old_velocity * math.exp(-duration / MECHANICAL_TAU_S)
                    + terminal_velocity * decay_difference)
        if not math.isfinite(angle) or not math.isfinite(velocity):
            raise FloatingPointError("Rotor integration overflow; no clipping or rescue")
        self.angle_rad, self.omega_rad_s = angle, velocity
        self.last_command, self.last_torque_nm = u, MAX_TORQUE_NM * u
        self.time_ns += dt_ns

    def pose(self):
        cosine, sine = math.cos(self.angle_rad), math.sin(self.angle_rad)
        local = np.array([[cosine, 0., sine], [0., 1., 0.], [-sine, 0., cosine]])
        rotation = self.initial_rotation @ local
        delta = rotation @ self.initial_rotation.T
        centers = {side: self.pivot_mm + delta @ (center - self.pivot_mm)
                   for side, center in self.initial_centers.items()}
        return rotation, centers

    def state_dict(self):
        return dict(schema=SCHEMA, parameters=self.parameters(), motor_ids=list(MOTOR_IDS),
                    start_ns=self.start_ns, time_ns=self.time_ns,
                    initial_rotation=self.initial_rotation.copy(),
                    initial_centers=copy.deepcopy(self.initial_centers), pivot_mm=self.pivot_mm.copy(),
                    angle_rad=self.angle_rad, omega_rad_s=self.omega_rad_s,
                    last_command=self.last_command, last_torque_nm=self.last_torque_nm)

    @classmethod
    def from_state(cls, state):
        keys = {"schema", "parameters", "motor_ids", "start_ns", "time_ns", "initial_rotation",
                "initial_centers", "pivot_mm", "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"}
        if (not isinstance(state, dict) or set(state) != keys or state["schema"] != SCHEMA
                or state["parameters"] != cls.parameters() or state["motor_ids"] != list(MOTOR_IDS)):
            raise ValueError("Unknown or modified rotor device contract")
        obj = cls.from_pose(state["start_ns"], state["initial_rotation"], state["initial_centers"])
        obj.time_ns = _clock(state["time_ns"])
        if obj.time_ns < obj.start_ns or not np.array_equal(np.asarray(state["pivot_mm"]), obj.pivot_mm):
            raise ValueError("Rotor clock or fixed pivot is inconsistent")
        for key in ("angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            setattr(obj, key, _scalar(state[key], key))
        if (not 0. <= obj.last_command <= 1.
                or obj.last_torque_nm != MAX_TORQUE_NM * obj.last_command):
            raise ValueError("Rotor last command and torque are inconsistent")
        if obj.time_ns == obj.start_ns and any(getattr(obj, k) != 0. for k in
                ("angle_rad", "omega_rad_s", "last_command", "last_torque_nm")):
            raise ValueError("A newly attached rotor must be at rest")
        return obj


class CyborgEye(CompoundEye):
    """Preserve the optical mapping; observe the independent motorized apparatus."""

    @classmethod
    def adopt(cls, parent_eye, rotor):
        return cls.from_state(parent_eye.brain, parent_eye.body, parent_eye.state_dict(), rotor)

    @classmethod
    def from_state(cls, brain, body, state, rotor):
        if not isinstance(rotor, CyborgRotor):
            raise ValueError("A validated cyborg rotor is required")
        obj = cls.__new__(cls)
        obj._initialize(brain, body, state)
        obj.rotor = rotor
        return obj

    def pose(self):
        return self.rotor.pose()
