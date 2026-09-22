"""Bounded cervical apparatus driven by the same eight canonical motor ports.

The range, critically damped servo, inertia and inelastic stops are engineering
choices. Gorko et al. 2024 supports relating cervical output to head pose, not
these actuator parameters or the signed VL1/VL2 wiring. Constant commands have
an analytic free solution and at most one impact; no integration microsteps or
neural gain calibration are introduced.
"""
import copy
import math

import numpy as np

from cyborg_rotor import CyborgRotor, _clock, _scalar
from cyborg_bidirectional_rotor import (
    CyborgBidirectionalRotor, MOTOR_IDS, VL1_MOTOR_IDS, VL2_MOTOR_IDS)
from cyborg_centered_servo import (
    CyborgCenteredServo, MECHANICAL_TAU_S, INERTIA_KG_M2)


SCHEMA = "matrix_cyborg_bounded_servo_v1"
HALF_RANGE_RAD = math.pi / 4.
_BASE_KEYS = {"schema", "parameters", "motor_ids", "start_ns", "time_ns",
    "initial_rotation", "initial_centers", "pivot_mm", "angle_rad",
    "omega_rad_s", "last_command", "last_torque_nm"}
_EXTRA_KEYS = {"reference_angle_rad", "half_range_rad", "adoption_time_ns",
    "adoption", "impact_count", "last_impact", "impact_energy_loss_total_j",
    "impact_impulse_total_nm_s"}
_IMPACT_KEYS = {"interval_start_ns", "interval_dt_ns", "offset_s", "side",
    "angle_rad", "velocity_before_rad_s", "impulse_nm_s", "energy_loss_j"}


def _reference(predecessor):
    turns = round((predecessor.angle_rad - predecessor.mount_angle_rad) / (2. * math.pi))
    return predecessor.mount_angle_rad + 2. * math.pi * turns, turns


class CyborgBoundedServo(CyborgBidirectionalRotor):
    @classmethod
    def from_pose(cls, *args, **kwargs):
        raise ValueError("Adopt the complete existing centered servo")

    @classmethod
    def from_predecessor(cls, predecessor, *, initial_command):
        if type(predecessor) is not CyborgCenteredServo:
            raise ValueError("Migration requires the explicit centered servo predecessor")
        previous = predecessor.state_dict()
        prior = CyborgCenteredServo.from_state(previous)
        reference, turns = _reference(prior)
        if not reference - HALF_RANGE_RAD <= prior.angle_rad <= reference + HALF_RANGE_RAD:
            raise ValueError("Inherited pose is outside the fixed range; no projection or reset")
        base = CyborgRotor.from_pose(prior.start_ns, prior.initial_rotation, prior.initial_centers)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        for name in ("time_ns", "angle_rad", "omega_rad_s"):
            setattr(obj, name, getattr(prior, name))
        obj.reference_angle_rad = reference
        obj.half_range_rad = HALF_RANGE_RAD
        obj.adoption_time_ns = prior.time_ns
        obj.adoption = dict(previous_state=copy.deepcopy(previous), reference_turns=turns,
            operation="Replace only the motor device transfer and add fixed inelastic travel stops",
            preserved="Numerical angle, angular velocity, optical geometry and both device clocks",
            command_initialization="Current raw mean(VL1)-mean(VL2), or zero when output is disconnected",
            momentum_reset=False, retroactive_pose_projection=False,
            biological_parameter_calibration=False)
        obj.last_command = _scalar(initial_command, "initial bounded servo command")
        obj.last_torque_nm = obj._torque(obj.angle_rad, obj.omega_rad_s,
                                        obj.target_angle(obj.last_command))
        obj.impact_count = 0
        obj.last_impact = None
        obj.impact_energy_loss_total_j = obj.impact_impulse_total_nm_s = 0.
        obj._validate()
        return obj

    @property
    def lower_limit_rad(self):
        return self.reference_angle_rad - self.half_range_rad

    @property
    def upper_limit_rad(self):
        return self.reference_angle_rad + self.half_range_rad

    def parameters(self):
        return dict(inertia_kg_m2=INERTIA_KG_M2, mechanical_tau_s=MECHANICAL_TAU_S,
            damping_nm_s=2. * INERTIA_KG_M2 / MECHANICAL_TAU_S,
            position_stiffness_nm_per_rad=INERTIA_KG_M2 / MECHANICAL_TAU_S**2,
            reference_angle_rad=self.reference_angle_rad, half_range_rad=HALF_RANGE_RAD,
            rotation_limit_rad=[self.lower_limit_rad, self.upper_limit_rad],
            command_rule="raw=mean(VL1)-mean(VL2); command=raw",
            command_range=[-1., 1.], baseline_subtraction=False,
            position_rule="theta_target=reference_angle_rad+half_range_rad*command",
            positive_motor_ids=list(VL1_MOTOR_IDS), negative_motor_ids=list(VL2_MOTOR_IDS),
            positive_direction="initial_rotation @ Ry(angle_rad); engineered wiring",
            biological_antagonism=False, anatomical_neck=False,
            internal_position_servo=True, target_orientation=True,
            visual_target_orientation=False, visual_feedback_controller=False,
            mechanical_equation="J*theta_ddot=J*((theta_target-theta)/tau^2-2*theta_dot/tau)",
            mechanical_parameters="Inherited engineering assumptions; neither speed nor inertia calibrated to a fly",
            travel_range_status="Fixed engineering +/-45 degrees; not a measured biological neck limit",
            reference_rule="Previous mounting angle plus the integer number of full turns nearest the inherited angle",
            anatomical_context="Gorko et al. 2024, doi:10.1038/s41586-024-07222-5; cervical output and head pose only",
            actuator_lag_s=0., max_torque_nm=None,
            torque_limit="Ideal powered servo; no continuous torque limit",
            stop_model="Perfectly inelastic instantaneous contact, restitution zero; at most one event per constant command",
            event_solver="Analytic monotonic segment, at most 64 bisections, then analytic remainder",
            torque_observation="Continuous mechanical torque at interval end; impact impulse is stored separately",
            zero_command="Powered return to reference angle; neural output disconnected, camera remains mobile")

    def target_angle(self, command):
        command = _scalar(command, "bounded servo command")
        if not -1. <= command <= 1.:
            raise ValueError("Servo command must be in [-1,1]")
        return self.reference_angle_rad + self.half_range_rad * command

    @staticmethod
    def map_release(node_ids, release):
        ports = CyborgBidirectionalRotor.map_release(node_ids, release)
        ports["raw_command"] = ports["command"]
        return ports

    @staticmethod
    def _torque(angle, velocity, target):
        tau = MECHANICAL_TAU_S
        return INERTIA_KG_M2 * ((target - angle) / tau**2 - 2. * velocity / tau)

    @staticmethod
    def _free(angle, velocity, target, duration_s):
        """Analytic critical solution, with a stable small-time displacement."""
        h = duration_s / MECHANICAL_TAU_S
        decay = math.exp(-h)
        fraction = -math.expm1(math.log1p(h) - h)
        displacement = target - angle
        result_angle = angle + displacement * fraction + velocity * duration_s * decay
        result_velocity = (velocity * (1. - h)
            + displacement * h / MECHANICAL_TAU_S) * decay
        if not math.isfinite(result_angle) or not math.isfinite(result_velocity):
            raise FloatingPointError("Bounded servo free solution overflow")
        return result_angle, result_velocity

    def _first_contact(self, target, duration_s):
        angle, velocity = self.angle_rad, self.omega_rad_s
        if velocity == 0.:
            return None
        direction = 1. if velocity > 0. else -1.
        side = "upper" if direction > 0. else "lower"
        boundary = self.upper_limit_rad if direction > 0. else self.lower_limit_rad
        if angle == boundary:
            return 0., side, boundary, velocity
        coefficient = velocity + (angle - target) / MECHANICAL_TAU_S
        turn_s = MECHANICAL_TAU_S * velocity / coefficient if coefficient else -1.
        endpoint = min(duration_s, turn_s) if turn_s > 0. else duration_s
        extremum, end_velocity = self._free(angle, velocity, target, endpoint)
        distance = direction * (extremum - boundary)
        if distance < 0. or (distance == 0. and direction * end_velocity <= 0.):
            return None
        low, high = 0., endpoint
        for _ in range(64):
            midpoint = .5 * (low + high)
            if midpoint == low or midpoint == high:
                break
            position, _ = self._free(angle, velocity, target, midpoint)
            if direction * (position - boundary) >= 0.:
                high = midpoint
            else:
                low = midpoint
        _, before = self._free(angle, velocity, target, high)
        if direction * before <= 0.:
            raise FloatingPointError("Contact solver did not find an outward first crossing")
        return high, side, boundary, before

    def advance(self, command, dt_ns):
        self._validate()
        command = _scalar(command, "bounded servo command")
        target = self.target_angle(command)
        if type(dt_ns) is not int or dt_ns <= 0:
            raise ValueError("Duration must be a positive Python integer in ns")
        duration = dt_ns * 1.e-9
        if not math.isfinite(duration):
            raise ValueError("Mechanical duration must be finite")
        contact = self._first_contact(target, duration)
        impact = None
        if contact is None:
            angle, velocity = self._free(self.angle_rad, self.omega_rad_s, target, duration)
        else:
            offset, side, boundary, before = contact
            angle, velocity = self._free(boundary, 0., target, duration - offset)
            impact = dict(interval_start_ns=self.time_ns, interval_dt_ns=dt_ns,
                offset_s=offset, side=side, angle_rad=boundary,
                velocity_before_rad_s=before, impulse_nm_s=-INERTIA_KG_M2 * before,
                energy_loss_j=.5 * INERTIA_KG_M2 * before**2)
        torque = self._torque(angle, velocity, target)
        if (not self.lower_limit_rad <= angle <= self.upper_limit_rad
                or not math.isfinite(torque)):
            raise FloatingPointError("Invalid bounded servo solution; no clipping or rescue")
        self.angle_rad, self.omega_rad_s = angle, velocity
        self.last_command, self.last_torque_nm = command, torque
        self.time_ns += dt_ns
        if impact is not None:
            self.impact_count += 1
            self.last_impact = impact
            self.impact_energy_loss_total_j += impact["energy_loss_j"]
            self.impact_impulse_total_nm_s += impact["impulse_nm_s"]

    def _validate(self):
        for name in ("reference_angle_rad", "half_range_rad", "angle_rad", "omega_rad_s",
                     "last_command", "last_torque_nm", "impact_energy_loss_total_j",
                     "impact_impulse_total_nm_s"):
            _scalar(getattr(self, name), name)
        if (self.half_range_rad != HALF_RANGE_RAD
                or not self.lower_limit_rad <= self.angle_rad <= self.upper_limit_rad
                or not _clock(self.start_ns) <= _clock(self.adoption_time_ns) <= _clock(self.time_ns)):
            raise ValueError("Bounded servo range, pose or clocks differ")
        expected = self._torque(self.angle_rad, self.omega_rad_s, self.target_angle(self.last_command))
        if not math.isfinite(expected) or self.last_torque_nm != expected:
            raise ValueError("Saved continuous torque does not match the bounded physical state")
        previous = CyborgCenteredServo.from_state(self.adoption["previous_state"])
        reference, turns = _reference(previous)
        if (self.reference_angle_rad != reference or self.adoption["reference_turns"] != turns
                or self.adoption_time_ns != previous.time_ns or self.start_ns != previous.start_ns
                or not self.lower_limit_rad <= previous.angle_rad <= self.upper_limit_rad
                or not np.array_equal(self.initial_rotation, previous.initial_rotation)
                or not np.array_equal(self.pivot_mm, previous.pivot_mm)
                or any(not np.array_equal(self.initial_centers[s], previous.initial_centers[s]) for s in ("L", "R"))):
            raise ValueError("Bounded servo reference or geometry differs from its recorded predecessor")
        if self.time_ns == self.adoption_time_ns and (
                self.angle_rad != previous.angle_rad or self.omega_rad_s != previous.omega_rad_s):
            raise ValueError("Device adoption must preserve angle and angular velocity exactly")
        if type(self.impact_count) is not int or self.impact_count < 0 or self.impact_energy_loss_total_j < 0.:
            raise ValueError("Invalid cumulative impact state")
        if self.impact_count == 0:
            if (self.last_impact is not None or self.impact_energy_loss_total_j != 0.
                    or self.impact_impulse_total_nm_s != 0.):
                raise ValueError("A device without impacts cannot contain an impulse history")
        else:
            p = self.last_impact
            if not isinstance(p, dict) or set(p) != _IMPACT_KEYS or p["side"] not in ("lower", "upper"):
                raise ValueError("Invalid last mechanical impact")
            for name in ("offset_s", "angle_rad", "velocity_before_rad_s", "impulse_nm_s", "energy_loss_j"):
                _scalar(p[name], name)
            start, dt = _clock(p["interval_start_ns"]), _clock(p["interval_dt_ns"])
            boundary = self.lower_limit_rad if p["side"] == "lower" else self.upper_limit_rad
            direction = -1. if p["side"] == "lower" else 1.
            if (dt <= 0 or not self.adoption_time_ns <= start < start + dt <= self.time_ns
                    or not 0. <= p["offset_s"] <= dt * 1.e-9 or p["angle_rad"] != boundary
                    or direction * p["velocity_before_rad_s"] <= 0.
                    or p["impulse_nm_s"] != -INERTIA_KG_M2 * p["velocity_before_rad_s"]
                    or p["energy_loss_j"] != .5 * INERTIA_KG_M2 * p["velocity_before_rad_s"]**2
                    or self.impact_energy_loss_total_j < p["energy_loss_j"]):
                raise ValueError("Impact time, impulse or energy is inconsistent")

    def state_dict(self):
        self._validate()
        result = CyborgBidirectionalRotor.state_dict(self)
        result.update(schema=SCHEMA)
        for name in _EXTRA_KEYS:
            result[name] = copy.deepcopy(getattr(self, name))
        return result

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or set(state) != _BASE_KEYS | _EXTRA_KEYS or state["schema"] != SCHEMA:
            raise ValueError("Unknown or incomplete bounded servo state")
        base = CyborgRotor.from_pose(state["start_ns"], state["initial_rotation"], state["initial_centers"])
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        for name in ("time_ns", "angle_rad", "omega_rad_s", "last_command", "last_torque_nm"):
            setattr(obj, name, copy.deepcopy(state[name]))
        for name in _EXTRA_KEYS:
            setattr(obj, name, copy.deepcopy(state[name]))
        obj._validate()
        if (state["parameters"] != obj.parameters() or state["motor_ids"] != list(MOTOR_IDS)
                or not np.array_equal(state["pivot_mm"], obj.pivot_mm)):
            raise ValueError("Modified bounded servo device contract")
        return obj
