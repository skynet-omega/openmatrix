"""Read contact impulses/slip after each existing, single MuJoCo step.

The model uses grams and centimetres: native force -> N is 1e-5 and
cm -> micrometres is 1e4. No callback, mj_forward, force or state write.
Forces are the solver's retained contact forces for the completed step.
Slip uses post-step qvel and that step's retained contact/Jacobian geometry;
it is an integrated speed diagnostic, not a solver-error estimate or a
tracked material contact point. Reset the window before retrying a rolled
back organism interval; observation counters are external to the organism.
"""
from __future__ import annotations

import operator
import re

import mujoco as mj
import numpy as np


LEG_ORDER = ("LF", "RF", "LM", "RM", "LH", "RH")
GROUP_ORDER = ("abdomen", "legs", "other")
FORCE_TO_N = 1e-5
CM_TO_UM = 1e4


class MechanicalObserver:
    """Context manager; reset(millisecond=k), step body, then read().

read() returns independent JSON-compatible values without resetting.
The module-level step wrapper only observes this exact model/data pair.
Target calls with nstep != 1 are rejected before calling MuJoCo.
"""

    def __init__(self, body, *, ground_geom_ids=None):
        self.body, self.model, self.data = body, body.model, body.data
        if ground_geom_ids is None:
            ground_geom_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")]
        self.ground_geom_ids = frozenset(operator.index(g) for g in ground_geom_ids)
        if not self.ground_geom_ids or any(
            g < 0 or g >= self.model.ngeom or self.model.geom_bodyid[g] != 0
            for g in self.ground_geom_ids
        ):
            raise ValueError("Specify existing, world-fixed ground geoms")
        self._groups = np.full(self.model.nbody, 2, dtype=int)
        self._legs = np.full(self.model.nbody, -1, dtype=int)
        for bid in range(1, self.model.nbody):
            parent = bid
            while parent:
                name = self.model.body(parent).name or ""
                match = re.search(r"T([123])_(left|right)(?:$|_)", name)
                if match:
                    self._groups[bid] = 1
                    self._legs[bid] = 2 * (int(match[1]) - 1) + (match[2] == "right")
                    break
                if name.startswith("abdomen"):
                    self._groups[bid] = 0
                    break
                parent = int(self.model.body_parentid[parent])
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._force = np.zeros(6)
        self._original_step = None
        self._wrapper = None
        self.reset()

    def reset(self, millisecond=None):
        """Reset only external counters; never read/alter integration state."""
        self.millisecond = millisecond
        self.duration_s = 0.0
        self.sample_steps = 0
        self.active_ground_contact_samples = 0
        self.vertical_impulse_Ns = np.zeros(3)
        self.normal_impulse_Ns = np.zeros(3)
        self.world_impulse_Ns = np.zeros((3, 3))
        self.slip_um = np.zeros(6)

    def start(self):
        if self._original_step is not None:
            raise RuntimeError("Observer is already installed")
        original = mj.mj_step

        def observed_step(model, data, *args, **kwargs):
            if model is not self.model or data is not self.data:
                return original(model, data, *args, **kwargs)
            if len(args) > 1 or set(kwargs) - {"nstep"} or (args and "nstep" in kwargs):
                raise TypeError("Expected mj_step(model, data, nstep=1)")
            nstep = args[0] if args else kwargs.get("nstep", 1)
            if operator.index(nstep) != 1:
                raise ValueError("Observation requires single mj_step calls; no state advanced")
            before = float(data.time)
            result = original(model, data, *args, **kwargs)
            elapsed = float(data.time) - before
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Completed step has no positive finite elapsed time")
            self._sample(elapsed)
            return result

        self._original_step = original
        self._wrapper = observed_step
        mj.mj_step = observed_step
        return self

    def _origin_velocity(self, bid, cache):
        if bid not in cache:
            # Explicit xpos avoids confusing body origin with its inertial COM.
            mj.mj_jac(self.model, self.data, self._jacp, self._jacr,
                      self.data.xpos[bid], bid)
            cache[bid] = (self._jacp @ self.data.qvel, self._jacr @ self.data.qvel)
        return cache[bid]

    def _point_velocity(self, bid, point, cache):
        linear, angular = self._origin_velocity(bid, cache)
        return linear + np.cross(angular, point - self.data.xpos[bid])

    def _sample(self, dt):
        world_force = np.zeros((3, 3))
        normal_force = np.zeros(3)
        peak_slip_cm_s = np.zeros(6)
        velocities = {}
        contacts = 0
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            floor1, floor2 = g1 in self.ground_geom_ids, g2 in self.ground_geom_ids
            if floor1 == floor2 or contact.efc_address < 0:
                continue
            bid = int(self.model.geom_bodyid[g2 if floor1 else g1])
            if bid == 0:
                continue
            group, leg = int(self._groups[bid]), int(self._legs[bid])
            frame = contact.frame.reshape(3, 3)
            mj.mj_contactForce(self.model, self.data, ci, self._force)
            # Contact wrench acts on geom2. Reverse if geom2 is the floor.
            force_on_fly = frame.T @ self._force[:3]
            if floor2:
                force_on_fly = -force_on_fly
            world_force[group] += force_on_fly * FORCE_TO_N
            normal_force[group] += max(0.0, float(self._force[0])) * FORCE_TO_N
            contacts += 1
            if leg >= 0:
                velocity = self._point_velocity(bid, contact.pos, velocities)
                # World-fixed floor has zero velocity. Contact frame rows 1/2
                # span its tangent plane, independent of normal orientation.
                speed = float(np.linalg.norm(frame[1:] @ velocity))
                peak_slip_cm_s[leg] = max(peak_slip_cm_s[leg], speed)
        self.world_impulse_Ns += world_force * dt
        self.vertical_impulse_Ns += world_force[:, 2] * dt
        self.normal_impulse_Ns += normal_force * dt
        self.slip_um += peak_slip_cm_s * dt * CM_TO_UM
        self.duration_s += dt
        self.sample_steps += 1
        self.active_ground_contact_samples += contacts

    def read(self):
        return dict(
            millisecond=self.millisecond,
            group_order=list(GROUP_ORDER), leg_order=list(LEG_ORDER),
            vertical_impulse_Ns=self.vertical_impulse_Ns.tolist(),
            normal_impulse_Ns=self.normal_impulse_Ns.tolist(),
            world_impulse_Ns=self.world_impulse_Ns.tolist(),
            slip_um=self.slip_um.tolist(), duration_s=self.duration_s,
            sample_steps=self.sample_steps,
            active_ground_contact_samples=self.active_ground_contact_samples,
            mgN=float(self.model.body_mass.sum() * np.linalg.norm(self.model.opt.gravity) * FORCE_TO_N),
            force_unit_N_per_native=FORCE_TO_N,
            slip_definition="Integral of maximum tangential contact-point speed per leg per step",
            velocity_sampling="Post-step qvel with retained contact-step geometry",
        )

    def close(self):
        if self._original_step is None:
            return
        if mj.mj_step is not self._wrapper:
            raise RuntimeError("Another wrapper replaced mj_step; close observers in reverse order")
        mj.mj_step = self._original_step
        self._original_step = self._wrapper = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
