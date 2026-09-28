"""Anatomical FeCO ports with explicitly unresolved transduction polarity.

This module senses physical femur-tibia angles and velocities. It contains no
desired posture, gait phase, motor command, force, behavioral target or feedback
controller. Marin et al. (eLife.97766.1, Fig. 59) motivates the four cell types;
neither that source nor the local functional recordings determine tuning for
these individual MaleCNS neurons. Therefore both claw and hook polarities are
mandatory explicit hypotheses, not inferred from their predicted motor effects.

Angle encoding is the candidate linear fraction of the geometric [0, pi]
interior angle. Hook velocity encoding is rectified v/(v + 240 deg/s). The latter
scale anchors the hypothesis to a published leg-movement stimulus, not a measured
half-response speed. A maximum current of 80 matches the pre-existing ORN input
amplitude for a declared engineering comparison; it is not a biological current
or a firing-rate target. No rate is clamped and recurrent synapses remain live.
"""
from pathlib import Path
import copy
import hashlib
import json

import mujoco
import numpy as np

from native_motor_body import LEGS
from session_io import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config/anatomical_proprioception_ports.json"
MAX_CURRENT_MODEL_UNITS = 80.0
VELOCITY_HALF_RAD_S = np.deg2rad(240.0)
POLARITY_KEYS = {"claw_50", "hook_39"}


class AnatomicalProprioception:
    """Instantaneous candidate afferent transduction on 151 existing CNS IDs."""

    def __init__(self, brain, body, polarity, manifest_path=None):
        manifest_path = Path(manifest_path or DEFAULT_MANIFEST).resolve()
        manifest = json.loads(manifest_path.read_text())
        if sha256(manifest["source_path"]) != manifest["source_sha256"]:
            raise ValueError("Proprioception annotation source identity differs")
        origin = {"path": str(manifest_path), "file_sha256": sha256(manifest_path)}
        self._initialize(brain, body, polarity, manifest, origin)

    def _initialize(self, brain, body, polarity, manifest, origin):
        """Build from a validated acquisition or an embedded checkpoint record."""
        if (not isinstance(polarity, dict) or set(polarity) != POLARITY_KEYS
                or any(v not in ("flexion", "extension") for v in polarity.values())):
            raise ValueError("Explicit claw_50 and hook_39 polarity hypotheses are required")
        self.brain, self.body = brain, body
        self.polarity = dict(polarity)
        self.manifest = copy.deepcopy(manifest)
        self.manifest_origin = copy.deepcopy(origin)
        if self.manifest.get("schema") != "matrix_anatomical_proprioception_ports_v1":
            raise ValueError("Unknown anatomical proprioception port manifest")
        if self.manifest.get("canonical_release") != "MaleCNS v1.0":
            raise ValueError("Current canonical MaleCNS v1.0 annotations required")
        ports = self.manifest["ports"]
        nerves = {"ProLN": "F", "MesoLN": "M", "MetaLN": "H"}
        for port in ports:
            if (type(port["body_id"]) is not int or port["body_id"] <= 0
                    or port["type"] not in ("SNpp39", "SNpp41", "SNpp50", "SNpp51")
                    or port["manc_type"] != port["type"]
                    or port["root_side"] not in ("L", "R")
                    or port["entry_nerve"] not in nerves
                    or port["leg"] != port["root_side"] + nerves[port["entry_nerve"]]
                    or port["family"] != ("hook" if port["type"] in ("SNpp39", "SNpp41") else "claw")):
                raise ValueError("Invalid or anatomically inconsistent FeCO port")
        self.node_ids = np.array([p["body_id"] for p in ports], dtype=np.int64)
        if (not len(ports) or np.any(np.diff(self.node_ids) <= 0)
                or len(np.unique(brain.node_ids)) != len(brain.node_ids)
                or np.any(np.diff(brain.node_ids) <= 0)):
            raise ValueError("Sensory and brain identities must be uniquely sorted")
        self.indices = np.searchsorted(brain.node_ids, self.node_ids)
        if (np.any(self.indices >= len(brain.node_ids))
                or not np.array_equal(brain.node_ids[self.indices], self.node_ids)):
            raise ValueError("Afferent identities are missing from this canonical brain")
        self.leg_indices = np.array([LEGS.index(p["leg"]) for p in ports])
        self.cell_types = tuple(p["type"] for p in ports)
        if any(t not in ("SNpp39", "SNpp41", "SNpp50", "SNpp51") for t in self.cell_types):
            raise ValueError("Ambiguous or unsupported FeCO input type")
        self.joint_ids, self.landmarks = [], []
        for leg in LEGS:
            j = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"matrixfly/joint_{leg}Tibia")
            b = [mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY,
                                  f"matrixfly/{leg}{segment}")
                 for segment in ("Femur", "Tibia", "Tarsus1")]
            if (min([j] + b) < 0
                    or body.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Missing named femur-tibia geometry or non-hinge joint")
            self.joint_ids.append(j)
            self.landmarks.append(b)
        for array in (self.indices, self.node_ids, self.leg_indices):
            array.flags.writeable = False
        self.identity = self._identity()

    def _identity(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                             separators=(",", ":")).encode()).hexdigest()
        return {
            "embedded_manifest_sha256": digest(self.manifest),
            "manifest_acquisition": copy.deepcopy(self.manifest_origin),
            "runtime_mapping_sha256": digest({"node_ids": self.node_ids.tolist(),
                "indices": self.indices.tolist(), "leg_indices": self.leg_indices.tolist(),
                "cell_types": self.cell_types, "joint_ids": self.joint_ids,
                "landmarks": self.landmarks}),
            "source_sha256": sha256(__file__),
            "brain_node_ids_sha256": hashlib.sha256(
                np.asarray(self.brain.node_ids, dtype="<i8").tobytes()).hexdigest(),
            "physical_model_sha256": self.body.identity["model_sha256"],
            "polarity_hypothesis": dict(self.polarity),
            "leg_order": list(LEGS),
            "amplitude_model_current": MAX_CURRENT_MODEL_UNITS,
            "velocity_half_rad_s": float(VELOCITY_HALF_RAD_S),
            "formula": "claw:theta/pi or 1-theta/pi; hook:max(0,+/-omega)/(max(0,+/-omega)+vhalf); drive=80*encoded",
            "biological_tuning_validated": False,
            "history_or_filter": "none; session must persist the pending sample",
            "geometry": "interior femur-tibia angle projected normal to tibial hinge; derivative from current oriented geometry times hinge qvel",
        }

    def encode(self, angles_rad, angular_velocity_rad_s):
        """Pure candidate transduction, also usable for instrument verification."""
        angles = np.asarray(angles_rad, dtype=float)
        velocity = np.asarray(angular_velocity_rad_s, dtype=float)
        if (angles.shape != (6,) or velocity.shape != (6,)
                or not np.isfinite(angles).all() or not np.isfinite(velocity).all()
                or np.any((angles < 0) | (angles > np.pi))):
            raise ValueError("Six finite interior angles [0,pi] and angular velocities required")
        extension = angles / np.pi
        flexion = 1.0 - extension
        extension_motion = np.maximum(0., velocity)
        flexion_motion = np.maximum(0., -velocity)
        extension_motion /= extension_motion + VELOCITY_HALF_RAD_S
        flexion_motion /= flexion_motion + VELOCITY_HALF_RAD_S
        claw = {"extension": extension, "flexion": flexion}
        hook = {"extension": extension_motion, "flexion": flexion_motion}
        opposite = {"extension": "flexion", "flexion": "extension"}
        signals = {"SNpp50": claw[self.polarity["claw_50"]],
                   "SNpp51": claw[opposite[self.polarity["claw_50"]]],
                   "SNpp39": hook[self.polarity["hook_39"]],
                   "SNpp41": hook[opposite[self.polarity["hook_39"]]]}
        normalized = np.array([signals[t][leg] for t, leg in
                               zip(self.cell_types, self.leg_indices)], dtype=np.float32)
        drive = np.zeros(len(self.brain.node_ids), dtype=np.float32)
        drive[self.indices] = np.float32(MAX_CURRENT_MODEL_UNITS) * normalized
        return {"drive": drive, "angles_rad": angles.copy(),
                "angular_velocity_rad_s": velocity.copy(),
                "normalized_afferent_drive": normalized}

    def sample(self):
        """Observe current physical state; never alter body or neuron dynamics."""
        self.body.observe()  # Reconstruct fresh geometry on separate MjData.
        view = self.body.scratch
        angles, velocities = [], []
        for j, (femur, tibia, tarsus) in zip(self.joint_ids, self.landmarks):
            axis = view.xaxis[j]
            u = view.xpos[femur] - view.xpos[tibia]
            v = view.xpos[tarsus] - view.xpos[tibia]
            u, v = u - np.dot(u, axis) * axis, v - np.dot(v, axis) * axis
            norm = np.linalg.norm(u) * np.linalg.norm(v)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Degenerate proprioceptive geometric landmarks")
            cosine = np.clip(np.dot(u, v) / norm, -1., 1.)
            sine = np.dot(axis, np.cross(u, v)) / norm
            angle = np.arctan2(abs(sine), cosine)
            qvel = view.qvel[self.body.model.jnt_dofadr[j]]
            if abs(sine) < 1e-10 and abs(qvel) > 1e-10:
                raise ValueError("Interior angle derivative undefined at a collinear hinge")
            angles.append(angle)
            velocities.append(np.sign(sine) * qvel)
        return self.encode(angles, velocities)

    def state_dict(self):
        current = self._identity()
        if current != self.identity:
            raise ValueError("Proprioception source, parameters or identity changed")
        return {"schema": 1, "manifest": copy.deepcopy(self.manifest),
                "manifest_origin": copy.deepcopy(self.manifest_origin),
                "identity": copy.deepcopy(current), "polarity": dict(self.polarity),
                "node_ids": self.node_ids.copy(), "indices": self.indices.copy()}

    @classmethod
    def from_state(cls, brain, body, state):
        keys = {"schema", "manifest", "manifest_origin", "identity", "polarity", "node_ids", "indices"}
        if set(state) != keys or state["schema"] != 1:
            raise ValueError("Unknown proprioception checkpoint state")
        sensor = cls.__new__(cls)
        sensor._initialize(brain, body, state["polarity"], state["manifest"], state["manifest_origin"])
        if (sensor.identity != state["identity"]
                or not np.array_equal(sensor.node_ids, state["node_ids"])
                or not np.array_equal(sensor.indices, state["indices"])):
            raise ValueError("Proprioceptive checkpoint identity or mapping differs")
        return sensor
