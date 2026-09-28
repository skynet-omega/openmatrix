"""Continue a native life with full-body floor contact, size and proprioception.

The anatomical IDs and all existing synaptic weights survive the explicit
model intervention. Four unresolved sensory polarity assignments are exposed,
never selected by whether they yield an attractive movement. Sensory currents
enter the existing recurrent graph; no proprioceptive signal commands a motor.
Old source modules are deliberately unchanged so their checkpoints still load.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numpy as np

from anatomical_rate_brain import AnatomicalRateBrain
from anatomical_plasticity import CandidateGammaPlasticity
from anatomical_morphometry import apply_morphometry
from anatomical_proprioception import AnatomicalProprioception
from native_motor_body import LEGS
from contact_complete_native_body import ContactCompleteNativeBody as NativeMotorBody
from native_organism_session import NativeOrganismSession, SOURCES as NATIVE_SOURCES, STATE_KEYS as NATIVE_KEYS
from organism_session import OdorPatchWorld, ROOT
from session_io import sha256, read_state, write_state


SOURCES = tuple(NATIVE_SOURCES) + ("anatomical_morphometry.py", "anatomical_proprioception.py",
                                  "contact_complete_native_body.py", "sensorimotor_contact_session.py")
SCHEMA = "matrix_sensorimotor_contact_session_v2"
STATE_KEYS = set(NATIVE_KEYS) | {"proprioception", "pending_proprioception", "proprioception_enabled",
                               "intervention", "branch_start_tick"}


def identical(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


class ContactSensorimotorSession(NativeOrganismSession):
    """An explicit intervention on an existing lifetime, with exact persistence."""

    @classmethod
    def from_native_checkpoint(cls, path, polarity, *, morphology=None, proprioception=True,
                               remove_tonic_and_patch_drive=True):
        parent = NativeOrganismSession.load(path)
        self = cls()
        self.__dict__.update(parent.__dict__)
        try:
            old_config = copy.deepcopy(self.config)
            before = {"rates": self.brain.rates.copy(), "weights": self.brain.W.data.copy(),
                      "body": self.body.state_dict(), "qpos": self.body.data.qpos.copy(),
                      "qvel": self.body.data.qvel.copy(), "body_time": self.body.data.time,
                      "plasticity": self.plasticity.state_dict(),
                      "world": copy.deepcopy(self.world.__dict__), "pending_action": self.last_action.copy(),
                      "time_ns": self.time_ns}
            former_body = self.body
            replacement_body = NativeMotorBody(**former_body.constructor)
            try:
                replacement_body.adopt_native(former_body)
            except BaseException:
                replacement_body.close()
                raise
            self.body = replacement_body
            former_body.close()
            self.intervention = {"parent_checkpoint": str(Path(path).resolve()),
                                 "parent_manifest_sha256": sha256(Path(path) / "manifest.json"),
                                 "time_ns": self.time_ns, "parent_source_identity": self.source_identity,
                                 "old_config": old_config,
                                 "operation": "Change F-I size scaling and attach a candidate sensory transduction at the recorded time; do not reset the life"}
            self.intervention["body_contact_replacement"] = copy.deepcopy(self.body.replacement)
            self.intervention["former_body_identity"] = before["body"]["identity"]
            self.intervention["new_body_identity"] = self.body.identity
            self.intervention["morphometry"] = apply_morphometry(self.brain, morphology)
            if type(proprioception) is not bool or type(remove_tonic_and_patch_drive) is not bool:
                raise ValueError("Sensory/scaffold flags must be boolean")
            self.proprioception_enabled = proprioception
            self.proprioception = AnatomicalProprioception(self.brain, self.body, polarity)
            self.pending_proprioception = self.proprioception.sample()
            self.branch_start_tick = self.ticks
            self.config = dict(self.config)
            self.config.update(candidate="ALL_MALECNS_SIZE_PROPRIOCEPTIVE_FULL_CONTACT_v2",
                               sensory_polarity_status="UNRESOLVED_EXPLICIT_HYPOTHESIS",
                               tonic_and_patch_drive_removed=remove_tonic_and_patch_drive)
            if remove_tonic_and_patch_drive:
                self.config.update(arousal_drive=0., reward_drive=0.)
            checks = {"neural_rates": identical(before["rates"], self.brain.rates),
                      "all_synaptic_weights": identical(before["weights"], self.brain.W.data),
                      "body_qpos": identical(before["qpos"], self.body.data.qpos),
                      "body_qvel": identical(before["qvel"], self.body.data.qvel),
                      "body_clock": before["body_time"] == self.body.data.time,
                      "plasticity_state": identical(before["plasticity"], self.plasticity.state_dict()),
                      "world_state": identical(before["world"], self.world.__dict__),
                      "pending_motor_command": identical(before["pending_action"], self.last_action),
                      "clock": before["time_ns"] == self.time_ns}
            self.intervention.update(preserved_fields=checks, new_config=copy.deepcopy(self.config),
                                     sensory_polarity=copy.deepcopy(polarity))
            if not all(checks.values()):
                raise ValueError("Model intervention silently altered an existing lifetime field")
            self.source_identity = {name: sha256(ROOT / "src" / name) for name in SOURCES}
            self._validate_clock()
            return self
        except BaseException:
            self.close()
            raise

    def _validate_clock(self):
        super()._validate_clock()
        if type(self.proprioception_enabled) is not bool or not 0 <= self.branch_start_tick <= self.ticks:
            raise ValueError("Invalid proprioceptive branch state")
        p = self.pending_proprioception
        if not isinstance(p, dict) or "drive" not in p:
            raise ValueError("Missing pending proprioception")
        drive = p["drive"]
        if drive.shape != (self.brain.n_neurons,) or drive.dtype != np.float32 or not np.isfinite(drive).all():
            raise ValueError("Invalid pending proprioceptive current")

    def step(self):
        if self.failed:
            raise ValueError("Failed sensorimotor sessions cannot silently resume")
        self._validate_clock()
        if self.sensory_replay is not None:
            raise ValueError("This session does not implement a replay encoder")
        sensory = self.pending_sensors.copy()
        if sensory.shape != (3,) or not np.isfinite(sensory).all() or np.any((sensory < 0) | (sensory > 1)):
            raise ValueError("Invalid pending environmental sensor")
        prop = self.pending_proprioception
        drive = prop["drive"].copy() if self.proprioception_enabled else np.zeros(self.brain.n_neurons, dtype=np.float32)
        for k, side in enumerate(("L", "R")):
            drive[self.port_indices[f"ORN_DM1_{side}"]] += self.config["odor_drive"] * sensory[k]
            drive[self.port_indices[f"PAM08_{side}"]] += self.config["reward_drive"] * sensory[2]
            drive[self.port_indices[f"DNg100_{side}"]] += self.config["arousal_drive"]
        applied = np.zeros(6) if self.motor_disconnected else self.last_action.copy()
        try:
            dt = self.CONTROL_NS * 1e-9
            self.brain.step(dt, drive)
            self.plasticity.step(dt)
            pools = {leg: {role: float(np.mean(self.brain.rates[idx] / self.brain.r_max[idx]))
                           for role, idx in groups.items()} for leg, groups in self.motor_indices.items()}
            decoded = self.config["motor_scale_native"] * self.body.flexion_sign * np.array(
                [pools[leg]["flexor"] - pools[leg]["extensor"] for leg in LEGS])
            next_action = np.zeros(6) if self.motor_disconnected else decoded
            self.body.advance(applied, self.CONTROL_NS // round(self.body.dt * 1e9))
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.pending_proprioception = self.proprioception.sample()
            self.last_action = next_action
            self.used_sensors.append(sensory)
            # No full-CNS trajectory dump: all rates are in the checkpoint;
            # history records identifiable causal ports and population summaries.
            row = dict(time_s=self.time_ns * 1e-9, position_mm=observation["position_mm"].tolist(),
                       qpos=observation["qpos"].tolist(), qvel=observation["qvel"].tolist(),
                       upright_cos=observation["upright_cos"], fallen=bool(observation["upright_cos"] < 0),
                       contact_count=observation["contact_count"], sensors_used=sensory.tolist(),
                       sensors_pending=self.pending_sensors.tolist(), action=applied.tolist(),
                       next_action=next_action.tolist(), decoded_torque=decoded.tolist(),
                       proprioception_enabled=self.proprioception_enabled,
                       proprioception_available=prop["normalized_afferent_drive"].tolist(),
                       proprioception_used=(prop["normalized_afferent_drive"] if self.proprioception_enabled
                                             else np.zeros_like(prop["normalized_afferent_drive"])).tolist(),
                       angles_used_rad=prop["angles_rad"].tolist(),
                       angular_velocity_used_rad_s=prop["angular_velocity_rad_s"].tolist(),
                       proprioception_pending=self.pending_proprioception["normalized_afferent_drive"].tolist(),
                       motor_pools_normalized=pools,
                       afferent_rates_model_hz=self.brain.rates[self.proprioception.indices].tolist(),
                       motor_neurons_hz={str(int(self.brain.node_ids[i])): float(self.brain.rates[i])
                                        for groups in self.motor_indices.values() for idx in groups.values() for i in idx},
                       ports_hz={name: float(self.brain.rates[idx].mean()) for name, idx in self.port_indices.items() if len(idx)},
                       active_neurons=int(np.count_nonzero(self.brain.rates > 1)),
                       mean_rate_hz=float(self.brain.rates.mean()),
                       saturated_neurons=int(np.count_nonzero(self.brain.rates > .95 * self.brain.r_max)),
                       candidate_factor_min=float(self.plasticity.factors.min()),
                       candidate_factor_mean=float(self.plasticity.factors.mean()),
                       max_actuator_force_native=observation["max_actuator_force_native"])
            self.history.append(row)
            self._validate_clock()
            return row
        except BaseException:
            self.failed = True
            raise

    def save(self, path):
        self._validate_clock()
        if self.failed or self.source_identity != {name: sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Failed session or changed source cannot be checkpointed")
        if not identical(self.pending_proprioception, self.proprioception.sample()):
            raise ValueError("Pending proprioception disagrees with the current body")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving existing checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
        try:
            self.brain.save_checkpoint(staging / "brain")
            special = {"schema", "world", "body", "plasticity", "used_sensors", "proprioception"}
            state = {name: getattr(self, name) for name in STATE_KEYS - special}
            state.update(schema=2, world=self.world.__dict__, body=self.body.state_dict(),
                         plasticity=self.plasticity.state_dict(), proprioception=self.proprioception.state_dict(),
                         used_sensors=np.asarray(self.used_sensors).reshape((-1, 3)))
            write_state(staging / "session", state)
            (staging / "source").mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT / "src" / name, staging / "source" / name)
            manifest = {"schema": SCHEMA, "time_ns": self.time_ns, "neuron_count": self.brain.n_neurons,
                        "stored_edges": self.brain.W.nnz, "no_cpg_templates_position_servos_adhesion": True,
                        "full_body_floor_contacts": True,
                        "proprioceptive_afferents": len(self.proprioception.indices),
                        "biological_validation": False, "sensory_polarity_resolved": False,
                        "files": {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown sensorimotor checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Incomplete sensorimotor checkpoint")
        for name, digest in manifest["files"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        state = read_state(path / "session")
        if set(state) != STATE_KEYS or state["schema"] != 2:
            raise ValueError("Unsupported sensorimotor state")
        if state["source_identity"] != {name: sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Checkpoint requires its archived source version")
        self = cls()
        self.brain = AnatomicalRateBrain.load_checkpoint(path / "brain")
        special = {"schema", "world", "body", "plasticity", "used_sensors", "proprioception"}
        for name in STATE_KEYS - special:
            setattr(self, name, state[name])
        self.world = OdorPatchWorld()
        if set(self.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete world state")
        self.world.__dict__.update(state["world"])
        self._index_ports()
        self._index_motors()
        self.plasticity = CandidateGammaPlasticity.from_state(self.brain, state["plasticity"])
        self.body = NativeMotorBody.from_state(state["body"])
        self.failed = False
        self.sensory_replay = None
        self.used_sensors = list(state["used_sensors"])
        try:
            self.proprioception = AnatomicalProprioception.from_state(self.brain, self.body, state["proprioception"])
            self._validate_clock()
            if manifest["time_ns"] != self.time_ns or len(self.history) != self.ticks or len(self.used_sensors) != self.ticks:
                raise ValueError("History and clocks disagree")
            if not np.array_equal(self.pending_sensors, self.world.sense(self.body.observe())):
                raise ValueError("Pending environmental input disagrees with body/world")
            if not identical(self.pending_proprioception, self.proprioception.sample()):
                raise ValueError("Pending proprioception disagrees with body")
            return self
        except BaseException:
            self.close()
            raise
