"""Full MaleCNS life with a declared direct-MN torque effector replacement.

The 27 MN IDs have concordant tibial muscle labels in MaleCNS and the saved
MANC crosswalk. This does not identify individual fast/slow motor units or a
physiological rate-to-force law. The body contains six memoryless torque
actuators, not reconstructed muscles. Odor/contact-to-ORN/PAM and tonic DNg100
inputs remain the reference experiment's explicit sensory/arousal scaffolds.
"""
from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import pandas as pd

from anatomical_rate_brain import AnatomicalRateBrain
from anatomical_plasticity import CandidateGammaPlasticity
from native_motor_body import NativeMotorBody, LEGS, MAX_TORQUE_NATIVE
from organism_session import OrganismSession, OdorPatchWorld, ROOT, SOURCES as REFERENCE_SOURCES
from session_io import sha256, write_state, read_state


SOURCES = tuple(REFERENCE_SOURCES) + ("native_motor_body.py", "native_organism_session.py")
MOTOR_IDS = {
    "LF": {"flexor": [807165, 809912, 818057, 909831], "extensor": [800636]},
    "RF": {"flexor": [810098, 821635, 1050189039], "extensor": [804257]},
    "LM": {"flexor": [802295, 818295, 823739, 927808], "extensor": [801234]},
    "RM": {"flexor": [804405, 805010, 816222, 903363], "extensor": [830514]},
    "LH": {"flexor": [908762, 1050100010], "extensor": [800621, 812953]},
    "RH": {"flexor": [803072, 827968], "extensor": [800158, 809935]},
}
STATE_KEYS = {"schema", "time_ns", "ticks", "config", "world", "body", "plasticity", "ports",
              "port_identity", "population_indices", "pending_sensors", "last_action", "motor_disconnected",
              "history", "used_sensors", "source_identity", "reference_provenance", "motor_ids"}


class NativeOrganismSession(OrganismSession):
    """Same neural/world continuity; new audited six-tibia torque body."""

    @classmethod
    def from_reference_checkpoint(cls, path):
        reference = OrganismSession.load(path)
        body = None
        try:
            if reference.time_ns != 0 or reference.ticks != 0:
                raise ValueError("Native v1 requires the reference initial checkpoint at t=0")
            body = NativeMotorBody(yaw=reference.body.constructor["yaw"], seed=reference.brain.seed)
            body.adopt_initial_reference(reference.body)
            self = cls()
            for key in ("brain", "plasticity", "world", "ports", "port_identity", "population_indices"):
                setattr(self, key, getattr(reference, key))
            self.body = body
            self.time_ns = self.ticks = 0
            self.failed = False
            self.motor_disconnected = False
            self.sensory_replay = None
            self.history, self.used_sensors = [], []
            self.last_action = np.zeros(6)
            self.config = dict(reference.config)
            self.config.update(candidate="ALL_MALECNS_27MN_6_TIBIA_TORQUE_v1",
                               motor_scale_native=MAX_TORQUE_NATIVE, activation_filter="none",
                               output_units="native MuJoCo torque; physiological calibration absent")
            self.motor_ids = json.loads(json.dumps(MOTOR_IDS))
            self._index_ports()
            self._index_motors()
            self.pending_sensors = self.world.sense(self.body.observe())
            if not np.array_equal(self.pending_sensors, reference.pending_sensors):
                raise ValueError("Effector replacement altered the initial geometric sensor observation")
            crosswalk = ROOT / "data/m1_identity/attempt_01/all_motor_target_crosswalk.parquet"
            table = pd.read_parquet(crosswalk).set_index("bodyId")
            for leg, groups in MOTOR_IDS.items():
                for role, ids in groups.items():
                    selected = table.loc[ids]
                    target = f"Ti {role}"
                    nerve = {"F": "ProLN", "M": "MesoLN", "H": "MetaLN"}[leg[1]]
                    if not (selected["type"].eq(target + " MN").all()
                            and selected.manc_s3_target.eq(target).all()
                            and selected.somaSide.eq(leg[0]).all()
                            and selected.exitNerve.eq(nerve).all()
                            and selected.manc_s3_exit_nerve.eq(f"{nerve}_{leg[0]}").all()
                            and selected.manc_s3_exact_id_match.all()
                            and selected.manc_binding_multiplicity.eq(1).all()):
                        raise ValueError(f"Motor mapping no longer has concordant anatomy: {leg}/{role}")
            self.reference_provenance = {
                "checkpoint": str(Path(path).resolve()), "manifest_sha256": sha256(Path(path) / "manifest.json"),
                "reference_source_identity": reference.source_identity,
                "reference_body_identity": reference.body.identity,
                "crosswalk_sha256": sha256(crosswalk),
                "operation": "Effector replacement at t=0; neural parameters/weights/state, plasticity and world preserved",
                "removed": ["CPG", "preprogrammed_joint_trajectories", "position_servos", "adhesion", "DN_to_behavior_decoder"],
                "remaining_hypotheses": ["rate_model_priors", "sensory_and_arousal_transduction", "candidate_plasticity",
                                         "normalized_MN_pool_difference_to_torque", "source_passive_mechanics"],
                "muscle_identity": "concordant muscle groups; no fast/slow unit assignment",
            }
            self.source_identity = {name: sha256(ROOT / "src" / name) for name in SOURCES}
            self._validate_clock()
            return self
        except BaseException:
            if body is not None:
                body.close()
            raise
        finally:
            reference.close()

    def _index_motors(self):
        if self.motor_ids != MOTOR_IDS:
            raise ValueError("Native v1 requires the recorded 27 concordant MN identities")
        self.motor_indices = {}
        for leg, groups in self.motor_ids.items():
            self.motor_indices[leg] = {}
            for role, ids in groups.items():
                idx = np.searchsorted(self.brain.node_ids, ids)
                if np.any(idx >= self.brain.n_neurons) or not np.array_equal(self.brain.node_ids[idx], ids):
                    raise ValueError("Motor identity absent from full brain")
                self.motor_indices[leg][role] = idx

    def _validate_clock(self):
        super()._validate_clock()
        if self.config["motor_scale_native"] != MAX_TORQUE_NATIVE or self.config["activation_filter"] != "none":
            raise ValueError("Unsupported native v1 torque configuration")
        if self.last_action.shape != (6,) or not np.isfinite(self.last_action).all() or np.any(np.abs(self.last_action) > MAX_TORQUE_NATIVE + 1e-12):
            raise ValueError("Invalid pending native torque")

    def step(self):
        if self.failed:
            raise ValueError("Failed native sessions cannot silently resume")
        self._validate_clock()
        sensory = self.pending_sensors.copy()
        if sensory.shape != (3,) or not np.isfinite(sensory).all() or np.any((sensory < 0) | (sensory > 1)):
            raise ValueError("Invalid pending environmental input")
        if self.sensory_replay is not None:
            raise ValueError("Native v1 does not accept unrecorded sensory replay state")
        drive = np.zeros(self.brain.n_neurons, dtype=np.float32)
        for i, side in enumerate(("L", "R")):
            drive[self.port_indices[f"ORN_DM1_{side}"]] = self.config["odor_drive"] * sensory[i]
            drive[self.port_indices[f"PAM08_{side}"]] = self.config["reward_drive"] * sensory[2]
            drive[self.port_indices[f"DNg100_{side}"]] = self.config["arousal_drive"]
        applied = self.last_action.copy()
        if self.motor_disconnected:
            applied[:] = 0.
        try:
            self.brain.step(self.CONTROL_NS * 1e-9, drive)
            self.plasticity.step(self.CONTROL_NS * 1e-9)
            pools = {}
            for leg, groups in self.motor_indices.items():
                pools[leg] = {role: float(np.mean(self.brain.rates[idx] / self.brain.r_max[idx]))
                              for role, idx in groups.items()}
            decoded = self.config["motor_scale_native"] * self.body.flexion_sign * np.array(
                [pools[leg]["flexor"] - pools[leg]["extensor"] for leg in LEGS])
            next_action = np.zeros(6) if self.motor_disconnected else decoded
            self.body.advance(applied, self.CONTROL_NS // round(self.body.dt * 1e9))
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.last_action = next_action
            self.used_sensors.append(sensory)
            row = {"time_s": self.time_ns * 1e-9, "position_mm": observation["position_mm"].tolist(),
                   "qpos": observation["qpos"].tolist(), "qvel": observation["qvel"].tolist(),
                   "yaw_rad": observation["yaw"], "upright_cos": observation["upright_cos"],
                   "fallen": bool(observation["upright_cos"] < 0), "contact_count": observation["contact_count"],
                   "sensors_used": sensory.tolist(), "sensors_pending": self.pending_sensors.tolist(),
                   "action": applied.tolist(), "next_action": next_action.tolist(), "decoded_torque": decoded.tolist(),
                   "motor_pools_normalized": pools,
                   "motor_neurons_hz": {str(int(self.brain.node_ids[i])): float(self.brain.rates[i])
                                        for groups in self.motor_indices.values() for idx in groups.values() for i in idx},
                   "motor_neurons_normalized": {str(int(self.brain.node_ids[i])): float(self.brain.rates[i] / self.brain.r_max[i])
                                                for groups in self.motor_indices.values() for idx in groups.values() for i in idx},
                   "interior_tibia_angles_rad": {leg: float(self.body._angle(self.body.scratch, leg)) for leg in LEGS},
                   "active_neurons": int(np.count_nonzero(self.brain.rates > 1)),
                   "mean_rate_hz": float(self.brain.rates.mean()), "max_rate_hz": float(self.brain.rates.max()),
                   "saturated_neurons": int(np.count_nonzero(self.brain.rates > .95 * self.brain.r_max)),
                   "max_actuator_force_native": observation["max_actuator_force_native"]}
            self.history.append(row)
            self._validate_clock()
            return row
        except BaseException:
            self.failed = True
            raise

    def save(self, path):
        self._validate_clock()
        if self.failed or self.source_identity != {name: sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Failed session or changed native source cannot be checkpointed")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving existing native checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
        try:
            self.brain.save_checkpoint(staging / "brain")
            state = {name: getattr(self, name) for name in STATE_KEYS - {"schema", "world", "body", "plasticity", "used_sensors"}}
            state.update(schema=1, world=self.world.__dict__, body=self.body.state_dict(),
                         plasticity=self.plasticity.state_dict(), used_sensors=np.asarray(self.used_sensors).reshape((-1, 3)))
            write_state(staging / "session", state)
            (staging / "source").mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT / "src" / name, staging / "source" / name)
            manifest = {"schema": "matrix_native_organism_session_v1", "time_ns": self.time_ns,
                        "neuron_count": self.brain.n_neurons, "stored_edges": self.brain.W.nnz,
                        "motor_neurons": 27, "torque_actuators": 6, "biological_validation": False,
                        "plasticity_enabled": self.plasticity.enabled,
                        "no_cpg_templates_position_servos_adhesion": True,
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
        if manifest.get("schema") != "matrix_native_organism_session_v1":
            raise ValueError("Unknown native organism checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Native checkpoint files missing or unrecognized")
        for name, digest in manifest["files"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"Native checkpoint integrity failed: {name}")
        state = read_state(path / "session")
        if set(state) != STATE_KEYS or state["schema"] != 1:
            raise ValueError("Unsupported native organism state")
        if state["source_identity"] != {name: sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Native checkpoint requires its archived source version")
        self = cls()
        self.brain = AnatomicalRateBrain.load_checkpoint(path / "brain")
        for key in STATE_KEYS - {"schema", "world", "body", "plasticity", "used_sensors"}:
            setattr(self, key, state[key])
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
            self._validate_clock()
            if manifest["time_ns"] != self.time_ns or len(self.history) != self.ticks or len(self.used_sensors) != self.ticks:
                raise ValueError("Native history and clock disagree")
            if not np.array_equal(self.pending_sensors, self.world.sense(self.body.observe())):
                raise ValueError("Native pending sensors differ from physical state/world")
            return self
        except BaseException:
            self.body.close()
            raise
