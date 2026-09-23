"""A continuable, explicitly experimental light-CNS-contractile-body loop.

The inherited life is never reset. Adopting new equations/optical geometry and
new zero-initialized muscle/phototransduction variables is a recorded model
intervention, not a claim that the old checkpoint already contained them.
"""
from pathlib import Path
import copy
import json
import math
import shutil
import tempfile

import numpy as np
import pandas as pd
import numba

from anatomical_rate_brain import AnatomicalRateBrain
from anatomical_plasticity import CandidateGammaPlasticity
from anatomical_proprioception import AnatomicalProprioception
from contact_complete_native_body import ContactCompleteNativeBody
from native_motor_body import LEGS
from native_organism_session import NativeOrganismSession
from organism_session import OdorPatchWorld, ROOT
from sensorimotor_contact_session import ContactSensorimotorSession, SOURCES as PARENT_SOURCES, identical
from retinal_world import CompoundEye, LuminousWorld
from hybrid_visual_brain import HybridVisualBrain
from contractile_tibia import ContractileTibia
from session_io import sha256, read_state, write_state


SCHEMA = "matrix_visuomotor_session_v1"
SOURCES = tuple(PARENT_SOURCES) + ("retinal_world.py", "visual_sparse_kernel.py", "hybrid_visual_brain.py", "contractile_tibia.py", "visuomotor_session.py")
SCALARS = {"time_ns", "start_ns", "ticks", "config", "ports", "motor_ids", "population_indices",
           "pending_sensors", "pending_proprioception", "pending_light", "pending_excitation",
           "history", "source_identity", "parent_archive", "intervention", "initial_camera",
           "visual_groups", "mode", "proprioception_enabled", "initial_plasticity_updates"}


class VisuomotorSession:
    CONTROL_NS = 1_000_000
    MODES = ("live", "dark", "motor_cut", "stationary", "camera_replay", "visual_output_cut", "tight_numerics")
    _index_ports = NativeOrganismSession._index_ports
    _index_motors = NativeOrganismSession._index_motors

    @classmethod
    def from_contact(cls, path, mode="live"):
        if mode not in cls.MODES:
            raise ValueError("Unknown predefined visuomotor intervention")
        parent = ContactSensorimotorSession.load(path)
        obj = cls()
        try:
            for name in ("brain", "plasticity", "body", "world", "proprioception", "ports", "motor_ids",
                         "population_indices", "pending_sensors", "pending_proprioception", "proprioception_enabled"):
                setattr(obj, name, getattr(parent, name))
            if parent.config["arousal_drive"] != 0. or parent.config["reward_drive"] != 0.:
                raise ValueError("This candidate requires prior tonic and patch-reward scaffolds already removed")
            obj.parent_archive = read_state(Path(path) / "session")
            obj.start_ns = obj.time_ns = parent.time_ns
            obj.initial_plasticity_updates = obj.plasticity.update_count
            obj.ticks = 0
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(control_ns=cls.CONTROL_NS, motor_delay_ns=cls.CONTROL_NS,
                              candidate="MALECNS_GRADED_VISION_CONTRACTILE_CANDIDATE_v1",
                              activation_filter="12 effective contractile elements, uncalibrated",
                              state_precision="authoritative float64 neural state; canonical float32 stored weights",
                              numba_version=numba.__version__, sparse_accumulation="float64 CSR row order; 8 independent row threads; fastmath=False")
            obj.mode = mode
            obj.history, obj.used_light = [], []
            obj.failed = False
            obj._index_ports()
            obj._index_motors()
            obj.eyes = CompoundEye(obj.brain, obj.body, ROOT / "data/anatomical_vision/optics_candidate_v1")
            table = pd.read_parquet(ROOT / "data/anatomical_vision/v0.1/visual_neurons.parquet").sort_values("bodyId")
            old_rates = obj.brain.rates.copy()
            obj.hybrid = HybridVisualBrain(obj.brain, table.bodyId.to_numpy(np.int64), obj.eyes.ids,
                                          rtol=1e-6 if mode == "tight_numerics" else 1e-5,
                                          atol=1e-8 if mode == "tight_numerics" else 1e-7)
            obj.hybrid.visual_output_connected = mode != "visual_output_cut"
            obj.hybrid.sync_plastic_weights(obj.plasticity)
            obj.visual_groups = {}
            for group in ("R1-R6", "R7", "R8", "L1", "L2", "L3", "Mi1", "Tm3", "T4", "T5", "HS", "VS", "LPLC2"):
                select = table.type.str.startswith(group, na=False) if group in ("R7", "R8", "T4", "T5", "HS", "VS") else table.type.eq(group)
                obj.visual_groups[group] = np.searchsorted(obj.brain.node_ids, table.loc[select, "bodyId"].to_numpy(np.int64))
            obj.light_world = LuminousWorld(obj.start_ns)
            if mode == "stationary":
                obj.light_world.angular_speed_rad_s = 0.
            obj.initial_camera = obj.eyes.pose()
            obj.muscles = ContractileTibia(obj.body)
            obj.pending_excitation = obj.motor_excitation()
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            obj.intervention = dict(parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/"manifest.json"), time_ns=obj.time_ns,
                preserved="all canonical neuron IDs, topology, weights, old parameters, body integration state, RNGs, world and plasticity",
                new_state_initialization="graded voltage consistent with prior normalized rate up to roundoff; phototransduction fast/adaptation=0; muscle activation=0",
                prior_rates_to_view_max_abs=float(np.max(np.abs(obj.brain.rates.astype(float)-old_rates))),
                prior_rates_to_view_changed=int(np.count_nonzero(obj.brain.rates != old_rates)),
                prior_pending_torque=parent.last_action.copy(),
                new_motor_boundary="prior direct-torque pending command explicitly replaced by MN excitation and initially relaxed contractile elements",
                optical_mapping="UNVALIDATED cross-individual registration; 5529 receptors, others receive no light drive",
                biological_status="UNVALIDATED; graded equation assigned to all annotated visual cells is a model simplification",
                acquisition_hashes={"visual_neurons": sha256(ROOT / "data/anatomical_vision/v0.1/visual_neurons.parquet"),
                                    "optics": sha256(ROOT / "data/anatomical_vision/optics_candidate_v1/manifest.json")})
            obj.source_identity = {name: sha256(ROOT / "src" / name) for name in SOURCES}
            obj._validate()
            return obj
        except BaseException:
            parent.close()
            raise

    def motor_excitation(self):
        release = self.hybrid.release()
        return np.array([[float(release[self.motor_indices[leg][role]].mean())
                          for role in ("flexor", "extensor")] for leg in LEGS])

    @property
    def last_action(self):
        return self.muscles.last_torque

    def _validate(self):
        if self.failed or self.mode not in self.MODES:
            raise ValueError("Failed or unsupported visuomotor session")
        clocks = [self.brain.time_ns, self.hybrid.time_ns, self.world.time_ns, self.plasticity.time_ns,
                  self.body.steps*round(self.body.dt*1e9), self.muscles.time_ns,
                  self.start_ns+self.ticks*self.CONTROL_NS]
        if any(t != self.time_ns for t in clocks) or self.plasticity.update_count != self.initial_plasticity_updates+self.ticks:
            raise ValueError("Visual, neural, plastic, contractile, body/world clocks disagree")
        if len(self.history) != self.ticks or len(self.used_light) != self.ticks:
            raise ValueError("Incomplete visuomotor trace")
        if self.pending_light.shape != (len(self.eyes.ids),) or not np.isfinite(self.pending_light).all():
            raise ValueError("Invalid pending retinal frame")
        if self.pending_excitation.shape != (6, 2) or not np.isfinite(self.pending_excitation).all() or np.any((self.pending_excitation < 0) | (self.pending_excitation > 1)):
            raise ValueError("Invalid pending neuromuscular signal")

    def step(self):
        self._validate()
        light = self.pending_light.copy()
        if self.mode == "dark":
            light[:] = 0.
        elif self.mode == "camera_replay":
            # Deterministic replay from the recorded initial camera. The moving
            # world persists, while the current body's image cannot enter.
            light = self.eyes.sample(self.light_world, self.time_ns, pose=self.initial_camera)
        drive = self.pending_proprioception["drive"].astype(float) if self.proprioception_enabled else np.zeros(self.brain.n_neurons)
        for k, side in enumerate(("L", "R")):
            drive[self.port_indices[f"ORN_DM1_{side}"]] += self.config["odor_drive"]*self.pending_sensors[k]
        excitation = self.pending_excitation.copy()
        if self.mode == "motor_cut":
            excitation[:] = 0.
        try:
            self.hybrid.advance(self.CONTROL_NS, drive, light)
            self.plasticity.step(self.CONTROL_NS*1e-9)
            self.hybrid.sync_plastic_weights(self.plasticity)
            next_excitation = self.motor_excitation()
            physical_dt_ns = round(self.body.dt*1e9)
            for _ in range(self.CONTROL_NS//physical_dt_ns):
                torque = self.muscles.advance(excitation, physical_dt_ns)
                self.body.advance(torque, 1)
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.pending_proprioception = self.proprioception.sample()
            self.pending_light = self.eyes.sample(self.light_world, self.time_ns)
            self.pending_excitation = next_excitation
            release = self.hybrid.release()
            frozen = self.eyes.sample(self.light_world, self.time_ns, pose=self.initial_camera)
            rotation, centers = self.eyes.pose()
            self.used_light.append(light)
            row = dict(time_s=self.time_ns*1e-9, position_mm=observation["position_mm"].tolist(),
                qpos=observation["qpos"].tolist(), qvel=observation["qvel"].tolist(),
                contact_count=observation["contact_count"], upright_cos=observation["upright_cos"],
                mn_excitation=next_excitation.tolist(), muscle_activation=self.muscles.activation.tolist(),
                muscle_force_native=self.muscles.last_force.tolist(), torque_native=torque.tolist(),
                retinal_used_mean=float(light.mean()), retinal_pending_mean=float(self.pending_light.mean()),
                retinal_body_motion_rms=float(np.sqrt(np.mean((self.pending_light-frozen)**2))),
                head_rotation=rotation.tolist(), eye_centers_mm={k:v.tolist() for k,v in centers.items()},
                visual_release={name:dict(count=len(idx), mean=float(release[idx].mean()),
                                         maximum=float(release[idx].max())) for name,idx in self.visual_groups.items() if len(idx)},
                active_release_above_point01=int(np.count_nonzero(release > .01)),
                nonvisual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.ri] > .95)),
                visual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.vi] > .95)),
                plastic_factor_min=float(self.plasticity.factors.min()),
                numeric_accepted=self.hybrid.statistics["accepted"], numeric_rejected=self.hybrid.statistics["rejected"])
            self.history.append(row)
            self._validate()
            return row
        except BaseException:
            self.failed = True
            raise

    def advance(self, seconds):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0 or not math.isclose(seconds*1e9/self.CONTROL_NS, round(seconds*1e9/self.CONTROL_NS), abs_tol=1e-9, rel_tol=0):
            raise ValueError("Continuation requires a nonnegative whole number of 1ms ticks")
        for _ in range(round(seconds*1e9/self.CONTROL_NS)):
            self.step()

    def state_dict(self):
        self._validate()
        result = {name: copy.deepcopy(getattr(self, name)) for name in SCALARS}
        result.update(schema=SCHEMA, world=copy.deepcopy(self.world.__dict__), body=self.body.state_dict(),
                      hybrid=self.hybrid.state_dict(), eyes=self.eyes.state_dict(), light_world=self.light_world.state_dict(),
                      muscles=self.muscles.state_dict(), plasticity=self.plasticity.state_dict(),
                      proprioception=self.proprioception.state_dict(),
                      used_light=np.asarray(self.used_light, dtype=np.float64).reshape((-1, len(self.eyes.ids))))
        return result

    def _validate_pending(self):
        if not identical(self.pending_proprioception, self.proprioception.sample()):
            raise ValueError("Pending proprioception disagrees with body")
        if not np.array_equal(self.pending_sensors, self.world.sense(self.body.observe())):
            raise ValueError("Pending odor disagrees with world/body")
        if not np.array_equal(self.pending_light, self.eyes.sample(self.light_world, self.time_ns)):
            raise ValueError("Pending retinal frame disagrees with world/body")

    def save(self, path):
        self._validate_pending()
        if self.source_identity != {name:sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Runtime source changed after session creation")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving checkpoint {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="."+path.name+"-", dir=path.parent))
        try:
            self.brain.save_checkpoint(staging / "brain")
            write_state(staging / "session", self.state_dict())
            (staging / "source").mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT / "src" / name, staging / "source" / name)
            manifest = dict(schema=SCHEMA, time_ns=self.time_ns, neuron_count=self.brain.n_neurons,
                            stored_edges=self.brain.W.nnz, mapped_photoreceptors=len(self.eyes.ids),
                            biological_validation=False, mode=self.mode,
                            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()})
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
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
            raise ValueError("Unsupported visuomotor checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Incomplete visuomotor checkpoint")
        for name, digest in manifest["files"].items():
            if sha256(path/name) != digest:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        state = read_state(path / "session")
        special = {"schema", "world", "body", "hybrid", "eyes", "light_world", "muscles", "plasticity", "proprioception", "used_light"}
        if set(state) != SCALARS | special or state["schema"] != SCHEMA:
            raise ValueError("Unsupported or incomplete visuomotor state")
        if state["source_identity"] != {name:sha256(ROOT / "src" / name) for name in SOURCES}:
            raise ValueError("Checkpoint requires its archived runtime source version")
        obj = cls()
        obj.brain = AnatomicalRateBrain.load_checkpoint(path / "brain")
        for key in SCALARS:
            setattr(obj, key, state[key])
        if obj.config["numba_version"] != numba.__version__:
            raise ValueError("Checkpoint requires its recorded numerical runtime")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        obj.hybrid = HybridVisualBrain.from_state(obj.brain, state["hybrid"])
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.body = ContactCompleteNativeBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.eyes = CompoundEye.from_state(obj.brain, obj.body, state["eyes"])
            obj.light_world = LuminousWorld.from_state(state["light_world"])
            obj.muscles = ContractileTibia.from_state(obj.body, state["muscles"])
            obj.used_light = list(state["used_light"])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            if manifest["time_ns"] != obj.time_ns:
                raise ValueError("Manifest clock differs")
            return obj
        except BaseException:
            obj.body.close()
            raise

    def close(self):
        self.body.close()
