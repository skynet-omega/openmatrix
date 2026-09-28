"""Continuing whole-CNS preparation coupled to an explicit engineering rotor.

The six-port arithmetic readout, torque law and detached optical mount belong
to the prosthesis. They are not a reconstructed neck, a neural sign assignment
or an animal ability. The inherited CNS, body and plasticity continue intact.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import tempfile

import numba
import numpy as np

from anatomical_rate_brain import AnatomicalRateBrain, ARRAY_FIELDS
from anatomical_plasticity import CandidateGammaPlasticity
from anatomical_proprioception import AnatomicalProprioception
from contractile_tibia import ContractileTibia
from cyborg_rotor import CyborgRotor, CyborgEye
from cyborg_visual_world import CyborgVisualWorld
from electrical_parameter_contract import guard_visual_session
from gpu_graded_descending_brain import GpuGradedDescendingBrain
from graded_descending_brain import GradedDescendingBrain
from gpu_visual_brain import GpuVisualBrain
from measured_visual_session import MeasuredVisualSession, SOURCES as PARENT_SOURCES
from organism_session import OdorPatchWorld, ROOT
from refined_contact_body import RefinedContactBody
from session_io import sha256, read_state, write_state
from visuomotor_session import SCALARS
from visual_descending_session import VisualDescendingSession


SCHEMA = "matrix_cyborg_visual_session_v1"
SOURCES = tuple(PARENT_SOURCES) + (
    "cyborg_rotor.py", "cyborg_visual_world.py", "cyborg_visual_session.py")
CYBORG_SCALARS = {"pending_cyborg_command", "output_connected"}


def _digest(value):
    """Hash exact scientific state without retaining a second whole checkpoint."""
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError("Object array in preserved scientific state")
            h.update(b"array" + x.dtype.str.encode() + repr(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, np.generic):
            visit(x.item())
        elif isinstance(x, dict):
            h.update(b"dict")
            for key in sorted(x):
                visit(key)
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(b"sequence" + str(len(x)).encode())
            for item in x:
                visit(item)
        else:
            h.update(type(x).__name__.encode() + repr(x).encode() + b"\0")
    visit(value)
    return h.hexdigest()


def preservation_fingerprints(session):
    """Exclude only the declared optical migration and numerical tolerances."""
    hybrid = session.hybrid.state_dict()
    hybrid["parameters"] = dict(hybrid["parameters"])
    for key in ("rtol", "atol"):
        hybrid["parameters"].pop(key, None)
    inherited = SCALARS - {"config", "source_identity", "intervention", "initial_camera", "pending_light"}
    return {
        "neural_arrays": _digest({n: getattr(session.brain, n) for n in ARRAY_FIELDS}),
        "topology_and_weights": _digest((session.brain.W.shape, session.brain.W.indptr,
            session.brain.W.indices, session.brain.W.data)),
        "neural_rng": _digest(session.brain.rng.bit_generator.state),
        "hybrid_except_solver_tolerances": _digest(hybrid),
        "plasticity": _digest(session.plasticity.state_dict()),
        "body": _digest(session.body.state_dict()),
        "muscles": _digest(session.muscles.state_dict()),
        "proprioception": _digest(session.proprioception.state_dict()),
        "optical_mapping": _digest(session.eyes.state_dict()),
        "inherited_clocks_history_and_pending": _digest({n: getattr(session, n) for n in inherited}),
        "odor_world": _digest(session.world.__dict__),
        "used_retinal_history": _digest(session.used_light),
    }


class CyborgVisualSession(MeasuredVisualSession):
    @classmethod
    def from_parent(cls, path, condition="down", speed_deg_s=60., output_connected=True, tight=False):
        if type(output_connected) is not bool or type(tight) is not bool:
            raise ValueError("Explicit boolean output connection and solver comparison required")
        path = Path(path)
        manifest = json.loads((path/"manifest.json").read_text())
        parents = {"matrix_visual_descending_session_v1": VisualDescendingSession,
                   "matrix_measured_visual_session_v1": MeasuredVisualSession}
        if manifest.get("schema") not in parents:
            raise ValueError("Cyborg migration accepts only native descending or measured parents")
        parent = parents[manifest["schema"]].load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            if obj.mode != "live":
                raise ValueError("Cyborg optical feedback requires inherited live mode")
            before = preservation_fingerprints(parent)
            rotation, centers = parent.eyes.pose()
            obj.rotor = CyborgRotor.from_pose(obj.time_ns, rotation, centers)
            obj.eyes = CyborgEye.adopt(parent.eyes, obj.rotor)
            adopted_rotation, adopted_centers = obj.eyes.pose()
            pose_error = max(float(np.max(np.abs(adopted_rotation-rotation))),
                *(float(np.max(np.abs(adopted_centers[s]-centers[s]))) for s in ("L", "R")))
            if pose_error > 1e-12:
                raise ValueError("Optical migration must preserve its initial physical pose")
            if manifest["schema"] == "matrix_measured_visual_session_v1":
                world_center, world_frame = parent.light_world.center_mm.copy(), parent.light_world.frame.copy()
            else:
                world_center = np.mean([centers[s] for s in ("L", "R")], axis=0)
                world_frame = rotation.copy()
            previous_light = obj.pending_light.copy()
            obj.light_world = CyborgVisualWorld(obj.time_ns, world_center, world_frame,
                condition=condition, speed_deg_s=speed_deg_s)
            obj.initial_camera = (rotation.copy(), {s: centers[s].copy() for s in ("L", "R")})
            obj.output_connected = output_connected
            obj.pending_cyborg_command = obj.cyborg_command()
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            prior_tolerances = {k: float(obj.hybrid.parameters[k]) for k in ("rtol", "atol")}
            if tight:
                obj.hybrid.parameters["rtol"] = prior_tolerances["rtol"]*.1
                obj.hybrid.parameters["atol"] = prior_tolerances["atol"]*.1
            after = preservation_fingerprints(obj)
            checks = {key: before[key] == after[key] for key in before}
            if not all(checks.values()):
                raise ValueError(f"Cyborg adoption changed inherited scientific state: {checks}")
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="EXTERNAL_CYBORG_ROTOR_ASSAY_v1", cyborg_output_connected=output_connected,
                cyborg_condition=condition, tight_neural_numerics=tight,
                biological_validation=False, optical_radiometry_calibrated=False,
                engineering_prosthesis=True, animal_ability_demonstrated=False)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(path.resolve()), parent_manifest_sha256=sha256(path/"manifest.json"),
                time_ns=obj.time_ns, operation="Mount inherited optical rays on an external six-port-driven engineering rotor",
                external_scope="Arithmetic motor pooling, SI inertia/damping/torque and fixed-anchor camera are engineering choices; not reconstructed cervical physiology",
                optical_pose_max_absolute_migration_error=pose_error,
                optical_motion_source="Detached physical rotor; inherited body continues independently",
                screen_frame="Fixed apparatus; preserve measured parent frame when present",
                pending_retina_change_max=float(np.max(np.abs(previous_light-obj.pending_light))),
                output_connected=output_connected, motor_delay_ns=obj.CONTROL_NS,
                preserved_state_checks=checks, preserved_state_sha256=before,
                prior_solver_tolerances=prior_tolerances,
                solver_tolerances={k: float(obj.hybrid.parameters[k]) for k in ("rtol", "atol")},
                neuronal_equations_changed=False, neural_signs_or_gains_changed=False,
                post_retinal_current_injection=False, biological_validation=False,
                animal_ability_demonstrated=False)
            obj.source_identity = {name: sha256(ROOT/"src"/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def cyborg_command(self):
        raw = self.rotor.map_release(self.brain.node_ids, self.hybrid.release())["command"]
        return float(raw) if self.output_connected else 0.

    def _validate(self):
        super()._validate()
        if self.mode != "live" or type(self.output_connected) is not bool:
            raise ValueError("Unsupported cyborg connection or visual mode")
        if self.rotor.time_ns != self.time_ns or self.eyes.rotor is not self.rotor:
            raise ValueError("Rotor clock or optical coupling disagrees with session")
        if (type(self.pending_cyborg_command) is not float
                or not np.isfinite(self.pending_cyborg_command)
                or not 0. <= self.pending_cyborg_command <= 1.
                or self.pending_cyborg_command != self.cyborg_command()):
            raise ValueError("Pending cyborg command differs from current six-port readout/connection")

    def _validate_pending(self):
        super()._validate_pending()
        if self.pending_cyborg_command != self.cyborg_command():
            raise ValueError("Pending cyborg command is not the recorded causal output")

    def step(self):
        # This is the inherited live step, with an independent rotor advanced
        # using the previous pending command before the next retinal sample.
        self._validate()
        light = self.pending_light.copy()
        drive = (self.pending_proprioception["drive"].astype(float) if self.proprioception_enabled
                 else np.zeros(self.brain.n_neurons))
        for k, side in enumerate(("L", "R")):
            drive[self.port_indices[f"ORN_DM1_{side}"]] += self.config["odor_drive"]*self.pending_sensors[k]
        excitation = self.pending_excitation.copy()
        command = self.pending_cyborg_command
        try:
            self.hybrid.advance(self.CONTROL_NS, drive, light)
            self.plasticity.step(self.CONTROL_NS*1e-9)
            self.hybrid.sync_plastic_weights(self.plasticity)
            next_excitation = self.motor_excitation()
            next_command = self.cyborg_command()
            physical_dt_ns = round(self.body.dt*1e9)
            for _ in range(self.CONTROL_NS//physical_dt_ns):
                torque = self.muscles.advance(excitation, physical_dt_ns)
                self.body.advance(torque, 1)
            self.rotor.advance(command, self.CONTROL_NS)
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.pending_proprioception = self.proprioception.sample()
            self.pending_light = self.eyes.sample(self.light_world, self.time_ns)
            self.pending_excitation = next_excitation
            self.pending_cyborg_command = next_command
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
                retinal_cyborg_motion_rms=float(np.sqrt(np.mean((self.pending_light-frozen)**2))),
                optical_mount_rotation=rotation.tolist(), eye_centers_mm={k: v.tolist() for k,v in centers.items()},
                visual_release={name: dict(count=len(idx), mean=float(release[idx].mean()),
                    maximum=float(release[idx].max())) for name,idx in self.visual_groups.items() if len(idx)},
                active_release_above_point01=int(np.count_nonzero(release > .01)),
                nonvisual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.ri] > .95)),
                visual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.vi] > .95)),
                plastic_factor_min=float(self.plasticity.factors.min()),
                numeric_accepted=self.hybrid.statistics["accepted"], numeric_rejected=self.hybrid.statistics["rejected"],
                cyborg=dict(command_used=command, command_pending=next_command, output_connected=self.output_connected,
                    angle_rad=self.rotor.angle_rad, omega_rad_s=self.rotor.omega_rad_s,
                    torque_nm=self.rotor.last_torque_nm, engineering_prosthesis=True))
            self.history.append(row)
            self._validate()
            return row
        except BaseException:
            self.failed = True
            raise

    def state_dict(self):
        result = super().state_dict()
        result.update(schema=SCHEMA, rotor=self.rotor.state_dict(),
            pending_cyborg_command=self.pending_cyborg_command, output_connected=self.output_connected)
        return result

    def save(self, path):
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Runtime source changed after cyborg session creation")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving checkpoint {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="."+path.name+"-", dir=path.parent))
        try:
            self.brain.save_checkpoint(staging/"brain")
            write_state(staging/"session", self.state_dict())
            (staging/"source").mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT/"src"/name, staging/"source"/name)
            manifest = dict(schema=SCHEMA, time_ns=self.time_ns, neuron_count=self.brain.n_neurons,
                stored_edges=self.brain.W.nnz, mapped_photoreceptors=len(self.eyes.ids),
                mode=self.mode, output_connected=self.output_connected, engineering_prosthesis=True,
                biological_validation=False, animal_ability_demonstrated=False,
                files={str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()})
            (staging/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported cyborg checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Incomplete cyborg checkpoint")
        for name,digest in manifest["files"].items():
            if sha256(path/name) != digest:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        state = read_state(path/"session")
        special = {"schema", "world", "body", "hybrid", "eyes", "light_world", "muscles",
                   "plasticity", "proprioception", "used_light", "rotor"}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state["schema"] != SCHEMA:
            raise ValueError("Unsupported or incomplete cyborg state")
        if state["source_identity"] != {name: sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Checkpoint requires its archived source versions")
        obj = cls()
        obj.brain = AnatomicalRateBrain.load_checkpoint(path/"brain")
        for key in SCALARS | CYBORG_SCALARS:
            setattr(obj, key, state[key])
        if obj.config["numba_version"] != numba.__version__:
            raise ValueError("Checkpoint requires its recorded numerical runtime")
        if obj.config["numerical_backend"] == "cuda_fp64" and obj.config["backend_identity"] != GpuVisualBrain.backend_identity():
            raise ValueError("Recorded GPU backend differs")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited odor world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        kinds = {kind.SCHEMA: kind for kind in (GradedDescendingBrain, GpuGradedDescendingBrain)}
        obj.hybrid = kinds[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        if isinstance(obj.hybrid, GpuGradedDescendingBrain) and obj.config["electrical_backend_identity"] != obj.hybrid.backend_identity():
            raise ValueError("Recorded local electrical GPU backend differs")
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.rotor = CyborgRotor.from_state(state["rotor"])
            obj.eyes = CyborgEye.from_state(obj.brain, obj.body, state["eyes"], obj.rotor)
            obj.light_world = CyborgVisualWorld.from_state(state["light_world"])
            obj.muscles = ContractileTibia.from_state(obj.body, state["muscles"])
            obj.used_light = list(state["used_light"])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            if manifest["time_ns"] != obj.time_ns or manifest["output_connected"] != obj.output_connected:
                raise ValueError("Manifest clock or output connection differs")
            return obj
        except BaseException:
            obj.body.close()
            raise
