"""Continuing whole-CNS experiment with a local PVLP adaptation hypothesis.

The prior session families remain loadable with their archived source identity.
Adaptation adds local memory, without resetting neural or mechanical history.
"""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
import numpy as np
from cyborg_looming_session import (
    CyborgLoomingSession, FixedCyborgEye, SOURCES as PARENT_SOURCES,
    CYBORG_SCALARS, AnatomicalRateBrain, AnatomicalProprioception,
    CandidateGammaPlasticity, ContractileTibia, CyborgEye,
    GpuGradedDescendingBrain, GpuVisualBrain, OdorPatchWorld,
    RefinedContactBody, ROOT, SCALARS, guard_visual_session,
    sha256, read_state, write_state, CyborgBidirectionalRotor,
    CyborgLoomingWorld, identical)
from pvlp_adaptation_brain import PvlpAdaptationBrain, GpuPvlpAdaptationBrain

SCHEMA = "matrix_pvlp_adaptation_session_v1"
SOURCES = tuple(PARENT_SOURCES) + ("pvlp_adaptation_brain.py", "pvlp_adaptation_session.py")


class PvlpAdaptationSession(CyborgLoomingSession):
    @classmethod
    def from_checkpoint(cls, path, *, beta, tau_s, enabled=True):
        parent = CyborgLoomingSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            inherited_state = parent.hybrid.state.copy()
            kind = GpuPvlpAdaptationBrain if isinstance(parent.hybrid, GpuGradedDescendingBrain) else PvlpAdaptationBrain
            obj.hybrid = kind.adopt(parent.hybrid, beta=beta, tau_s=tau_s, enabled=enabled)
            np.testing.assert_array_equal(obj.hybrid.state[:len(inherited_state)], inherited_state)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="PVLP_LOCAL_ACTIVITY_ADAPTATION_v1",
                              pvlp_adaptation_enabled=enabled,
                              biological_validation=False, animal_ability_demonstrated=False)
            if isinstance(obj.hybrid, GpuGradedDescendingBrain):
                obj.config["electrical_backend_identity"] = obj.hybrid.backend_identity()
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                time_ns=obj.time_ns, operation="Add local activity-dependent adaptation in PVLP024",
                beta=beta, tau_s=tau_s, enabled=enabled,
                new_memory_initialization="zero at adoption; no invented pre-adoption memory",
                inherited_neural_state_preserved=True, initial_weights_and_signs_preserved=True,
                equation_scope="Only the selected PVLP neural targets acquire intrinsic feedback",
                biological_validation=False)
            obj.source_identity = {name:sha256(ROOT/"src"/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def begin_stimulus(self, condition, *, lv_s=.060, camera_fixed=True, output_connected=False):
        """Start a physical display epoch, retaining all neural/device state."""
        if type(camera_fixed) is not bool or type(output_connected) is not bool:
            raise ValueError("Explicit camera and motor connection flags required")
        rotation, centers = self.eyes.pose()
        initial_camera = copy.deepcopy((rotation, centers))
        world = CyborgLoomingWorld.from_camera(self.time_ns, rotation, centers, condition, lv_s)
        eyes = CyborgEye.from_state(self.brain, self.body, self.eyes.state_dict(), self.rotor)
        if camera_fixed:
            eyes = FixedCyborgEye.adopt_fixed(eyes, self.rotor, initial_camera)
        self.initial_camera, self.eyes, self.light_world = initial_camera, eyes, world
        self.output_connected = output_connected
        self.config.update(looming_camera_fixed=camera_fixed,
                           cyborg_output_connected=output_connected,
                           cyborg_condition=condition, looming_lv_s=float(lv_s))
        self.pending_cyborg_command = self.cyborg_command()
        self.pending_light = self.eyes.sample(self.light_world, self.time_ns)
        self._validate()
        self._validate_pending()

    def state_dict(self):
        result = super().state_dict()
        result["schema"] = SCHEMA
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
                camera_fixed=self.config["looming_camera_fixed"], physical_looming_stimulus=True,
                pvlp_adaptation_enabled=self.hybrid.pvlp_adaptation_manifest["enabled"],
                retinal_port_enabled=self.config["retinal_port_enabled"], biological_retina_reconstructed=False,
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
            raise ValueError("Unsupported retinal-prosthesis checkpoint")
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
        kinds = {kind.SCHEMA: kind for kind in (PvlpAdaptationBrain, GpuPvlpAdaptationBrain)}
        obj.hybrid = kinds[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        if isinstance(obj.hybrid, GpuGradedDescendingBrain) and obj.config["electrical_backend_identity"] != obj.hybrid.backend_identity():
            raise ValueError("Recorded local electrical GPU backend differs")
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.rotor = CyborgBidirectionalRotor.from_state(state["rotor"])
            obj.eyes = CyborgEye.from_state(obj.brain, obj.body, state["eyes"], obj.rotor)
            if obj.config["looming_camera_fixed"]:
                obj.eyes = FixedCyborgEye.adopt_fixed(obj.eyes, obj.rotor, obj.initial_camera)
            obj.light_world = CyborgLoomingWorld.from_state(state["light_world"])
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
