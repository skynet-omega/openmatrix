"""Persist the measured optical world with the continuing full-CNS preparation.

This separate schema preserves the 43 archived sources. Native restoration is
explicit; the new world is never disguised as one of the historical worlds.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from anatomical_rate_brain import AnatomicalRateBrain
from anatomical_plasticity import CandidateGammaPlasticity
from anatomical_proprioception import AnatomicalProprioception
from contractile_tibia import ContractileTibia
from electrical_parameter_contract import guard_visual_session
from gpu_graded_descending_brain import GpuGradedDescendingBrain
from graded_descending_brain import GradedDescendingBrain
from gpu_visual_brain import GpuVisualBrain
from measured_luminance_world import MeasuredLuminanceWorld
from organism_session import OdorPatchWorld, ROOT
from refined_contact_body import RefinedContactBody
from retinal_world import CompoundEye
from session_io import sha256, read_state, write_state
from visuomotor_session import SCALARS
from visual_descending_session import VisualDescendingSession, SOURCES as PARENT_SOURCES


SCHEMA = "matrix_measured_visual_session_v1"
SOURCES = tuple(PARENT_SOURCES) + ("electrical_parameter_contract.py", "measured_luminance_world.py", "measured_visual_session.py")


class MeasuredVisualSession(VisualDescendingSession):
    @classmethod
    def from_parent(cls, path, sample_elapsed_ns, luminance, condition="measured"):
        # The organism samples its physical eyes once per inherited 1 ms tick.
        # Reject exact-replay claims for off-grid events instead of rounding.
        clocks = np.asarray(sample_elapsed_ns)
        if clocks.dtype.kind not in "iu" or np.any(clocks % cls.CONTROL_NS):
            raise ValueError("Exact presentation requires sample events aligned to inherited 1 ms eye ticks")
        manifest = json.loads((Path(path)/"manifest.json").read_text())
        parent = cls.load(path) if manifest.get("schema") == SCHEMA else VisualDescendingSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            if obj.mode != "live":
                raise ValueError("Measured light assay requires the live optical pathway")
            if isinstance(parent.light_world, MeasuredLuminanceWorld):
                # Branches following common adaptation retain the physical
                # apparatus frame, even if the head has moved meanwhile.
                center, rotation = parent.light_world.center_mm.copy(), parent.light_world.frame.copy()
            else:
                rotation, centers = obj.eyes.pose()
                center = np.mean([centers[k] for k in sorted(centers)], axis=0)
            previous = obj.pending_light.copy()
            obj.light_world = MeasuredLuminanceWorld(obj.time_ns, center, rotation,
                sample_elapsed_ns, luminance, condition)
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="MEASURED_LUMINANCE_ASSAY_v1", measured_luminance_condition=condition,
                              biological_validation=False, optical_radiometry_calibrated=False)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(Path(path).resolve()), parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                time_ns=obj.time_ns, operation="Replace physical light world by explicit measured/timed luminance sequence",
                world_frame="Fixed apparatus; first assay starts at mean eye center/head orientation, subsequent measured-session branches preserve that same frame",
                optical_sequence_sha256=obj.light_world.sequence_sha256,
                preserved="All neural coordinates, weights, topology, excitability, phototransduction, chemical transmission, plasticity, body and pending motor/proprioceptive signals",
                pending_retina_change_max=float(np.max(np.abs(previous-obj.pending_light))),
                neural_current_injection=False, new_neural_signs=False, biological_validation=False,
                optical_scope="Normalized luminance and inherited candidate optics; no photon-flux or visual physiological calibration inferred")
            obj.source_identity = {name:sha256(ROOT/"src"/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def state_dict(self):
        result = super().state_dict()
        result["schema"] = SCHEMA
        return result

    def save(self, path):
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name:sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Runtime source changed after session creation")
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
                biological_validation=False, mode=self.mode,
                files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()})
            (staging/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        # Explicit restoration mirrors the frozen parent but dispatches only
        # this schema and this optical world. No archived source is edited.
        path = Path(path)
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported measured visual checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Incomplete measured visual checkpoint")
        for name, digest in manifest["files"].items():
            if sha256(path/name) != digest:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        state = read_state(path/"session")
        special = {"schema", "world", "body", "hybrid", "eyes", "light_world", "muscles", "plasticity", "proprioception", "used_light"}
        if set(state) != SCALARS | special or state["schema"] != SCHEMA:
            raise ValueError("Unsupported or incomplete measured visual state")
        if state["source_identity"] != {name:sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Checkpoint requires its archived source versions")
        obj = cls()
        obj.brain = AnatomicalRateBrain.load_checkpoint(path/"brain")
        for key in SCALARS:
            setattr(obj, key, state[key])
        if obj.config["numba_version"] != numba.__version__:
            raise ValueError("Checkpoint requires its recorded numerical runtime")
        if obj.config["numerical_backend"] == "cuda_fp64" and obj.config["backend_identity"] != GpuVisualBrain.backend_identity():
            raise ValueError("Recorded GPU backend differs")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        kinds = {kind.SCHEMA:kind for kind in (GradedDescendingBrain, GpuGradedDescendingBrain)}
        obj.hybrid = kinds[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        if isinstance(obj.hybrid, GpuGradedDescendingBrain) and obj.config["electrical_backend_identity"] != obj.hybrid.backend_identity():
            raise ValueError("Recorded local electrical GPU backend differs")
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.eyes = CompoundEye.from_state(obj.brain, obj.body, state["eyes"])
            obj.light_world = MeasuredLuminanceWorld.from_state(state["light_world"])
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
