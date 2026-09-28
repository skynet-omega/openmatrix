"""Whole-organism cardinal light assay continuing the saved receptor CNS."""
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


from visuomotor_session import VisuomotorSession, SOURCES as VISUAL_SOURCES, SCALARS
from gpu_visual_brain import GpuVisualBrain
from refined_contact_body import RefinedContactBody
from periodic_light_world import PeriodicLightWorld
from visual_challenge_world import VisualChallengeWorld
from synaptic_visual_brain import SynapticVisualBrain
from gpu_synaptic_visual_brain import GpuSynapticVisualBrain
from precision_visuomotor_session import PrecisionVisuomotorSession, SOURCES as PRECISION_SOURCES

SCHEMA = "matrix_cardinal_visual_session_v1"
from visual_transmission_session import VisualTransmissionSession, SOURCES as TRANSMISSION_SOURCES
from receptor_visual_brain import ReceptorVisualBrain
from gpu_receptor_visual_brain import GpuReceptorVisualBrain

from visual_receptor_session import VisualReceptorSession, SOURCES as RECEPTOR_SOURCES
from cardinal_grating_world import CardinalGratingWorld
SOURCES = tuple(RECEPTOR_SOURCES) + ("cardinal_grating_world.py", "cardinal_visual_session.py")


class CardinalVisualSession(VisualReceptorSession):
    @classmethod
    def from_receptor(cls, path, condition="azimuth_positive", tight=False):
        parent = VisualReceptorSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            rotation, centers = obj.eyes.pose()
            center = np.mean([centers[k] for k in sorted(centers)], axis=0)
            obj.light_world = CardinalGratingWorld(obj.time_ns, center, rotation, condition)
            previous_light = obj.pending_light.copy()
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            if tight:
                obj.hybrid.parameters["rtol"] = 1e-6
                obj.hybrid.parameters["atol"] = 1e-8
            obj.mode = "live"
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="CARDINAL_VISUAL_ASSAY_v1", visual_challenge=condition,
                tight_neural_numerics=bool(tight), biological_validation=False)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(Path(path).resolve()), parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                time_ns=obj.time_ns, operation="Replace physical arena texture with a declared cardinal grating",
                all_neural_and_body_state_preserved_exactly=True,
                anatomical_weights_and_topology_preserved=True,
                neural_parameter_change="rtol=1e-6, atol=1e-8" if tight else None,
                world_frame="Fixed to initial head orientation and mean eye center; never follows the animal",
                pending_retina_change_max=float(np.max(np.abs(obj.pending_light-previous_light))),
                fitting="No neural, receptor, optical, motor or body parameter fitted")
            obj.source_identity = {name:sha256(ROOT/"src"/name) for name in SOURCES}
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            parent.close()
            raise

    def state_dict(self):
        result=super().state_dict()
        result["schema"]=SCHEMA
        return result

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
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding='utf-8'))
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
        if obj.config["numerical_backend"] == "cuda_fp64" and obj.config["backend_identity"] != GpuVisualBrain.backend_identity():
            raise ValueError("Recorded GPU backend differs")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        brain_types = {kind.SCHEMA: kind for kind in (ReceptorVisualBrain, GpuReceptorVisualBrain)}
        obj.hybrid = brain_types[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.eyes = CompoundEye.from_state(obj.brain, obj.body, state["eyes"])
            world_type=CardinalGratingWorld
            obj.light_world = world_type.from_state(state["light_world"])
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
