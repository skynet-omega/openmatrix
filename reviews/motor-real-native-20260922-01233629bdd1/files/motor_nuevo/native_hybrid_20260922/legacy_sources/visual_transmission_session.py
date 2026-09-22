"""Persistent finite-transmission visual experiment on the continuing whole CNS."""
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

SCHEMA = "matrix_visual_transmission_session_v1"
SOURCES = tuple(PRECISION_SOURCES) + ("synaptic_visual_brain.py", "gpu_synaptic_visual_brain.py", "visual_challenge_world.py", "visual_transmission_session.py")


class VisualTransmissionSession(PrecisionVisuomotorSession):
    @classmethod
    def from_precision(cls, path, condition="static", kinetic=True, tight=False):
        parent = PrecisionVisuomotorSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            before_state = parent.hybrid.state.copy()
            before_rates = parent.brain.rates.copy()
            before_release = parent.hybrid.release()
            if kinetic:
                backend = GpuSynapticVisualBrain if parent.config["numerical_backend"] == "cuda_fp64" else SynapticVisualBrain
                obj.hybrid = backend.adopt(parent.hybrid)
            obj.hybrid.sync_plastic_weights(obj.plasticity)
            np.testing.assert_array_equal(before_state, obj.hybrid.state[:len(before_state)])
            np.testing.assert_array_equal(before_release, obj.hybrid.release())
            np.testing.assert_array_equal(before_rates, obj.brain.rates)
            if tight:
                obj.hybrid.parameters["rtol"] = 1e-6
                obj.hybrid.parameters["atol"] = 1e-8
            reference = parent.light_world.reference if isinstance(parent.light_world, PeriodicLightWorld) else parent.light_world
            obj.light_world = VisualChallengeWorld(reference, obj.time_ns, condition)
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            obj.mode = "live"
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="VISUAL_TRANSMISSION_v1", synaptic_transmission=bool(kinetic),
                synaptic_tau_s=.005 if kinetic else None, visual_challenge=condition,
                tight_neural_numerics=bool(tight), biological_validation=False)
            obj.intervention = dict(parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                previous_intervention=copy.deepcopy(parent.intervention), time_ns=obj.time_ns,
                operation="Add finite chemical transmission candidate and/or explicit physical light challenge",
                existing_neural_state_preserved_exactly=True, weights_and_topology_preserved=True,
                physical_integration_muscles_and_pending_motor_preserved=True,
                new_synaptic_state="initialized at current release; no initial transmission jump" if kinetic else None,
                synaptic_tau_status="5 ms model prior, not measured class-specific kinetics" if kinetic else None,
                world_change="Existing sphere geometry and texture; uniform contrast or world-axis rotation challenge",
                pending_retina_change_max=float(np.max(np.abs(obj.pending_light-parent.pending_light))))
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
        if obj.config["numerical_backend"] == "cuda_fp64" and obj.config["backend_identity"] != GpuVisualBrain.backend_identity():
            raise ValueError("Recorded GPU backend differs")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        brain_types = {kind.SCHEMA: kind for kind in (GpuVisualBrain, HybridVisualBrain, SynapticVisualBrain, GpuSynapticVisualBrain)}
        obj.hybrid = brain_types[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.eyes = CompoundEye.from_state(obj.brain, obj.body, state["eyes"])
            world_type=VisualChallengeWorld
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
