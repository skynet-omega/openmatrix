"""Checkpointed numerical refinement of the continuing visuomotor life."""
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

SCHEMA = "matrix_precision_visuomotor_session_v1"
SOURCES = tuple(VISUAL_SOURCES) + ("gpu_visual_brain.py", "refined_contact_body.py", "periodic_light_world.py", "precision_visuomotor_session.py")


class PrecisionVisuomotorSession(VisuomotorSession):
    @classmethod
    def from_visual(cls, path, dt=.000025, backend="cuda_fp64", periodic=False):
        if backend not in ("cuda_fp64", "cpu_fp64"):
            raise ValueError("Unknown numerical backend")
        schema=json.loads((Path(path)/"manifest.json").read_text())["schema"]
        parent=cls.load(path) if schema==SCHEMA else VisuomotorSession.load(path)
        obj=cls()
        obj.__dict__.update(parent.__dict__)
        old_body=parent.body
        try:
            obj.body=RefinedContactBody(yaw=old_body.constructor["yaw"],seed=old_body.constructor["seed"],dt=dt)
            obj.body.adopt(old_body)
            obj.muscles=ContractileTibia.from_state(obj.body,parent.muscles.state_dict())
            obj.eyes=CompoundEye.from_state(obj.brain,obj.body,parent.eyes.state_dict())
            proprio_state=parent.proprioception.state_dict()
            obj.proprioception=AnatomicalProprioception.__new__(AnatomicalProprioception)
            obj.proprioception._initialize(obj.brain,obj.body,proprio_state['polarity'],
                proprio_state['manifest'],proprio_state['manifest_origin'])
            expected_identity=copy.deepcopy(proprio_state['identity'])
            expected_identity['physical_model_sha256']=obj.body.identity['model_sha256']
            if obj.proprioception.identity!=expected_identity:
                raise ValueError('Precision replacement changed sensory mapping or tuning')
            if backend=="cuda_fp64":
                obj.hybrid=GpuVisualBrain.adopt(parent.hybrid)
            else:
                state=parent.hybrid.state_dict()
                state["schema"]=HybridVisualBrain.SCHEMA
                obj.hybrid=HybridVisualBrain.from_state(obj.brain,state)
            obj.hybrid.sync_plastic_weights(obj.plasticity)
            if periodic:
                if not isinstance(parent.light_world,LuminousWorld):
                    raise ValueError('Periodic protocol requires an inherited simple arena')
                obj.light_world=PeriodicLightWorld(parent.light_world,obj.time_ns)
                obj.pending_light=obj.eyes.sample(obj.light_world,obj.time_ns)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate="PRECISION_VISUOMOTOR_v1",numerical_backend=backend,
                backend_identity=GpuVisualBrain.backend_identity() if backend=="cuda_fp64" else {"numba":numba.__version__},
                physical_dt_s=dt)
            obj.intervention=dict(parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                previous_intervention=copy.deepcopy(parent.intervention), time_ns=obj.time_ns,
                operation="Change numerical backend and/or physical timestep; retain equations, weights, all state and sensory delay",
                physical_dt_s=dt,numerical_backend=backend, periodic_light_stimulus_added=periodic,
                biological_calibration_added=False)
            obj.source_identity={name:sha256(ROOT/"src"/name) for name in SOURCES}
            obj._validate()
            obj._validate_pending()
            old_body.close()
            return obj
        except BaseException:
            old_body.close()
            if obj.body is not old_body:
                obj.body.close()
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
        obj.hybrid = (GpuVisualBrain if obj.config["numerical_backend"] == "cuda_fp64" else HybridVisualBrain).from_state(obj.brain, state["hybrid"])
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.eyes = CompoundEye.from_state(obj.brain, obj.body, state["eyes"])
            world_type=PeriodicLightWorld if state['light_world'].get('schema')==PeriodicLightWorld.SCHEMA else LuminousWorld
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
