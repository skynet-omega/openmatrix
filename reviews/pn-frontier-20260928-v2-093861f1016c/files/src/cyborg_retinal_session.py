"""A continuing cyborg whose mapped R1-R6 boundary is a local sensor device.

This candidate replaces only the selected optical-port membrane derivatives.
Its input encoding is engineered and is not a recovered biological retina.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from cyborg_visual_session import (CyborgVisualSession,
    CYBORG_SCALARS, preservation_fingerprints, _digest, AnatomicalRateBrain,
    AnatomicalProprioception, CandidateGammaPlasticity, ContractileTibia,
    CyborgEye, GradedDescendingBrain, GpuGradedDescendingBrain, GpuVisualBrain,
    MeasuredVisualSession, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    guard_visual_session, sha256, read_state, write_state)
from cyborg_bidirectional_rotor import CyborgBidirectionalRotor, DESIGN_SHA256
from cyborg_phase_world import CyborgPhaseWorld
from cyborg_bidirectional_session import CyborgBidirectionalSession, SOURCES as PARENT_SOURCES, REQUIRED_PARENT
from cyborg_retinal_port import CyborgRetinalPortBrain, GpuCyborgRetinalPortBrain
from sensorimotor_contact_session import identical


SCHEMA = "matrix_cyborg_retinal_session_v1"
SOURCES = tuple(PARENT_SOURCES) + ("cyborg_retinal_port.py", "cyborg_retinal_session.py")


class CyborgRetinalSession(CyborgBidirectionalSession):
    @classmethod
    def from_parent(cls, path, condition="down", speed_deg_s=60., output_connected=True, tight=False, enabled=True):
        parent = CyborgBidirectionalSession.from_parent(path, condition, speed_deg_s, output_connected, tight)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            before = parent.state_dict()
            before_weights = parent.brain.W.data.copy()
            before_rates = parent.brain.rates.copy()
            kind = GpuCyborgRetinalPortBrain if isinstance(parent.hybrid, GpuGradedDescendingBrain) else CyborgRetinalPortBrain
            obj.hybrid = kind.adopt(parent.hybrid, obj.eyes.ids, enabled=enabled)
            after = obj.state_dict()
            checks = {key: identical(value, after[key]) for key,value in before.items() if key not in {"schema", "hybrid"}}
            checks["inherited_hybrid_state"] = all(identical(value, after["hybrid"][key]) for key,value in before["hybrid"].items() if key != "schema")
            checks["weights"] = np.array_equal(before_weights, obj.brain.W.data)
            checks["published_rates"] = np.array_equal(before_rates, obj.brain.rates)
            if not all(checks.values()):
                raise ValueError(f"Retinal port adoption changed inherited state: {checks}")
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="CYBORG_RETINAL_PORT_v1", retinal_port_enabled=enabled,
                retinal_prosthesis=True, biological_photoreceptor_reconstruction=False,
                biological_validation=False, animal_ability_demonstrated=False)
            if isinstance(obj.hybrid, GpuCyborgRetinalPortBrain):
                obj.config["electrical_backend_identity"] = obj.hybrid.backend_identity()
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(Path(path).resolve()), parent_manifest_sha256=sha256(Path(path)/"manifest.json"),
                time_ns=obj.time_ns, operation="Replace mapped R1-R6 membrane derivatives by independent local luminance ports",
                retinal_port_enabled=enabled, preserved_state_checks=checks,
                optical_mapping_preserved=True, initial_state_jump=False,
                neuronal_equations_changed=enabled, neural_signs_or_gains_changed=False,
                synthetic_input_encoding="Local light 0..1 to inherited release range 0..1; first-order voltage relaxation 5ms",
                unavailable_biological_credit="These selected photoreceptor dynamics and recurrent input integration are replaced by a sensory prosthesis.",
                no_object_direction_or_action_decoder=True, biological_validation=False,
                inherited_device_migration_checks=parent.intervention["preserved_state_checks"])
            obj.source_identity = {name: sha256(ROOT/"src"/name) for name in SOURCES}
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
        kinds = {kind.SCHEMA: kind for kind in (CyborgRetinalPortBrain, GpuCyborgRetinalPortBrain)}
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
            obj.light_world = CyborgPhaseWorld.from_state(state["light_world"])
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
