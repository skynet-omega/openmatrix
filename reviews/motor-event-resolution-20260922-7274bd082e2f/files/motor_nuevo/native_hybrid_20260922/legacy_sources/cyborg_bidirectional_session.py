"""Continue the saved cyborg with signed device wiring and a continuous screen.

The live step is inherited unchanged. Native persistence is explicit because
the predecessor schema fixes a different rotor, optical world and source set.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from cyborg_visual_session import (CyborgVisualSession, SOURCES as PARENT_SOURCES,
    CYBORG_SCALARS, preservation_fingerprints, _digest, AnatomicalRateBrain,
    AnatomicalProprioception, CandidateGammaPlasticity, ContractileTibia,
    CyborgEye, GradedDescendingBrain, GpuGradedDescendingBrain, GpuVisualBrain,
    MeasuredVisualSession, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    guard_visual_session, sha256, read_state, write_state)
from cyborg_bidirectional_rotor import CyborgBidirectionalRotor, DESIGN_SHA256
from cyborg_phase_world import CyborgPhaseWorld


SCHEMA = "matrix_cyborg_bidirectional_session_v1"
SOURCES = tuple(PARENT_SOURCES) + ("cyborg_bidirectional_rotor.py", "cyborg_phase_world.py",
                                   "cyborg_bidirectional_session.py")
REQUIRED_PARENT = ROOT / "runs/cyborg_20260908/cli_continued"


class CyborgBidirectionalSession(CyborgVisualSession):
    @classmethod
    def from_parent(cls, path, condition="down", speed_deg_s=60., output_connected=True, tight=False):
        path = Path(path).resolve()
        if path != REQUIRED_PARENT.resolve():
            raise ValueError("This candidate continues only runs/cyborg_20260908/cli_continued")
        if type(output_connected) is not bool or type(tight) is not bool:
            raise ValueError("Explicit boolean connection and tolerance choices required")
        parent = CyborgVisualSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            if obj.mode != "live":
                raise ValueError("Bidirectional candidate requires the live visual pathway")
            before = preservation_fingerprints(parent)
            initial_rotation, initial_centers = parent.eyes.pose()
            prior_rotor = parent.rotor.state_dict()
            previous_light = parent.pending_light.copy()
            prior_phase = parent.light_world.phase_rad(parent.time_ns)
            obj.rotor = CyborgBidirectionalRotor.from_predecessor(parent.rotor)
            obj.eyes = CyborgEye.adopt(parent.eyes, obj.rotor)
            obj.light_world = CyborgPhaseWorld.from_predecessor(parent.light_world, obj.time_ns,
                                                               condition, speed_deg_s)
            obj.initial_camera = (initial_rotation.copy(), {s: v.copy() for s, v in initial_centers.items()})
            obj.output_connected = output_connected
            obj.pending_cyborg_command = obj.cyborg_command()
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            prior_tolerances = {k: float(obj.hybrid.parameters[k]) for k in ("rtol", "atol")}
            if tight:
                for key in prior_tolerances:
                    obj.hybrid.parameters[key] = prior_tolerances[key] * .1
            after = preservation_fingerprints(obj)
            checks = {key: before[key] == after[key] for key in before}
            rotor_fields = ("start_ns", "time_ns", "initial_rotation", "initial_centers", "pivot_mm",
                            "angle_rad", "omega_rad_s", "last_command", "last_torque_nm")
            current_rotor = obj.rotor.state_dict()
            checks.update(rotor_physical_state=_digest({k: prior_rotor[k] for k in rotor_fields})
                          == _digest({k: current_rotor[k] for k in rotor_fields}),
                          current_optical_pose=_digest(parent.eyes.pose()) == _digest(obj.eyes.pose()),
                          pending_retina_exact=np.array_equal(previous_light, obj.pending_light),
                          current_screen_phase_exact=prior_phase == obj.light_world.phase_rad(obj.time_ns))
            if not all(checks.values()):
                raise ValueError(f"Signed device migration changed protected state: {checks}")
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate="BIDIRECTIONAL_CYBORG_ASSAY_v1",
                cyborg_output_connected=output_connected, cyborg_condition=condition,
                tight_neural_numerics=tight, engineering_prosthesis=True,
                biological_validation=False, animal_ability_demonstrated=False)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(path), parent_manifest_sha256=sha256(path / "manifest.json"),
                time_ns=obj.time_ns, operation="Replace six-port device wiring with signed VL1-minus-VL2 terminals",
                design_sha256=DESIGN_SHA256, preserved_state_checks=checks, preserved_state_sha256=before,
                pending_retina_change_max=float(np.max(np.abs(previous_light-obj.pending_light))),
                phase_offset_rad=prior_phase, output_connected=output_connected,
                initial_rotor_angle_rad=obj.rotor.angle_rad, initial_rotor_omega_rad_s=obj.rotor.omega_rad_s,
                initial_velocity_reset=False, output_cut="Zero subsequent applied torque; existing angular momentum coasts",
                neuronal_equations_changed=False, neural_signs_or_gains_changed=False,
                prior_solver_tolerances=prior_tolerances,
                solver_tolerances={k: float(obj.hybrid.parameters[k]) for k in prior_tolerances},
                motor_delay_ns=obj.CONTROL_NS, biological_antagonism_claimed=False,
                biological_validation=False, animal_ability_demonstrated=False)
            obj.source_identity = {name: sha256(ROOT / "src" / name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def _validate(self):
        # Bypass only the predecessor's [0,1] device-command restriction.
        MeasuredVisualSession._validate(self)
        if self.mode != "live" or type(self.output_connected) is not bool:
            raise ValueError("Unsupported signed cyborg connection or visual mode")
        if (not isinstance(self.rotor, CyborgBidirectionalRotor) or self.rotor.time_ns != self.time_ns
                or self.eyes.rotor is not self.rotor):
            raise ValueError("Signed rotor identity, clock or optical coupling differs")
        if (type(self.pending_cyborg_command) is not float or not np.isfinite(self.pending_cyborg_command)
                or not -1. <= self.pending_cyborg_command <= 1.
                or self.pending_cyborg_command != self.cyborg_command()):
            raise ValueError("Pending signed command differs from the fixed eight-port wiring")

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
