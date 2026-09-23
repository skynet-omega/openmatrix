"""Whole continuing CNS under a physical looming display, with optional fixed optics.

A fixed camera is an explicit experimental instrument. The old rotor, body and
neural dynamics keep evolving; no neural coordinate is clamped or reset.
"""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
import numpy as np
from cyborg_retinal_session import (CyborgRetinalSession, SOURCES as PARENT_SOURCES,
    CYBORG_SCALARS, preservation_fingerprints, _digest, AnatomicalRateBrain,
    AnatomicalProprioception, CandidateGammaPlasticity, ContractileTibia,
    CyborgEye, GradedDescendingBrain, GpuGradedDescendingBrain, GpuVisualBrain,
    MeasuredVisualSession, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    guard_visual_session, sha256, read_state, write_state,
    CyborgBidirectionalRotor, CyborgRetinalPortBrain, GpuCyborgRetinalPortBrain)
from cyborg_looming_world import CyborgLoomingWorld
from sensorimotor_contact_session import identical

SCHEMA="matrix_cyborg_looming_session_v1"
SOURCES=tuple(PARENT_SOURCES)+("cyborg_looming_world.py","cyborg_looming_session.py")
REQUIRED_PARENT=ROOT/"runs/cyborg_functional_20260908/cli_retinal_continued"

class FixedCyborgEye(CyborgEye):
    @classmethod
    def adopt_fixed(cls,reference,rotor,fixed_pose):
        obj=cls.from_state(reference.brain,reference.body,reference.state_dict(),rotor)
        obj.fixed_pose=copy.deepcopy(fixed_pose)
        return obj

    def pose(self):
        return copy.deepcopy(self.fixed_pose)

class CyborgLoomingSession(CyborgRetinalSession):
    @classmethod
    def from_parent(cls,path,condition="right",lv_s=.060,camera_fixed=True,output_connected=False,tight=False):
        path=Path(path).resolve()
        if path!=REQUIRED_PARENT.resolve():
            raise ValueError("This prospective assay requires its predeclared continuing retinal parent.")
        if any(type(value) is not bool for value in (camera_fixed,output_connected,tight)):
            raise ValueError("Explicit instrument/connection/tolerance flags required.")
        parent=CyborgRetinalSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            if obj.mode!="live" or not obj.config["retinal_port_enabled"]:
                raise ValueError("Requires the currently enabled local retinal device in live mode.")
            before=preservation_fingerprints(parent)
            old_rotor=_digest(parent.rotor.state_dict())
            old_pose=parent.eyes.pose()
            old_light=parent.pending_light.copy()
            rotation,centers=old_pose
            obj.initial_camera=copy.deepcopy(old_pose)
            if camera_fixed:
                obj.eyes=FixedCyborgEye.adopt_fixed(parent.eyes,obj.rotor,obj.initial_camera)
            obj.light_world=CyborgLoomingWorld.from_camera(obj.time_ns,rotation,centers,condition,lv_s)
            obj.output_connected=output_connected
            obj.pending_cyborg_command=obj.cyborg_command()
            obj.pending_light=obj.eyes.sample(obj.light_world,obj.time_ns)
            prior={k:float(obj.hybrid.parameters[k]) for k in ("rtol","atol")}
            if tight:
                for key,value in prior.items():obj.hybrid.parameters[key]=.1*value
            after=preservation_fingerprints(obj)
            checks={k:before[k]==after[k] for k in before}
            checks.update(initial_rotor_state=old_rotor==_digest(obj.rotor.state_dict()),
                          initial_optical_pose=identical(old_pose,obj.eyes.pose()))
            if not all(checks.values()):raise ValueError(f"Looming migration changed protected state: {checks}")
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate="CYBORG_LOOMING_v1",looming_camera_fixed=camera_fixed,
                cyborg_output_connected=output_connected,cyborg_condition=condition,
                looming_lv_s=float(lv_s),tight_neural_numerics=tight,
                biological_validation=False,animal_ability_demonstrated=False)
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/"manifest.json"),
                time_ns=obj.time_ns,operation="Present physical lateral looming with an explicit optical bench",
                preserved_state_checks=checks,preserved_state_sha256=before,
                initial_neural_state_reset=False,initial_rotor_velocity_reset=False,
                retinal_pending_changed_by_new_display=True,pending_retina_change_max=float(np.max(np.abs(obj.pending_light-old_light))),
                camera_fixed=camera_fixed,output_connected=output_connected,
                fixed_camera_scope="Camera pose held by an independent instrument; the inherited rotor still evolves with its old momentum. No neural feedback or action credited to fixation.",
                stimulus_scope="Known angular stimulus with declared reconstructed elevation, display clock and radiometry; no exact reproduction of recorded pattern images.",
                neuronal_equations_changed=False,neural_signs_or_gains_changed=False,
                post_retinal_current_injection=False,biological_validation=False,
                prior_solver_tolerances=prior,solver_tolerances={k:float(obj.hybrid.parameters[k]) for k in prior})
            obj.source_identity={name:sha256(ROOT/"src"/name) for name in SOURCES}
            guard_visual_session(obj);obj._validate();obj._validate_pending()
            return obj
        except BaseException:
            obj.close();raise

    def _validate(self):
        super()._validate()
        if type(self.config.get("looming_camera_fixed")) is not bool:
            raise ValueError("Missing optical instrument policy.")
        if self.config["looming_camera_fixed"]:
            if not isinstance(self.eyes,FixedCyborgEye) or not identical(self.eyes.pose(),self.initial_camera):
                raise ValueError("Fixed camera differs from its declared physical pose.")
        elif isinstance(self.eyes,FixedCyborgEye):
            raise ValueError("Unexpected fixed optics in live-camera assay.")

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
