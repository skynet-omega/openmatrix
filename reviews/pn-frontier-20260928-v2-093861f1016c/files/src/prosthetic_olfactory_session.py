"""Continuing CNS with an explicit prosthetic olfactory afferent port.

The new synaptic states are integrated together with the existing neural
state. The inherited brain, world, body, motor interface and sensory input
schedule retain their previous implementation. No gamma current ablation.
"""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
import numpy as np
import numpy as np
from pvlp_adaptation_session import (
    PvlpAdaptationSession, SOURCES as PARENT_SOURCES,
    CYBORG_SCALARS, AnatomicalRateBrain, AnatomicalProprioception,
    CandidateGammaPlasticity, ContractileTibia, CyborgEye, FixedCyborgEye,
    GpuGradedDescendingBrain, GpuVisualBrain, OdorPatchWorld,
    RefinedContactBody, ROOT, SCALARS, guard_visual_session,
    sha256, read_state, write_state, CyborgLoomingWorld,
    PvlpAdaptationBrain, GpuPvlpAdaptationBrain)
from cyborg_centered_servo import CyborgCenteredServo

from pvlp_servo_session import PvlpServoSession, SOURCES as SERVO_SOURCES

from kcgamma_sensory_probe import KcGammaSensoryProbeSession, SOURCES as PROBE_SOURCES
from prosthetic_olfactory_brain import ProstheticOlfactoryBrain, GpuProstheticOlfactoryBrain
from kcgamma_sensory_probe import pulse_amplitude
from cyborg_visual_session import preservation_fingerprints

SCHEMA = 'matrix_prosthetic_olfactory_session_v1'
SOURCES = tuple(PROBE_SOURCES) + ('olfactory_synapse_candidate.py', 'orn_pn_synaptic_brain.py', 'prosthetic_olfactory_brain.py', 'prosthetic_olfactory_session.py')


class ProstheticOlfactorySession(KcGammaSensoryProbeSession):
    @classmethod
    def from_checkpoint(cls, path, *, selection_path, initialization, condition,
                        onset_ms=200, pulse_ms=500, pulse_hz=20., baseline_hz=7., enabled=True, afferent_connected=True, tight=False):
        parent = KcGammaSensoryProbeSession.from_checkpoint(path,
            roi_path=ROOT/'evidence/circuit_integration_20260908/memory/KCgamma_complete_ROI_CSR_counts.npz',
            mask_enabled=False, condition=condition, onset_ms=onset_ms,
            pulse_ms=pulse_ms, amplitude=0., tight=tight)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            inherited = preservation_fingerprints(parent)
            kind = GpuProstheticOlfactoryBrain if isinstance(parent.hybrid, GpuPvlpAdaptationBrain) else ProstheticOlfactoryBrain
            obj.hybrid = kind.adopt(parent.hybrid, selection_path, initialization, enabled=enabled, baseline_hz=baseline_hz)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate='PROSTHETIC_OLFACTORY_AFFERENT_v1',
                orn_pn_component_enabled=enabled,
                electrical_backend_identity=obj.hybrid.backend_identity() if hasattr(obj.hybrid, 'backend_identity') else obj.config.get('electrical_backend_identity'))
            obj.probe = copy.deepcopy(parent.probe)
            obj.probe.update(initialization=initialization, component_enabled=enabled,
                selection_sha256=sha256(selection_path), inherited_fingerprints=inherited,
                afferent_baseline_hz=float(baseline_hz), afferent_pulse_hz=float(pulse_hz),
                afferent_connected=afferent_connected)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                time_ns=obj.time_ns, operation='Engineering afferent rate replaces the presynaptic input to the selected ORN-to-PN synaptic candidate; native ORN outputs elsewhere persist',
                initialization=initialization, component_enabled=enabled,
                graph_and_archived_weights_preserved=True, new_states_initialization_is_explicit=True,
                parameter_fitting_to_sensory_or_motor_result=False, biological_validation=False,
                experimental_rate_benchmark='Kazama and Wilson 2008, VM2 nerve stimulation: 7 Hz background, 20 and 50 Hz test rates; transferred as device settings, not measured DM1 transduction')
            obj.source_identity = {name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj.hybrid.hold_afferent_rate(obj.afferent_rate())
            guard_visual_session(obj)
            obj._validate()
            obj._validate_pending()
            obj._validate_afferent_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def _validate(self):
        super()._validate()
        p = self.probe
        if type(p['afferent_connected']) is not bool:
            raise ValueError('Afferent connection must be an explicit boolean')
        bounds = np.asarray([p['afferent_baseline_hz'], p['afferent_pulse_hz']], dtype=float)
        if not np.isfinite(bounds).all() or bounds[0] < 0. or bounds[1] < bounds[0] or bounds[1] > self.hybrid.source_caps.min():
            raise ValueError('Invalid prosthetic afferent rate range')
        if self.probe['mask_enabled']:
            raise ValueError('This preparation does not apply the gamma current mask')
        if self.probe['component_enabled'] != self.hybrid.orn_pn_synaptic_manifest['enabled']:
            raise ValueError('Session and neural component connection disagree')

    def additional_drive(self):
        # The additional stimulus belongs to the afferent device. It must not
        # also be injected into the native ORN membrane current.
        return 0.

    def afferent_rate(self):
        p = self.probe
        baseline = p['afferent_baseline_hz']
        delta = p['afferent_pulse_hz']-baseline if p['afferent_connected'] else 0.
        return baseline+pulse_amplitude(self.time_ns, p['start_ns'], p['onset_ms'], p['pulse_ms'], delta)

    def step(self):
        used = self.afferent_rate()
        self.hybrid.hold_afferent_rate(used)
        row = super().step()
        self.hybrid.hold_afferent_rate(self.afferent_rate())
        row['prosthetic_afferent_rate_used_hz'] = used
        row['prosthetic_afferent_rate_pending_hz'] = self.afferent_rate()
        return row

    def _validate_afferent_pending(self):
        p = self.probe
        if type(p['afferent_connected']) is not bool:
            raise ValueError('Afferent connection must be an explicit boolean')
        if not np.array_equal(self.hybrid.held_afferent_rate_hz,
                              np.full(len(self.hybrid.orn_source_ids), self.afferent_rate())):
            raise ValueError('Pending afferent input and device clock disagree')

    def state_dict(self):
        result = super().state_dict()
        result["schema"] = SCHEMA
        result["probe"] = copy.deepcopy(self.probe)
        return result

    def save(self, path):
        self._validate_afferent_pending()
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
                prosthesis_kind="centered_position_servo",
                diagnostic_current_mask=False, plasticity_updates_enabled=False,
                orn_pn_synaptic_component=self.probe["component_enabled"],
                prosthetic_afferent=True, native_olfactory_transduction_claimed=False,
                afferent_connected=self.probe["afferent_connected"],
                synaptic_initialization=self.probe["initialization"],
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
                   "plasticity", "proprioception", "used_light", "rotor", "probe"}
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
        kinds = {kind.SCHEMA: kind for kind in (ProstheticOlfactoryBrain, GpuProstheticOlfactoryBrain)}
        obj.hybrid = kinds[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        if isinstance(obj.hybrid, GpuGradedDescendingBrain) and obj.config["electrical_backend_identity"] != obj.hybrid.backend_identity():
            raise ValueError("Recorded local electrical GPU backend differs")
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.probe = copy.deepcopy(state["probe"])
        obj._apply_probe_mask()
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.rotor = CyborgCenteredServo.from_state(state["rotor"])
            obj.eyes = CyborgEye.from_state(obj.brain, obj.body, state["eyes"], obj.rotor)
            if obj.config["looming_camera_fixed"]:
                obj.eyes = FixedCyborgEye.adopt_fixed(obj.eyes, obj.rotor, obj.initial_camera)
            obj.light_world = CyborgLoomingWorld.from_state(state["light_world"])
            obj.muscles = ContractileTibia.from_state(obj.body, state["muscles"])
            obj.used_light = list(state["used_light"])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            obj._validate_afferent_pending()
            if manifest["time_ns"] != obj.time_ns or manifest["output_connected"] != obj.output_connected:
                raise ValueError("Manifest clock or output connection differs")
            return obj
        except BaseException:
            obj.body.close()
            raise
