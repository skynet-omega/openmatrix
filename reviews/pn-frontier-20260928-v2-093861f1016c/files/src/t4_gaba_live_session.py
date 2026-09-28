"""Continue the live canonical CNS with the T4 GABA reversal correction.

The complete regional Kenyon state, body, optical world and pending sensory
inputs are inherited. No stimulus, cell state or simulation epoch is reset.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from kcgamma_live_session import (
    KcGammaLiveSession, SOURCES as PARENT_SOURCES, _preserved, _digest,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CyborgCenteredServo, CyborgLoomingWorld, GpuKcGammaRegionalBrain)
from t4_gaba_brain import T4GabaBrain, GpuT4GabaBrain

SCHEMA = 'matrix_t4_gaba_live_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES) + (
    't4_gaba_brain.py', 't4_gaba_live_session.py')))


class T4GabaLiveSession(KcGammaLiveSession):
    @classmethod
    def from_checkpoint(cls, path, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('T4 GABA correction requires an explicit boolean')
        path = Path(path).resolve()
        parent = KcGammaLiveSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            before = _preserved(parent)
            old = parent.hybrid
            previous_state = old.state.copy()
            inherited = old.state_dict()
            inherited.pop('schema')
            inherited_digest = _digest(inherited)
            del inherited
            kind = (GpuT4GabaBrain if isinstance(old, GpuKcGammaRegionalBrain)
                    else T4GabaBrain)
            obj.hybrid = kind.adopt(old, enabled=enabled)
            after = _preserved(obj)
            checks = {k: v == after[k] for k, v in before.items()
                      if k != 'hybrid_except_solver_tolerances'}
            checks['complete_inherited_neural_state'] = np.array_equal(
                previous_state, obj.hybrid.state)
            adopted = obj.hybrid.state_dict()
            for key in ('schema', 't4_gaba_manifest'):
                adopted.pop(key)
            checks['all_inherited_hybrid_metadata'] = (
                inherited_digest == _digest(adopted))
            if not all(checks.values()):
                raise AssertionError(checks)
            m = obj.hybrid.t4_gaba_manifest
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='T4_GABA_LIVE_INTEGRATION_v1',
                t4_gaba_enabled=enabled, t4_gaba_E_mv=m['E_GABA_mV'],
                biological_validation=False, learning_demonstrated=False)
            if hasattr(obj.hybrid, 'backend_identity'):
                obj.config['electrical_backend_identity'] = obj.hybrid.backend_identity()
            obj.intervention = dict(
                previous_intervention=copy.deepcopy(obj.intervention),
                operation='Continue canonical live CNS with T4 GABA reversal correction; preserve regional KC states, apparatus and pending inputs',
                time_ns=obj.time_ns, parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled, E_GABA_mV=m['E_GABA_mV'],
                preserved_state_checks=checks,
                t4_gaba_manifest_sha256=_digest(m),
                inherited_regional_manifest_sha256=_digest(obj.hybrid.regional_manifest),
                new_neural_states=0, weight_updates_enabled=False,
                biological_validation=False)
            obj.source_identity = {name: sha256(ROOT/'src'/name) for name in SOURCES}
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
        if not isinstance(self.hybrid, (T4GabaBrain, GpuT4GabaBrain)):
            raise ValueError('Live T4 session requires its saved GABA component')
        m = self.hybrid.t4_gaba_manifest
        if (type(m['enabled']) is not bool
                or self.config['t4_gaba_enabled'] != m['enabled']
                or self.config['t4_gaba_E_mv'] != m['E_GABA_mV']
                or m['E_GABA_mV'] != -68.):
            raise ValueError('Live T4 GABA configuration differs from its component')

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def _manifest_fields(self):
        regional = self.hybrid.regional_manifest
        gaba = self.hybrid.t4_gaba_manifest
        return dict(schema=SCHEMA, time_ns=self.time_ns,
            neuron_count=self.brain.n_neurons, stored_edges=self.brain.W.nnz,
            mapped_photoreceptors=len(self.eyes.ids), mode=self.mode,
            output_connected=self.output_connected, engineering_prosthesis=True,
            experimental_integration=True, complete_cns_coupled=True,
            camera_fixed=self.config['looming_camera_fixed'],
            physical_looming_stimulus=True, optical_apparatus_replaced=False,
            pvlp_adaptation_enabled=self.hybrid.pvlp_adaptation_manifest['enabled'],
            prosthesis_kind='centered_position_servo', diagnostic_current_mask=False,
            plasticity_updates_enabled=False, learning_demonstrated=False,
            kcgamma_regional_enabled=regional['enabled'],
            kcgamma_regional_eta=regional['eta'],
            kcgamma_regional_config_sha256=regional['config_source']['sha256'],
            kcgamma_regional_cells=len(regional['target_ids']),
            regional_parameter_status=regional['parameter_status'],
            t4_gaba_enabled=gaba['enabled'], t4_gaba_E_mv=gaba['E_GABA_mV'],
            t4_gaba_cells=len(gaba['target_ids']),
            separate_neuron_cholinergic_port_integrated=False,
            orn_pn_synaptic_component=self.probe['component_enabled'],
            prosthetic_afferent=True, native_olfactory_transduction_claimed=False,
            afferent_connected=self.probe['afferent_connected'],
            synaptic_initialization=self.probe['initialization'],
            retinal_port_enabled=self.config['retinal_port_enabled'],
            biological_retina_reconstructed=False, biological_validation=False,
            animal_ability_demonstrated=False)

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after live T4 session creation')
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f'Preserving checkpoint {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.'+path.name+'-', dir=path.parent))
        try:
            self.brain.save_checkpoint(staging/'brain')
            write_state(staging/'session', self.state_dict())
            (staging/'source').mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT/'src'/name, staging/'source'/name)
            manifest = self._manifest_fields()
            manifest['files'] = {str(p.relative_to(staging)): sha256(p)
                                for p in sorted(staging.rglob('*')) if p.is_file()}
            (staging/'manifest.json').write_text(
                json.dumps(manifest, indent=2, allow_nan=False)+'\n')
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest.get('schema') != SCHEMA:
            raise ValueError('Unsupported live T4 GABA checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete live T4 GABA checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete live T4 GABA state')
        if state['source_identity'] != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Checkpoint requires its archived source versions')
        obj = cls()
        obj.brain = AnatomicalRateBrain.load_checkpoint(path/'brain')
        for key in SCALARS | CYBORG_SCALARS:
            setattr(obj, key, state[key])
        if obj.config['numba_version'] != numba.__version__:
            raise ValueError('Checkpoint requires its recorded numerical runtime')
        if (obj.config['numerical_backend'] == 'cuda_fp64'
                and obj.config['backend_identity'] != GpuVisualBrain.backend_identity()):
            raise ValueError('Recorded GPU backend differs')
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state['world']):
            raise ValueError('Incomplete inherited odor world')
        obj.world.__dict__.update(state['world'])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state['plasticity'])
        kinds = {kind.SCHEMA: kind for kind in (T4GabaBrain, GpuT4GabaBrain)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown live T4 neural component schema')
        obj.hybrid = kind.from_state(obj.brain, state['hybrid'])
        if (isinstance(obj.hybrid, GpuGradedDescendingBrain)
                and obj.config['electrical_backend_identity'] != obj.hybrid.backend_identity()):
            raise ValueError('Recorded local electrical GPU backend differs')
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.probe = copy.deepcopy(state['probe'])
        obj._apply_probe_mask()
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state['body'])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(
                obj.brain, obj.body, state['proprioception'])
            obj.rotor = CyborgCenteredServo.from_state(state['rotor'])
            obj.eyes = CyborgEye.from_state(obj.brain, obj.body, state['eyes'], obj.rotor)
            if obj.config['looming_camera_fixed']:
                obj.eyes = FixedCyborgEye.adopt_fixed(obj.eyes, obj.rotor, obj.initial_camera)
            obj.light_world = CyborgLoomingWorld.from_state(state['light_world'])
            obj.muscles = ContractileTibia.from_state(obj.body, state['muscles'])
            obj.used_light = list(state['used_light'])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            obj._validate_afferent_pending()
            if any(manifest.get(k) != v for k, v in obj._manifest_fields().items()):
                raise ValueError('Manifest and live T4 neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
