"""Continue the existing live MaleCNS with its regional KCgamma candidate.

The optical apparatus, body, sensory schedules and pending causal inputs are
inherited without a new experiment epoch. This adopts the existing regional
axonal hypothesis, not the separate NEURON cholinergic single-cell model.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from prosthetic_olfactory_session import (
    ProstheticOlfactorySession, SOURCES as PARENT_SOURCES,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CyborgCenteredServo, CyborgLoomingWorld, GpuProstheticOlfactoryBrain)
from cyborg_visual_session import preservation_fingerprints, _digest
from kcgamma_regional_brain import (
    KcGammaRegionalBrain, GpuKcGammaRegionalBrain, DEFAULT_CONFIG)

SCHEMA = 'matrix_kcgamma_live_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES) + (
    'kcgamma_axon_candidate.py', 'kcgamma_regional_brain.py',
    'kcgamma_live_session.py')))


def _preserved(session):
    result = preservation_fingerprints(session)
    # The general optical-migration helper intentionally excludes some camera
    # and pending-input state; this migration must preserve that state too.
    scalars = SCALARS | CYBORG_SCALARS
    result.update(
        all_inherited_scalars=_digest({k: getattr(session, k) for k in scalars}),
        optical_world=_digest(session.light_world.state_dict()),
        rotor=_digest(session.rotor.state_dict()),
        sensory_probe=_digest(session.probe),
        effective_weights=_digest(session.hybrid.weights64))
    return result


class KcGammaLiveSession(ProstheticOlfactorySession):
    @classmethod
    def from_checkpoint(cls, path, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('Regional connection must be an explicit boolean')
        path = Path(path).resolve()
        parent = ProstheticOlfactorySession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            if obj.mode != 'live' or obj.brain.n_neurons != 166700:
                raise ValueError('Requires the complete live 166700-neuron MaleCNS')
            if type(obj.light_world) is not CyborgLoomingWorld:
                raise ValueError('Requires the inherited live looming world')
            before = _preserved(parent)
            old = parent.hybrid
            previous_state = old.state.copy()
            inherited = old.state_dict()
            inherited.pop('schema')
            inherited.pop('state')
            inherited_digest = _digest(inherited)
            del inherited
            kind = (GpuKcGammaRegionalBrain
                    if isinstance(old, GpuProstheticOlfactoryBrain)
                    else KcGammaRegionalBrain)
            # No eta, tolerance, sensory or parameter overrides: retain the
            # published project candidate configuration exactly as stored.
            obj.hybrid = kind.adopt(old, config=DEFAULT_CONFIG, enabled=enabled,
                                    activity_dependent=True)
            size = obj.hybrid.regional_parent_state_size
            after = _preserved(obj)
            checks = {k: v == after[k] for k, v in before.items()
                      if k != 'hybrid_except_solver_tolerances'}
            checks['inherited_neural_state_prefix'] = (
                size == len(previous_state)
                and np.array_equal(previous_state, obj.hybrid.state[:size]))
            adopted = obj.hybrid.state_dict()
            for key in ('schema', 'state', 'regional_manifest'):
                adopted.pop(key)
            checks['all_inherited_hybrid_metadata'] = (
                inherited_digest == _digest(adopted))
            if not all(checks.values()):
                raise AssertionError(checks)
            m = obj.hybrid.regional_manifest
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='KCGAMMA_LIVE_REGIONAL_INTEGRATION_v1',
                kcgamma_regional_enabled=enabled, kcgamma_regional_eta=m['eta'],
                kcgamma_regional_config_sha256=m['config_source']['sha256'],
                biological_validation=False, learning_demonstrated=False)
            if hasattr(obj.hybrid, 'backend_identity'):
                obj.config['electrical_backend_identity'] = obj.hybrid.backend_identity()
            obj.intervention = dict(
                previous_intervention=copy.deepcopy(obj.intervention),
                operation='Continue live CNS with the existing regional KCgamma axonal hypothesis; preserve apparatus and every pending input',
                time_ns=obj.time_ns, parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled, eta=m['eta'], preserved_state_checks=checks,
                regional_manifest_sha256=_digest(m),
                initialization=m['initialization'],
                parameter_status=m['parameter_status'],
                separate_neuron_cholinergic_port_integrated=False,
                weight_updates_enabled=False, biological_validation=False)
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
        if not isinstance(self.hybrid, (KcGammaRegionalBrain, GpuKcGammaRegionalBrain)):
            raise ValueError('Live regional session requires its saved neural component')
        m = self.hybrid.regional_manifest
        if (self.config['kcgamma_regional_enabled'] != m['enabled']
                or self.config['kcgamma_regional_eta'] != m['eta']
                or self.config['kcgamma_regional_config_sha256'] != m['config_source']['sha256']
                or m['parameters'] != m['config']['parameters']
                or m['activity_dependent'] is not True
                or self.plasticity.enabled
                or self.mode != 'live'
                or self.brain.n_neurons != 166700
                or type(self.light_world) is not CyborgLoomingWorld):
            raise ValueError('Live state, frozen regional configuration or learning policy differs')

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after live session creation')
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
            m = self.hybrid.regional_manifest
            manifest = dict(schema=SCHEMA, time_ns=self.time_ns,
                neuron_count=self.brain.n_neurons, stored_edges=self.brain.W.nnz,
                mapped_photoreceptors=len(self.eyes.ids), mode=self.mode,
                output_connected=self.output_connected, engineering_prosthesis=True,
                experimental_integration=True, complete_cns_coupled=True,
                camera_fixed=self.config['looming_camera_fixed'],
                physical_looming_stimulus=True, optical_apparatus_replaced=False,
                pvlp_adaptation_enabled=self.hybrid.pvlp_adaptation_manifest['enabled'],
                prosthesis_kind='centered_position_servo', diagnostic_current_mask=False,
                plasticity_updates_enabled=False, learning_demonstrated=False,
                kcgamma_regional_enabled=m['enabled'], kcgamma_regional_eta=m['eta'],
                kcgamma_regional_config_sha256=m['config_source']['sha256'],
                kcgamma_regional_cells=len(m['target_ids']),
                regional_parameter_status=m['parameter_status'],
                separate_neuron_cholinergic_port_integrated=False,
                orn_pn_synaptic_component=self.probe['component_enabled'],
                prosthetic_afferent=True, native_olfactory_transduction_claimed=False,
                afferent_connected=self.probe['afferent_connected'],
                synaptic_initialization=self.probe['initialization'],
                retinal_port_enabled=self.config['retinal_port_enabled'],
                biological_retina_reconstructed=False, biological_validation=False,
                animal_ability_demonstrated=False,
                files={str(p.relative_to(staging)): sha256(p)
                       for p in sorted(staging.rglob('*')) if p.is_file()})
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
            raise ValueError('Unsupported live regional checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete live regional checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete live regional state')
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
        kinds = {kind.SCHEMA: kind for kind in (KcGammaRegionalBrain, GpuKcGammaRegionalBrain)}
        obj.hybrid = kinds[state['hybrid']['schema']].from_state(obj.brain, state['hybrid'])
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
            m = obj.hybrid.regional_manifest
            expected = dict(time_ns=obj.time_ns, output_connected=obj.output_connected,
                neuron_count=obj.brain.n_neurons, stored_edges=obj.brain.W.nnz,
                camera_fixed=obj.config['looming_camera_fixed'],
                kcgamma_regional_enabled=m['enabled'], kcgamma_regional_eta=m['eta'],
                kcgamma_regional_config_sha256=m['config_source']['sha256'],
                kcgamma_regional_cells=len(m['target_ids']),
                biological_validation=False, plasticity_updates_enabled=False,
                learning_demonstrated=False, physical_looming_stimulus=True,
                separate_neuron_cholinergic_port_integrated=False)
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError('Manifest and live neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
