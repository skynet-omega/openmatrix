"""Continue the full canonical CNS with persistent phototransduction states.

Preserve all inherited neuronal coordinates, KC regional states, T4 inhibition,
body, optical world and pending inputs. New receptor memories start only at the
explicit adoption boundary; the former voltage clamp remains a bypass option.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from t4_gaba_live_session import (
    T4GabaLiveSession, SOURCES as PARENT_SOURCES, _preserved, _digest,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CyborgCenteredServo, CyborgLoomingWorld, GpuT4GabaBrain)
from retinal_transduction_brain import (
    RetinalTransductionBrain, GpuRetinalTransductionBrain)

SCHEMA = 'matrix_retinal_live_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES) + (
    'retinal_phototransduction.py', 'retinal_transduction_brain.py',
    'retinal_live_session.py')))


class RetinalLiveSession(T4GabaLiveSession):
    @classmethod
    def from_checkpoint(cls, path, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('Retinal transduction requires an explicit boolean')
        path = Path(path).resolve()
        parent = T4GabaLiveSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            before = _preserved(parent)
            old = parent.hybrid
            previous_state = old.state.copy()
            inherited = old.state_dict()
            for key in ('schema', 'state'):
                inherited.pop(key)
            inherited_digest = _digest(inherited)
            del inherited
            kind = (GpuRetinalTransductionBrain if isinstance(old, GpuT4GabaBrain)
                    else RetinalTransductionBrain)
            obj.hybrid = kind.adopt(old, enabled=enabled, initial_light=parent.pending_light)
            m = obj.hybrid.retinal_transduction_manifest
            size = m['parent_state_size']
            after = _preserved(obj)
            checks = {k: v == after[k] for k, v in before.items()
                      if k != 'hybrid_except_solver_tolerances'}
            checks['complete_inherited_neural_state_prefix'] = (
                size == len(previous_state)
                and np.array_equal(previous_state, obj.hybrid.state[:size]))
            adopted = obj.hybrid.state_dict()
            for key in ('schema', 'state', 'retinal_transduction_manifest'):
                adopted.pop(key)
            checks['all_inherited_hybrid_metadata'] = (
                inherited_digest == _digest(adopted))
            if not all(checks.values()):
                raise AssertionError(checks)
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='RETINAL_TRANSDUCTION_LIVE_INTEGRATION_v1',
                retinal_transduction_enabled=enabled,
                retinal_voltage_clamp_active=(not enabled and obj.config['retinal_port_enabled']),
                biological_validation=False, learning_demonstrated=False)
            if hasattr(obj.hybrid, 'backend_identity'):
                obj.config['electrical_backend_identity'] = obj.hybrid.backend_identity()
            obj.intervention = dict(
                previous_intervention=copy.deepcopy(obj.intervention),
                operation='Continue canonical CNS with retinal transduction; preserve every inherited neural state, regional KC component, T4 GABA correction, apparatus and pending input',
                time_ns=obj.time_ns, parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled, preserved_state_checks=checks,
                retinal_transduction_manifest_sha256=_digest(m),
                inherited_regional_manifest_sha256=_digest(obj.hybrid.regional_manifest),
                inherited_t4_gaba_manifest_sha256=_digest(obj.hybrid.t4_gaba_manifest),
                parameter_status=copy.deepcopy(m['parameter_status']),
                new_neural_states=len(obj.hybrid.state)-size,
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
        if not isinstance(self.hybrid, (RetinalTransductionBrain, GpuRetinalTransductionBrain)):
            raise ValueError('Live retinal session requires its saved transduction component')
        m = self.hybrid.retinal_transduction_manifest
        clamp_active = not m['enabled'] and self.config['retinal_port_enabled']
        if (type(m['enabled']) is not bool
                or self.config['retinal_transduction_enabled'] != m['enabled']
                or self.config['retinal_voltage_clamp_active'] != clamp_active):
            raise ValueError('Live retinal configuration differs from its component')

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def _manifest_fields(self):
        result = super()._manifest_fields()
        m = self.hybrid.retinal_transduction_manifest
        result.update(schema=SCHEMA, retinal_transduction_enabled=m['enabled'],
            retinal_transduction_cells=len(m['target_ids']),
            retinal_transduction_parameter_status=copy.deepcopy(m['parameter_status']),
            retinal_voltage_clamp_active=(not m['enabled'] and self.config['retinal_port_enabled']))
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after live retinal session creation')
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
            raise ValueError('Unsupported live retinal checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete live retinal checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete live retinal state')
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
        kinds = {kind.SCHEMA: kind for kind in (RetinalTransductionBrain, GpuRetinalTransductionBrain)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown live retinal neural component schema')
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
                raise ValueError('Manifest and live retinal neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
