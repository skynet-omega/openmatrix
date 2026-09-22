"""Continue the one canonical CNS with a conductance-driven CvN7 rate proxy.

No state, scene or motor device is reset. The local quasistationary LIF
transfer remains a declared prosthesis; it is not an event/spike simulation.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import numba
import numpy as np

from retinal_motion_session import (
    RetinalMotionSession, SOURCES as PARENT_SOURCES, _preserved, _digest,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CyborgCenteredServo, RetinalTransductionBrain, GpuRetinalTransductionBrain,
    CardinalGratingWorld)
from cvn7_conductance_brain import Cvn7ConductanceBrain, GpuCvn7ConductanceBrain

SCHEMA = 'matrix_cvn7_conductance_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES)+(
    'cvn7_conductance_brain.py','cvn7_conductance_session.py')))


class Cvn7ConductanceSession(RetinalMotionSession):
    @classmethod
    def from_checkpoint(cls, path, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('CvN7 enable flag must be boolean')
        path = Path(path).resolve()
        parent = RetinalMotionSession.load(path)
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
            kind = GpuCvn7ConductanceBrain if isinstance(old,GpuRetinalTransductionBrain) else Cvn7ConductanceBrain
            obj.hybrid = kind.adopt(old,enabled=enabled)
            after = _preserved(obj)
            checks = {k:v==after[k] for k,v in before.items() if k!='hybrid_except_solver_tolerances'}
            checks['every_neural_coordinate_unchanged'] = np.array_equal(previous_state,obj.hybrid.state)
            inherited = obj.hybrid.state_dict()
            inherited.pop('schema'); inherited.pop('cvn7_manifest')
            checks['every_inherited_hybrid_field_unchanged'] = inherited_digest==_digest(inherited)
            if not all(checks.values()):
                raise AssertionError(checks)
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='CVN7_CONDUCTANCE_LIVE_v1',cvn7_conductance_enabled=enabled,
                cvn7_conductance_policy=obj.hybrid.cvn7_manifest['policy'],
                biological_validation=False,learning_demonstrated=False)
            if hasattr(obj.hybrid,'backend_identity'):
                obj.config['electrical_backend_identity'] = obj.hybrid.backend_identity()
            obj.intervention = dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Replace the two canonical CvN7 rate targets by a conductance-driven quasistationary LIF proxy; retain all incoming pairs and every neural/body/retinal state',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,cvn7_manifest_sha256=_digest(obj.hybrid.cvn7_manifest),
                new_neural_states=0,neural_equations_changed=enabled,visual_scene_replaced=False,
                neural_parameter_fitting=False,weight_updates_enabled=False,biological_validation=False,
                approximation='A filtered stationary-rate approximation, not a spike train or recovered individual-cell physiology.')
            obj.source_identity = {name:sha256(ROOT/'src'/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate(); obj._validate_pending(); obj._validate_afferent_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,(Cvn7ConductanceBrain,GpuCvn7ConductanceBrain)):
            raise ValueError('The saved CvN7 component is required')
        m = self.hybrid.cvn7_manifest
        if (type(m['enabled']) is not bool
                or self.config['cvn7_conductance_enabled'] != m['enabled']
                or self.config['cvn7_conductance_policy'] != m['policy']):
            raise ValueError('CvN7 configuration and saved component disagree')

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def _manifest_fields(self):
        result = super()._manifest_fields()
        m = self.hybrid.cvn7_manifest
        result.update(schema=SCHEMA,cvn7_conductance_enabled=m['enabled'],
            cvn7_conductance_policy=m['policy'],cvn7_conductance_cells=len(m['target_ids']),
            cvn7_parameter_status=m['parameter_status'],cvn7_event_spikes_simulated=False,
            cvn7_new_state_variables=0)
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after CvN7 conductance session creation')
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
            raise ValueError('Unsupported CvN7 conductance checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete CvN7 conductance checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete CvN7 conductance state')
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
        kinds = {kind.SCHEMA: kind for kind in (Cvn7ConductanceBrain, GpuCvn7ConductanceBrain)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown CvN7 conductance neural component schema')
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
            obj.light_world = CardinalGratingWorld.from_state(state['light_world'])
            obj.muscles = ContractileTibia.from_state(obj.body, state['muscles'])
            obj.used_light = list(state['used_light'])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            obj._validate_afferent_pending()
            if any(manifest.get(k) != v for k, v in obj._manifest_fields().items()):
                raise ValueError('Manifest and CvN7 conductance neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
