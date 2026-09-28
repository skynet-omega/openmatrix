"""Present a physical pitch-motion scene to the same continuing retinal CNS.

No neuronal or mechanical component is replaced. Only the light world and its
next retinal sample change at adoption; an optional neural-output cut sends
zero to the still-powered position servo and preserves mechanical momentum.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile

import numba
import numpy as np

from retinal_live_session import (
    RetinalLiveSession, SOURCES as PARENT_SOURCES, _preserved, _digest,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CyborgCenteredServo, RetinalTransductionBrain, GpuRetinalTransductionBrain)
from prosthetic_olfactory_session import ProstheticOlfactorySession
from kcgamma_regional_brain import KcGammaRegionalBrain, GpuKcGammaRegionalBrain
from t4_gaba_brain import T4GabaBrain, GpuT4GabaBrain
from cardinal_grating_world import CardinalGratingWorld

SCHEMA = 'matrix_retinal_motion_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES) + (
    'cardinal_grating_world.py', 'retinal_motion_session.py')))
CONDITIONS = ('elevation_positive', 'elevation_negative', 'elevation_static')


def _scene_preserved(session):
    result = _preserved(session)
    result.pop('optical_world')
    excluded = {'config', 'source_identity', 'intervention', 'pending_light',
                'output_connected', 'pending_cyborg_command'}
    result['all_inherited_scalars'] = _digest({
        key: getattr(session, key) for key in SCALARS | CYBORG_SCALARS
        if key not in excluded})
    result['exact_neural_parameters'] = _digest(session.hybrid.parameters)
    return result


class RetinalMotionSession(RetinalLiveSession):
    @classmethod
    def from_checkpoint(cls, path, *, condition='elevation_positive', output_connected=True):
        if condition not in CONDITIONS or type(output_connected) is not bool:
            raise ValueError('Requires a declared elevation condition and boolean motor connection')
        path = Path(path).resolve()
        parent = RetinalLiveSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            if parent.config['looming_camera_fixed']:
                raise ValueError('Retinal motion requires the inherited live servo camera')
            before = _scene_preserved(parent)
            previous_world = copy.deepcopy(parent.light_world.state_dict())
            previous_light = _digest(parent.pending_light)
            previous_command = parent.pending_cyborg_command
            rotation, centers = parent.eyes.pose()
            center = .5 * (np.asarray(centers['L']) + np.asarray(centers['R']))
            obj.light_world = CardinalGratingWorld(obj.time_ns, center, rotation, condition)
            obj.pending_light = obj.eyes.sample(obj.light_world, obj.time_ns)
            obj.output_connected = output_connected
            obj.pending_cyborg_command = obj.cyborg_command()
            after = _scene_preserved(obj)
            checks = {key: value == after[key] for key, value in before.items()}
            if not all(checks.values()):
                raise AssertionError(checks)
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='RETINAL_MOTION_LIVE_v1',
                retinal_motion_condition=condition,
                retinal_motion_world_schema=CardinalGratingWorld.SCHEMA,
                retinal_motion_start_ns=obj.time_ns,
                biological_validation=False, learning_demonstrated=False)
            obj.intervention = dict(
                previous_intervention=copy.deepcopy(obj.intervention),
                operation='Present a physical elevation grating to the continuing retinal CNS without resetting neural, body or servo states',
                time_ns=obj.time_ns, parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),
                previous_world=previous_world, previous_pending_light_sha256=previous_light,
                pending_light_sha256=_digest(obj.pending_light),
                previous_pending_cyborg_command=previous_command,
                output_connected=output_connected, preserved_state_checks=checks,
                new_neural_states=0, neural_equations_changed=False,
                neural_parameter_fitting=False, optical_apparatus_replaced=False,
                visual_scene_replaced=True,
                cut_scope='Neural command becomes zero; the powered servo returns toward its unchanged mounting angle, preserving angle and angular velocity at adoption',
                stimulus_scope='Physical exploratory elevation grating, not a replication of a physiological recording or a visual controller',
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
        # Retain the ancestral neural/body/servo checks while replacing only
        # KcGammaLiveSession's exact looming-world restriction.
        ProstheticOlfactorySession._validate(self)
        if not isinstance(self.hybrid, (KcGammaRegionalBrain, GpuKcGammaRegionalBrain)):
            raise ValueError('Retinal motion requires its saved regional KC component')
        m = self.hybrid.regional_manifest
        if (self.config['kcgamma_regional_enabled'] != m['enabled']
                or self.config['kcgamma_regional_eta'] != m['eta']
                or self.config['kcgamma_regional_config_sha256'] != m['config_source']['sha256']
                or m['parameters'] != m['config']['parameters']
                or m['activity_dependent'] is not True
                or self.plasticity.enabled or self.mode != 'live'
                or self.brain.n_neurons != 166700):
            raise ValueError('Complete live CNS, regional configuration or learning policy differs')
        if not isinstance(self.hybrid, (T4GabaBrain, GpuT4GabaBrain)):
            raise ValueError('Retinal motion requires its saved T4 GABA component')
        m = self.hybrid.t4_gaba_manifest
        if (type(m['enabled']) is not bool
                or self.config['t4_gaba_enabled'] != m['enabled']
                or self.config['t4_gaba_E_mv'] != m['E_GABA_mV']
                or m['E_GABA_mV'] != -68.):
            raise ValueError('T4 GABA configuration differs from its component')
        if not isinstance(self.hybrid, (RetinalTransductionBrain, GpuRetinalTransductionBrain)):
            raise ValueError('Retinal motion requires its saved phototransduction component')
        m = self.hybrid.retinal_transduction_manifest
        clamp_active = not m['enabled'] and self.config['retinal_port_enabled']
        if (type(m['enabled']) is not bool
                or self.config['retinal_transduction_enabled'] != m['enabled']
                or self.config['retinal_voltage_clamp_active'] != clamp_active):
            raise ValueError('Retinal configuration differs from its component')
        world = self.light_world
        if (type(world) is not CardinalGratingWorld
                or world.condition not in CONDITIONS
                or self.config['retinal_motion_world_schema'] != world.SCHEMA
                or self.config['retinal_motion_condition'] != world.condition
                or self.config['retinal_motion_start_ns'] != world.start_ns
                or world.start_ns > self.time_ns
                or self.config['looming_camera_fixed']
                or type(self.rotor) is not CyborgCenteredServo):
            raise ValueError('Saved motion scene, live optical coupling or servo differs')
        world._validate()
        reference = CardinalGratingWorld(world.start_ns, world.center_mm, world.frame, world.condition)
        if _digest(reference.state_dict()) != _digest(world.state_dict()):
            raise ValueError('Motion parameters differ from the fixed cardinal protocol')

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def _manifest_fields(self):
        result = super()._manifest_fields()
        result.update(schema=SCHEMA, physical_looming_stimulus=False,
            physical_motion_stimulus=True, optical_apparatus_replaced=False,
            visual_scene_replaced=True, motion_condition=self.light_world.condition,
            motion_start_ns=self.light_world.start_ns,
            motion_world_schema=self.light_world.SCHEMA,
            motor_cut_scope='Neutral command; powered servo retains inherited mechanics')
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after retinal motion session creation')
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
            raise ValueError('Unsupported retinal motion checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete retinal motion checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete retinal motion state')
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
            raise ValueError('Unknown retinal motion neural component schema')
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
                raise ValueError('Manifest and retinal motion neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
