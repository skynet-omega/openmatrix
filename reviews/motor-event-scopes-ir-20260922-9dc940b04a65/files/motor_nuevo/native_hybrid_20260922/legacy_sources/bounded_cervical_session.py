"""Continue the complete CNS while replacing only the cervical apparatus.

The finite travel range is engineering, not recovered fly anatomy. All neural
states, equations, weights and physical pose survive adoption without reset.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import numba
import numpy as np

from cvn7_conductance_session import (
    Cvn7ConductanceSession, SOURCES as PARENT_SOURCES, _digest,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    RetinalTransductionBrain, GpuRetinalTransductionBrain,
    CardinalGratingWorld, Cvn7ConductanceBrain, GpuCvn7ConductanceBrain)
from cyborg_bounded_servo import CyborgBoundedServo
from prosthetic_olfactory_session import ProstheticOlfactorySession
from kcgamma_regional_brain import KcGammaRegionalBrain, GpuKcGammaRegionalBrain
from t4_gaba_brain import T4GabaBrain, GpuT4GabaBrain
from retinal_motion_session import CONDITIONS

SCHEMA = 'matrix_bounded_cervical_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES) + (
    'cyborg_bounded_servo.py', 'bounded_cervical_session.py')))


def _fingerprints(session, excluded):
    result = {k: _digest(v) for k, v in session.state_dict().items() if k not in excluded}
    result['canonical_topology'] = _digest((session.brain.node_ids, session.brain.W.indptr,
                                         session.brain.W.indices, session.brain.W.data))
    result['brain_rates'] = _digest(session.brain.rates)
    result['physical_eye_pose'] = _digest(session.eyes.pose())
    return result


class BoundedCervicalSession(Cvn7ConductanceSession):
    @classmethod
    def from_checkpoint(cls, path):
        path = Path(path).resolve()
        parent = Cvn7ConductanceSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema', 'config', 'source_identity', 'intervention',
                        'rotor', 'pending_cyborg_command'}
            before = _fingerprints(parent, excluded)
            old = parent.rotor
            physical_names = ('angle_rad', 'omega_rad_s', 'start_ns', 'time_ns',
                              'initial_rotation', 'initial_centers', 'pivot_mm')
            physical = {k: _digest(getattr(old, k)) for k in physical_names}
            raw = CyborgBoundedServo.map_release(obj.brain.node_ids, obj.hybrid.release())['command']
            obj.rotor = CyborgBoundedServo.from_predecessor(old,
                initial_command=float(raw) if obj.output_connected else 0.)
            obj.eyes = CyborgEye.adopt(parent.eyes, obj.rotor)
            obj.pending_cyborg_command = obj.cyborg_command()
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='BOUNDED_CERVICAL_LIVE_v1', bounded_cervical_enabled=True,
                cervical_device_schema=obj.rotor.state_dict()['schema'],
                cervical_reference_angle_rad=obj.rotor.reference_angle_rad,
                cervical_half_range_rad=obj.rotor.half_range_rad,
                biological_validation=False, learning_demonstrated=False)
            after = _fingerprints(obj, excluded)
            checks = {k: v == after[k] for k, v in before.items()}
            checks.update({'device_' + k: v == _digest(getattr(obj.rotor, k))
                           for k, v in physical.items()})
            if not all(checks.values()):
                raise AssertionError(checks)
            obj.intervention = dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Replace only the incompatible high-gain cervical device with a fixed bounded position apparatus; preserve the complete neural and body history',
                time_ns=obj.time_ns, parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),
                preserved_state_checks=checks, new_neural_states=0,
                neural_equations_changed=False, neural_parameters_changed=False,
                neural_parameter_fitting=False, weight_updates_enabled=False,
                visual_scene_replaced=False, optical_apparatus_replaced=True,
                physical_pose_reset=False, initial_pending_command=parent.pending_cyborg_command,
                replacement_pending_command=obj.pending_cyborg_command,
                device_parameters=obj.rotor.parameters(),
                biological_validation=False,
                scope='Engineering cervical repair. Does not demonstrate visual tracking or biological head mechanics.')
            obj.source_identity = {name: sha256(ROOT/'src'/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate(); obj._validate_pending(); obj._validate_afferent_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def present_scene(self, condition, *, output_connected=True):
        if condition not in CONDITIONS or type(output_connected) is not bool:
            raise ValueError('Requires a fixed elevation scene and boolean output connection')
        excluded = {'schema', 'config', 'source_identity', 'intervention', 'light_world',
                    'pending_light', 'output_connected', 'pending_cyborg_command'}
        before = _fingerprints(self, excluded)
        previous_world = copy.deepcopy(self.light_world.state_dict())
        previous_pending_light = _digest(self.pending_light)
        rotation, centers = self.eyes.pose()
        self.light_world = CardinalGratingWorld(self.time_ns, .5*(centers['L']+centers['R']),
                                               rotation, condition)
        self.pending_light = self.eyes.sample(self.light_world, self.time_ns)
        self.output_connected = output_connected
        self.pending_cyborg_command = self.cyborg_command()
        self.config.update(retinal_motion_condition=condition,
            retinal_motion_world_schema=CardinalGratingWorld.SCHEMA,
            retinal_motion_start_ns=self.time_ns)
        after = _fingerprints(self, excluded)
        checks = {k: v == after[k] for k, v in before.items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        self.intervention = dict(previous_intervention=copy.deepcopy(self.intervention),
            operation='Present a fixed physical elevation scene to the same bounded cervical CNS',
            time_ns=self.time_ns, previous_world=previous_world,
            previous_pending_light_sha256=previous_pending_light,
            output_connected=output_connected, preserved_state_checks=checks,
            neural_equations_changed=False, neural_parameters_changed=False,
            optical_apparatus_replaced=False, visual_scene_replaced=True,
            cut_scope='Zero neural command to the powered servo; return toward reference with inherited angle and velocity',
            biological_validation=False)
        self._validate(); self._validate_pending(); self._validate_afferent_pending()

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
                or type(self.rotor) is not CyborgBoundedServo):
            raise ValueError('Saved motion scene, live optical coupling or servo differs')
        world._validate()
        reference = CardinalGratingWorld(world.start_ns, world.center_mm, world.frame, world.condition)
        if _digest(reference.state_dict()) != _digest(world.state_dict()):
            raise ValueError('Motion parameters differ from the fixed cardinal protocol')

        m = self.hybrid.cvn7_manifest
        if (not isinstance(self.hybrid, (Cvn7ConductanceBrain, GpuCvn7ConductanceBrain))
                or type(m['enabled']) is not bool
                or self.config['cvn7_conductance_enabled'] != m['enabled']
                or self.config['cvn7_conductance_policy'] != m['policy']):
            raise ValueError('CvN7 component or policy differs')
        if (self.config['bounded_cervical_enabled'] is not True
                or self.config['cervical_device_schema'] != 'matrix_cyborg_bounded_servo_v1'
                or self.config['cervical_reference_angle_rad'] != self.rotor.reference_angle_rad
                or self.config['cervical_half_range_rad'] != self.rotor.half_range_rad):
            raise ValueError('Saved cervical device and configuration differ')
        self.rotor._validate()

    def state_dict(self):
        result = super().state_dict()
        result['schema'] = SCHEMA
        return result

    def _manifest_fields(self):
        result = super()._manifest_fields()
        result.update(schema=SCHEMA, prosthesis_kind='bounded_position_servo',
            bounded_cervical_enabled=True, optical_apparatus_replaced=True,
            cervical_device_schema='matrix_cyborg_bounded_servo_v1',
            cervical_reference_angle_rad=self.rotor.reference_angle_rad,
            cervical_half_range_rad=self.rotor.half_range_rad,
            cervical_range_status='Engineering +/-45 degrees; not a measured anatomical limit',
            motor_cut_scope='Zero neural command; powered return to the fixed cervical reference')
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after bounded cervical session creation')
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
            raise ValueError('Unsupported bounded cervical checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete bounded cervical checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete bounded cervical state')
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
            raise ValueError('Unknown bounded cervical neural component schema')
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
            obj.rotor = CyborgBoundedServo.from_state(state['rotor'])
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
                raise ValueError('Manifest and bounded cervical neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
