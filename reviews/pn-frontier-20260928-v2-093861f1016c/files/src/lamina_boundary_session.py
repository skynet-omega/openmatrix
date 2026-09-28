"""Retinal boundary prosthesis on the same complete, continuing canonical CNS.

Adds optical sensor states, never anatomical neurons or edges. The original
retina, neuronal history, all afferents and physical apparatus remain intact.
"""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import numba
import numpy as np

from bounded_cervical_session import (
    BoundedCervicalSession, SOURCES as PARENT_SOURCES, _digest, _fingerprints,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CardinalGratingWorld, Cvn7ConductanceBrain, GpuCvn7ConductanceBrain,
    CyborgBoundedServo, CONDITIONS)
from lamina_boundary_brain import LaminaBoundaryBrain, GpuLaminaBoundaryBrain
from retinal_world import aperture_rays

SCHEMA = 'matrix_lamina_boundary_session_v1'
SOURCES = tuple(dict.fromkeys(tuple(PARENT_SOURCES)+(
    'lamina_boundary_brain.py', 'lamina_boundary_session.py')))
SELECTION = ROOT/'data/retinal_boundary_20260910/selection.npz'


def _sample(session, rays, sides):
    rotation, centers = session.eyes.pose()
    directions = rays @ rotation.T
    origins = np.array([centers[side] for side in sides])[:, None, :]
    light = session.light_world.luminance(origins, directions, session.time_ns)
    result = .5*light[:,0]+np.sum(light[:,1:],axis=1)/12.
    if not np.isfinite(result).all() or np.any((result<0.) | (result>1.)):
        raise ValueError('Boundary optical sample outside normalized instrument domain')
    return result


class LaminaBoundarySession(BoundedCervicalSession):
    def _build_boundary_optics(self):
        m = self.hybrid.boundary_manifest
        self._boundary_apertures = aperture_rays(m['sensor_rays'])
        self._boundary_sides = m['sensor_sides'].copy()

    def boundary_light(self):
        return _sample(self, self._boundary_apertures, self._boundary_sides)

    @classmethod
    def from_checkpoint(cls, path, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('Boundary enable flag must be boolean')
        path = Path(path).resolve()
        parent = BoundedCervicalSession.load(path)
        obj = cls(); obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema','config','source_identity','intervention','hybrid'}
            before = _fingerprints(parent, excluded)
            old = parent.hybrid; old_state = old.state.copy()
            old_saved = old.state_dict(); old_saved.pop('schema')
            old_digest = _digest(old_saved)
            with np.load(SELECTION,allow_pickle=False) as a:
                initial = _sample(parent, aperture_rays(a['sensor_rays']), a['sensor_sides'])
            kind = GpuLaminaBoundaryBrain if isinstance(old,GpuCvn7ConductanceBrain) else LaminaBoundaryBrain
            obj.hybrid = kind.adopt(old, initial_light=initial, enabled=enabled, selection_path=SELECTION)
            obj._build_boundary_optics()
            obj.config = copy.deepcopy(obj.config)
            obj.config.update(candidate='LAMINA_BOUNDARY_LIVE_v1',lamina_boundary_enabled=enabled,
                lamina_boundary_target_count=len(obj.hybrid.boundary_manifest['target_ids']),
                lamina_boundary_sensor_count=len(initial),prosthesis_kind='bounded_position_servo',
                biological_validation=False, learning_demonstrated=False)
            if hasattr(obj.hybrid,'backend_identity'):
                obj.config['electrical_backend_identity']=obj.hybrid.backend_identity()
            after = _fingerprints(obj, excluded)
            checks = {k:v==after[k] for k,v in before.items()}
            new_saved = obj.hybrid.state_dict()
            for k in ('schema','boundary_manifest','held_boundary_light'):new_saved.pop(k)
            new_saved['state']=new_saved['state'][:len(old_state)]
            checks['every_inherited_hybrid_field']=old_digest==_digest(new_saved)
            checks['all_inherited_neural_coordinates']=np.array_equal(old_state,obj.hybrid.state[:len(old_state)])
            if not all(checks.values()):raise AssertionError(checks)
            obj.intervention=dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Connect an explicit retinal boundary prosthesis only to canonical L1/L2 without an existing R1-R6 afferent and with a consistent mapped column',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,new_canonical_neurons=0,new_anatomical_edges=0,
                inherited_neural_state_variables=len(old_state),new_external_sensor_states=len(obj.hybrid.state)-len(old_state),
                incoming_neural_equations_changed=enabled,neural_parameter_fitting=False,weight_updates_enabled=False,
                visual_scene_replaced=False,optical_apparatus_replaced=False,biological_validation=False,
                boundary_manifest_sha256=_digest(obj.hybrid.boundary_manifest),
                scope='Coverage prosthesis; shared column sensors, explicit unknown older sensor history, no claimed recovered synapses or visual behavior.')
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            guard_visual_session(obj)
            obj._validate();obj._validate_pending();obj._validate_afferent_pending()
            return obj
        except BaseException:
            obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,(LaminaBoundaryBrain,GpuLaminaBoundaryBrain)):
            raise ValueError('Boundary session requires its saved sensor component')
        m=self.hybrid.boundary_manifest
        if (self.config['lamina_boundary_enabled'] != m['enabled']
                or self.config['lamina_boundary_target_count'] != len(m['target_ids'])
                or self.config['lamina_boundary_sensor_count'] != len(m['sensor_sides'])):
            raise ValueError('Boundary component and session configuration disagree')

    def _validate_pending(self):
        super()._validate_pending()
        if not np.array_equal(self.hybrid.held_boundary_light,self.boundary_light()):
            raise ValueError('Pending boundary image differs from physical scene and saved clock')

    def step(self):
        self._validate_pending()
        used=self.hybrid.held_boundary_light.copy()
        row=super().step()
        self.hybrid.set_boundary_light(self.boundary_light())
        row['lamina_boundary_light_used_mean']=float(used.mean())
        row['lamina_boundary_light_pending_mean']=float(self.hybrid.held_boundary_light.mean())
        row['lamina_boundary_enabled']=self.hybrid.boundary_manifest['enabled']
        self._validate_pending()
        return row

    def present_scene(self, condition, *, output_connected=True):
        if condition not in CONDITIONS or type(output_connected) is not bool:
            raise ValueError('Requires fixed elevation scene and boolean output connection')
        excluded={'schema','config','source_identity','intervention','light_world','pending_light',
                  'output_connected','pending_cyborg_command','hybrid'}
        before=_fingerprints(self,excluded)
        old_hybrid=self.hybrid.state_dict();old_hybrid.pop('held_boundary_light')
        old_digest=_digest(old_hybrid)
        previous_world=copy.deepcopy(self.light_world.state_dict())
        rotation,centers=self.eyes.pose()
        self.light_world=CardinalGratingWorld(self.time_ns,.5*(centers['L']+centers['R']),rotation,condition)
        self.pending_light=self.eyes.sample(self.light_world,self.time_ns)
        self.hybrid.set_boundary_light(self.boundary_light())
        self.output_connected=output_connected;self.pending_cyborg_command=self.cyborg_command()
        self.config.update(retinal_motion_condition=condition,retinal_motion_world_schema=CardinalGratingWorld.SCHEMA,
                           retinal_motion_start_ns=self.time_ns)
        after=_fingerprints(self,excluded);checks={k:v==after[k] for k,v in before.items()}
        new_hybrid=self.hybrid.state_dict();new_hybrid.pop('held_boundary_light')
        checks['all_hybrid_state_except_pending_boundary_light']=old_digest==_digest(new_hybrid)
        if not all(checks.values()):raise AssertionError(checks)
        self.intervention=dict(previous_intervention=copy.deepcopy(self.intervention),
            operation='Present physical scene to the same complete CNS and shared-column retinal boundary sensors',
            time_ns=self.time_ns,previous_world=previous_world,output_connected=output_connected,
            preserved_state_checks=checks,neural_equations_changed=False,neural_parameters_changed=False,
            optical_apparatus_replaced=False,visual_scene_replaced=True,biological_validation=False)
        self._validate();self._validate_pending();self._validate_afferent_pending()

    def state_dict(self):
        result=super().state_dict();result['schema']=SCHEMA;return result

    def _manifest_fields(self):
        result=super()._manifest_fields();m=self.hybrid.boundary_manifest
        result.update(schema=SCHEMA,lamina_boundary_enabled=m['enabled'],
            lamina_boundary_targets=len(m['target_ids']),lamina_boundary_sensors=len(m['sensor_sides']),
            lamina_boundary_external_states=4*len(m['sensor_sides']),
            inherited_neural_state_variables=m['parent_state_size'],
            total_hybrid_state_variables=len(self.hybrid.state),
            lamina_boundary_new_canonical_neurons=0,lamina_boundary_new_anatomical_edges=0,
            retinal_boundary_status='Explicit coverage prosthesis; not measured missing synapses')
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after lamina boundary session creation')
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
            raise ValueError('Unsupported lamina boundary checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete lamina boundary checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete lamina boundary state')
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
        kinds = {kind.SCHEMA: kind for kind in (LaminaBoundaryBrain, GpuLaminaBoundaryBrain)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown lamina boundary neural component schema')
        obj.hybrid = kind.from_state(obj.brain, state['hybrid'])
        obj._build_boundary_optics()
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
                raise ValueError('Manifest and lamina boundary neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
