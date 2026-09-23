"""Save the complete canonical CNS/body with empirical KC/APL physiology."""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
import numpy as np
from kc_apl_dynamic_session import (
    KcAplDynamicSession,SOURCES as PARENT_SOURCES,_digest,_fingerprints,
    AnatomicalRateBrain,AnatomicalProprioception,CandidateGammaPlasticity,
    ContractileTibia,CyborgEye,FixedCyborgEye,GpuGradedDescendingBrain,
    GpuVisualBrain,OdorPatchWorld,RefinedContactBody,ROOT,SCALARS,
    CYBORG_SCALARS,guard_visual_session,sha256,read_state,write_state,
    CardinalGratingWorld,CyborgBoundedServo,KcGammaOutputBrain,GpuKcGammaOutputBrain)
from kc_electrical_scales_brain import GpuKcElectricalScalesBrain
from kc_apl_dynamic_brain import GpuKcAplDynamicBrain

SCHEMA='matrix_kc_electrical_scales_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('kc_electrical_scales.py','kc_electrical_scales_brain.py','kc_electrical_scales_session.py')


class KcElectricalScalesSession(KcAplDynamicSession):
    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        path=Path(path).resolve();parent=KcAplDynamicSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            old=parent.hybrid;old_saved=old.state_dict();old_saved.pop('schema');old_digest=_digest(old_saved)
            obj.hybrid=GpuKcElectricalScalesBrain.adopt(old,enabled=enabled);m=obj.hybrid.kc_electrical_scales_manifest
            obj.config=copy.deepcopy(obj.config)
            obj.config.update(candidate='KC_ELECTRICAL_SCALES_v1',kc_electrical_scales_enabled=enabled,
                electrical_backend_identity=obj.hybrid.backend_identity(),biological_validation=False,learning_demonstrated=False)
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            new=obj.hybrid.state_dict();new.pop('schema');new.pop('kc_electrical_scales_manifest');new.pop('kc_electrical_scales_state')
            checks['every_inherited_hybrid_field']=old_digest==_digest(new)
            if not all(checks.values()):raise AssertionError(checks)
            obj.intervention=dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Separate PN-gamma quantal conductance and APL-gamma graded nS interfaces in the continuing full CNS',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,new_canonical_neurons=0,new_anatomical_edges=0,
                new_PN_filter_states=2*len(m['source_rows']),APL_absolute_conductance_calibrated=False,
                parameter_fitting_to_CNS=False,biological_validation=False,learning_demonstrated=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            guard_visual_session(obj);obj._validate();obj._validate_pending();obj._validate_afferent_pending()
            return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuKcElectricalScalesBrain):raise ValueError('Missing projection conductance component')
        self.hybrid.validate_scales()
        if self.config['kc_electrical_scales_enabled']!=self.hybrid.kc_electrical_scales_manifest['enabled']:
            raise ValueError('Projection-scale configuration disagrees')

    def state_dict(self):
        result=super().state_dict();result['schema']=SCHEMA;return result

    def _manifest_fields(self):
        result=super()._manifest_fields();m=self.hybrid.kc_electrical_scales_manifest
        result.update(schema=SCHEMA,kc_electrical_scales_enabled=m['enabled'],
            PN_gamma_pairs=len(m['pn_positions']),APL_gamma_pairs=len(m['apl_positions']),
            new_PN_conductance_filter_states=2*len(m['source_rows']),coupling_step_ns=m['source']['coupling_step_ns'],
            APL_absolute_conductance_calibrated=False,PN_gamma_absolute_pair_strength_calibrated=False,
            PN_quantum_scope='Somatic-equivalent201Y+ miniature-current waveform with declared pairing and shape assumptions')
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after projection-scale session creation')
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
            raise ValueError('Unsupported projection-scale checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete projection-scale checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete projection-scale state')
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
        kinds = {kind.SCHEMA: kind for kind in (GpuKcElectricalScalesBrain,)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown projection-scale neural component schema')
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
                raise ValueError('Manifest and projection-scale neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
