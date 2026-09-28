"""Persist the same whole CNS/body with complete mapped gamma output routing."""
from pathlib import Path
import copy, json, shutil, tempfile
import numba
import numpy as np
from pnkc_receptor_session import (
    PnkcReceptorSession, SOURCES as PARENT_SOURCES, _digest, _fingerprints,
    AnatomicalRateBrain, AnatomicalProprioception, CandidateGammaPlasticity,
    ContractileTibia, CyborgEye, FixedCyborgEye, GpuGradedDescendingBrain,
    GpuVisualBrain, OdorPatchWorld, RefinedContactBody, ROOT, SCALARS,
    CYBORG_SCALARS, guard_visual_session, sha256, read_state, write_state,
    CardinalGratingWorld, CyborgBoundedServo, PnkcReceptorBrain, GpuPnkcReceptorBrain)
from kcgamma_output_brain import KcGammaOutputBrain, GpuKcGammaOutputBrain, PARENT_STATE_SIZE

SCHEMA='matrix_kcgamma_output_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('kcgamma_output_brain.py','kcgamma_output_session.py')


class KcGammaOutputSession(PnkcReceptorSession):
    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        path=Path(path).resolve();parent=PnkcReceptorSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            old=parent.hybrid;old_saved=old.state_dict();old_saved.pop('schema');old_digest=_digest(old_saved)
            kind=GpuKcGammaOutputBrain if isinstance(old,GpuPnkcReceptorBrain) else KcGammaOutputBrain
            obj.hybrid=kind.adopt(old,enabled=enabled);m=obj.hybrid.kcgamma_output_manifest
            obj.config=copy.deepcopy(obj.config)
            obj.config.update(candidate='KCGAMMA_OUTPUT_LIVE_v1',kcgamma_output_enabled=enabled,
                kcgamma_output_targets=len(m['target_rows']),biological_validation=False,learning_demonstrated=False)
            if hasattr(obj.hybrid,'backend_identity'):obj.config['electrical_backend_identity']=obj.hybrid.backend_identity()
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            new=obj.hybrid.state_dict();new.pop('schema');new.pop('kcgamma_output_manifest')
            checks['every_inherited_hybrid_field']=old_digest==_digest(new)
            checks['all_inherited_neural_coordinates']=np.array_equal(old.state,obj.hybrid.state)
            if not all(checks.values()):raise AssertionError(checks)
            obj.intervention=dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Route persisted KCgamma axonal release along additional known gamma-lobe output fractions in the continuing whole CNS',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,new_canonical_neurons=0,new_anatomical_edges=0,
                inherited_neural_state_variables=PARENT_STATE_SIZE,new_dynamic_states=0,new_parameters=0,
                neural_parameter_fitting=False,weight_updates_enabled=False,biological_validation=False,learning_demonstrated=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            guard_visual_session(obj);obj._validate();obj._validate_pending();obj._validate_afferent_pending()
            return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,(KcGammaOutputBrain,GpuKcGammaOutputBrain)):
            raise ValueError('Gamma output session requires its persistent mapped component')
        m=self.hybrid.kcgamma_output_manifest
        if self.config['kcgamma_output_enabled']!=m['enabled'] or self.config['kcgamma_output_targets']!=len(m['target_rows']):
            raise ValueError('Gamma output component and session configuration disagree')

    def state_dict(self):
        result=super().state_dict();result['schema']=SCHEMA;return result

    def _manifest_fields(self):
        result=super()._manifest_fields();m=self.hybrid.kcgamma_output_manifest
        result.update(schema=SCHEMA,kcgamma_output_enabled=m['enabled'],
            kcgamma_output_sources=len(m['source_rows']),kcgamma_output_targets=len(m['target_rows']),
            kcgamma_output_pairs=len(m['csr_positions']),kcgamma_output_new_states=0,
            kcgamma_output_status='Regional routing completed on mapped eligible outputs; inherited gamma kinetics and APL physiology remain hypotheses',
            total_hybrid_state_variables=len(self.hybrid.state))
        return result

    def save(self, path):
        self._validate_afferent_pending()
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Runtime source changed after gamma output session creation')
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
            raise ValueError('Unsupported gamma output checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete gamma output checkpoint')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Checkpoint integrity failed: {name}')
        state = read_state(path/'session')
        special = {'schema', 'world', 'body', 'hybrid', 'eyes', 'light_world', 'muscles',
                   'plasticity', 'proprioception', 'used_light', 'rotor', 'probe'}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state['schema'] != SCHEMA:
            raise ValueError('Unsupported or incomplete gamma output state')
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
        kinds = {kind.SCHEMA: kind for kind in (KcGammaOutputBrain, GpuKcGammaOutputBrain)}
        kind = kinds.get(state['hybrid']['schema'])
        if kind is None:
            raise ValueError('Unknown gamma output neural component schema')
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
                raise ValueError('Manifest and gamma output neural/body state disagree')
            return obj
        except BaseException:
            obj.body.close()
            raise
