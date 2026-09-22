"""Strict shared storage for new spatial-KC session extensions.

Historical loaders retain their archived source identities. New extensions use
one serializer/restorer rather than copying a complete loader for each piece.
"""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
from kc_spatial_session import (ROOT,SCALARS,CYBORG_SCALARS,sha256,read_state,write_state,
    AnatomicalRateBrain,OdorPatchWorld,CandidateGammaPlasticity,GpuVisualBrain,
    RefinedContactBody,AnatomicalProprioception,CyborgBoundedServo,CyborgEye,
    FixedCyborgEye,CardinalGratingWorld,ContractileTibia,guard_visual_session)


def save_session(obj,path,sources):
    obj._validate_afferent_pending();obj._validate();obj._validate_pending();obj.light_world._validate()
    if obj.source_identity!={name:sha256(ROOT/'src'/name) for name in sources}:
        raise ValueError('Source changed after session creation')
    path=Path(path).resolve()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+path.name+'-',dir=path.parent))
    try:
        obj.brain.save_checkpoint(staging/'brain');write_state(staging/'session',obj.state_dict())
        (staging/'source').mkdir()
        for name in sources:shutil.copyfile(ROOT/'src'/name,staging/'source'/name)
        manifest=obj._manifest_fields()
        manifest['files']={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        staging.rename(path);return path
    except BaseException:shutil.rmtree(staging);raise


def load_session(path,session_class,schema,sources,brain_class):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('schema')!=schema:raise ValueError('Wrong session schema')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Incomplete session archive')
    for name,h in manifest['files'].items():
        if sha256(path/name)!=h:raise ValueError('Checkpoint integrity failed: '+name)
    state=read_state(path/'session')
    special={'schema','world','body','hybrid','eyes','light_world','muscles','plasticity','proprioception','used_light','rotor','probe'}
    if set(state)!=SCALARS|CYBORG_SCALARS|special or state['schema']!=schema:raise ValueError('Incomplete session state')
    if state['source_identity']!={name:sha256(ROOT/'src'/name) for name in sources}:
        raise ValueError('Checkpoint requires its archived source versions')
    obj=session_class();obj.brain=AnatomicalRateBrain.load_checkpoint(path/'brain')
    for key in SCALARS|CYBORG_SCALARS:setattr(obj,key,state[key])
    if obj.config['numba_version']!=numba.__version__:raise ValueError('Numerical runtime changed')
    if obj.config['backend_identity']!=GpuVisualBrain.backend_identity():raise ValueError('GPU runtime changed')
    obj._index_ports();obj._index_motors();obj.world=OdorPatchWorld()
    if set(obj.world.__dict__)!=set(state['world']):raise ValueError('Incomplete world state')
    obj.world.__dict__.update(state['world'])
    obj.plasticity=CandidateGammaPlasticity.from_state(obj.brain,state['plasticity'])
    if state['hybrid']['schema']!=brain_class.SCHEMA:raise ValueError('Wrong neural backend')
    obj.hybrid=brain_class.from_state(obj.brain,state['hybrid']);obj._build_boundary_optics()
    if obj.config['electrical_backend_identity']!=obj.hybrid.backend_identity():raise ValueError('Electrical runtime changed')
    obj.hybrid.sync_plastic_weights(obj.plasticity);obj.probe=copy.deepcopy(state['probe']);obj._apply_probe_mask();guard_visual_session(obj)
    obj.body=RefinedContactBody.from_state(state['body'])
    try:
        obj.proprioception=AnatomicalProprioception.from_state(obj.brain,obj.body,state['proprioception'])
        obj.rotor=CyborgBoundedServo.from_state(state['rotor'])
        obj.eyes=CyborgEye.from_state(obj.brain,obj.body,state['eyes'],obj.rotor)
        if obj.config['looming_camera_fixed']:obj.eyes=FixedCyborgEye.adopt_fixed(obj.eyes,obj.rotor,obj.initial_camera)
        obj.light_world=CardinalGratingWorld.from_state(state['light_world'])
        obj.muscles=ContractileTibia.from_state(obj.body,state['muscles']);obj.used_light=list(state['used_light']);obj.failed=False
        obj._validate();obj._validate_pending();obj._validate_afferent_pending()
        if any(manifest.get(k)!=v for k,v in obj._manifest_fields().items()):raise ValueError('Manifest and state disagree')
        return obj
    except BaseException:obj.body.close();raise
