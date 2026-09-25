"""Outer contact runtime with explicit persistent antennal-world ownership.

No standalone carrier load, generic odor-world reconstruction, body replay,
force-law change, or neural decoder change is performed by this adapter.
"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import shutil
import numpy as np
import numba

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
CONTACT=ROOT/'work/stage2_contact_cns_20260915/contact_runtime.py'
sys.path.insert(0,str(CONTACT.parent))
import contact_runtime as contact
if Path(contact.__file__).resolve()!=CONTACT:raise ValueError('Conflicting contact runtime import')
from antennal_world import AntennalWorld,BOUNDARY,digest
from coefficient_buffer_session import CoefficientBufferSession,SOURCES,ENTRYPOINT
from kc_audited_session import require_covered_dependencies
from session_io import read_state,write_state
import rh_tarsal_storage as storage

class AntennalCarrier(CoefficientBufferSession):
    SCHEMA='CNS223_carrier_requires_antennal_contact_runtime_v1'

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA
        if type(self.world) is not AntennalWorld:raise ValueError('Missing antennal world authority')
        out['world']=self.world.state_dict();return out

    def _manifest_fields(self):
        out=super()._manifest_fields()
        out.update(schema=self.SCHEMA,outer_runtime_required=True,antennal_world_schema=AntennalWorld.SCHEMA)
        return out

    def save(self,path):raise ValueError('Carrier requires AntennalContactRuntime.save')

    @classmethod
    def load(cls,path):raise ValueError('Carrier requires AntennalContactRuntime.load')

def _load_carrier(path):
    """Version of RH223 restoration retaining every check, with a typed world.

    Original source: src/rh_tarsal_storage.py. The only restoration differences
    are the carrier class/schema, world constructor, and binding after body load.
    """
    require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('schema')!=AntennalCarrier.SCHEMA:raise ValueError('Wrong antennal carrier schema')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Incomplete session archive')
    for name,h in manifest['files'].items():
        if digest(path/name)!=h:raise ValueError('Checkpoint integrity failed: '+name)
    state=read_state(path/'session')
    special={'schema','world','body','hybrid','eyes','light_world','muscles','plasticity','proprioception','used_light','rotor','probe'}
    if set(state)!=storage.SCALARS|storage.CYBORG_SCALARS|special or state['schema']!=AntennalCarrier.SCHEMA:
        raise ValueError('Incomplete antennal carrier state')
    if state['source_identity']!={name:digest(ROOT/'src'/name) for name in SOURCES}:
        raise ValueError('Checkpoint requires its archived source versions')
    world=AntennalWorld.from_state(state['world'])  # schema validation before allocating the CNS
    obj=AntennalCarrier();obj.brain=storage.AnatomicalRateBrain.load_checkpoint(path/'brain')
    for key in storage.SCALARS|storage.CYBORG_SCALARS:setattr(obj,key,state[key])
    if obj.config['numba_version']!=numba.__version__:raise ValueError('Numerical runtime changed')
    if obj.config['backend_identity']!=storage.GpuVisualBrain.backend_identity():raise ValueError('GPU runtime changed')
    obj._index_ports();obj._index_motors();obj.world=world
    obj.plasticity=storage.CandidateGammaPlasticity.from_state(obj.brain,state['plasticity'])
    if state['hybrid']['schema']!=obj.BRAIN.SCHEMA:raise ValueError('Wrong neural backend')
    obj.hybrid=obj.BRAIN.from_state(obj.brain,state['hybrid']);obj._build_boundary_optics()
    if obj.config['electrical_backend_identity']!=obj.hybrid.backend_identity():raise ValueError('Electrical runtime changed')
    obj.hybrid.sync_plastic_weights(obj.plasticity);obj.probe=copy.deepcopy(state['probe']);obj._apply_probe_mask();storage.guard_visual_session(obj)
    obj.body=storage.RHTarsalBody.from_state(state['body']);world.bind(obj.body)
    obj._index_coxa();obj._index_tr();obj._index_rf();obj._index_tarsal();obj._index_serial();obj._index_rh()
    try:
        obj.proprioception=storage.FlyBodyProprioception.from_state(obj.brain,obj.body,state['proprioception'])
        obj.rotor=storage.CyborgBoundedServo.from_state(state['rotor'])
        obj.eyes=storage.FlyBodyEye.from_state(obj.brain,obj.body,state['eyes'],obj.rotor)
        obj.light_world=storage.CardinalGratingWorld.from_state(state['light_world'])
        obj.muscles=storage.CNSFlyBodyMuscles.from_state(obj.body,state['muscles']);obj.body.bind_muscles(obj.muscles)
        obj.used_light=list(state['used_light']);obj.failed=False
        obj._validate();obj._validate_pending();obj._validate_afferent_pending();obj._validate_coxal_pending()
        obj._validate_tr_pending();obj._validate_rf_pending();obj._validate_tarsal_pending();obj._validate_serial_pending()
        obj._validate_cxhp8_pending();obj._validate_rh_pending()
        if any(manifest.get(k)!=v for k,v in obj._manifest_fields().items()):raise ValueError('Manifest and state disagree')
        return obj
    except BaseException:obj.body.close();raise

class AntennalContactRuntime(contact.ContactRuntime):
    SCHEMA='CNS223_antennal_synthetic_contact_prosthesis_v1'
    RESTORE_FIELDS=('origin_ns','active','withdrawals','command_mode','requested','last_force','last_ignored_tibia',
                    'last_work_J','synthetic_steps','dn_baseline','last_dn','parent','parent_manifest')

    @staticmethod
    def _adapter_sources():
        return {str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/'antennal_world.py',BOUNDARY,ROOT/'src/rh_tarsal_storage.py')}

    @classmethod
    def adopt(cls,runtime):
        if type(runtime) is not contact.ContactRuntime or type(runtime.core) is not CoefficientBufferSession:
            raise ValueError('Adopt exactly one loaded ContactRuntime with its continuing CNS223')
        core=runtime.core;body=runtime.body
        if core.failed or body is not core.body or core.time_ns!=core.world.time_ns or body.steps*round(body.dt*1e9)!=core.time_ns:
            raise ValueError('Cannot install antennae on failed or separately advanced body/CNS')
        before=body.integration_state().copy();pending=core.pending_sensors.copy()
        world=AntennalWorld.adopt(core.world,body,pending,core.CONTROL_NS)
        np.testing.assert_array_equal(world.sense(body.observe()),pending)
        runtime.__class__=cls;core.__class__=AntennalCarrier;core.world=world
        runtime.adapter_source_identity=cls._adapter_sources()
        runtime.body.advance=runtime.body_advance
        np.testing.assert_array_equal(body.integration_state(),before)
        return runtime

    @classmethod
    def from_contact(cls,path):return cls.adopt(contact.ContactRuntime.load(path))

    @classmethod
    def from_core(cls,path):
        raise ValueError('Load/adopt a ContactRuntime explicitly; do not imply recovered body continuity')

    def _validate_adapter(self):
        if type(self.core) is not AntennalCarrier or type(self.core.world) is not AntennalWorld:
            raise ValueError('Wrong antennal CNS/world owner')
        if self.core.world.body is not self.body or self.core.time_ns!=self.core.world.time_ns:
            raise ValueError('World is not bound to this continuing body')
        if self.adapter_source_identity!=self._adapter_sources():raise ValueError('Changed antenna adapter source')
        np.testing.assert_array_equal(self.core.pending_sensors,self.core.world.sense(self.body.observe()))
        if self.withdrawals!=int(self.active):raise ValueError('Force ownership mismatch')
        if (self.controller is None)!= (not self.active):raise ValueError('Missing or unexpected force controller')

    def step(self):
        self._validate_adapter();used=self.core.pending_sensors.copy();oldtime=self.core.time_ns
        row=super().step();self._validate_adapter()
        row['antennal_boundary']=dict(world_schema=AntennalWorld.SCHEMA,installed_ns=self.core.world.installed_ns,
            used_sampling_time_ns=oldtime,pending_sampling_time_ns=self.core.time_ns,
            sensors_used=used.tolist(),sensors_pending=self.core.pending_sensors.tolist(),
            committed_prior_interval_used=oldtime==self.core.world.installed_ns,
            antennae_mm=self.core.world.sample_geometry()['antennae_mm'].tolist(),
            contact_reinforcement_pending=0.,orientation_demonstrated=False)
        return row

    def state(self):
        self._validate_adapter();state=super().state();state['schema']=self.SCHEMA
        state['adapter_source_identity']=copy.deepcopy(self.adapter_source_identity)
        state['world_authority']=dict(schema=AntennalWorld.SCHEMA,owner='core_carrier/session.world',
            installed_ns=self.core.world.installed_ns,boundary_replacements=1)
        state['sources'].update(self.adapter_source_identity)
        return state

    def save(self,path):
        self._validate_adapter();path=Path(path).resolve()
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.'+path.name+'-',dir=path.parent))
        try:
            CoefficientBufferSession.save(self.core,staging/'core_carrier')
            write_state(staging/'prosthesis',self.state())
            manifest=dict(schema=self.SCHEMA,entrypoint=str(Path(__file__).resolve()),
                core_carrier_schema=AntennalCarrier.SCHEMA,world_schema=AntennalWorld.SCHEMA,
                neuron_count=self.core.brain.n_neurons,core_runtime_sources=len(self.core.source_identity),
                time_ns=self.core.time_ns,synthetic_function=True,orientation_demonstrated=False,
                files={str(p.relative_to(staging)):digest(p) for p in (staging/'core_carrier/manifest.json',staging/'prosthesis.json',staging/'prosthesis.npz')})
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');staging.rename(path)
        except BaseException:shutil.rmtree(staging);raise
        return path

    @classmethod
    def load(cls,path):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('schema')!=cls.SCHEMA or manifest.get('core_carrier_schema')!=AntennalCarrier.SCHEMA or manifest.get('world_schema')!=AntennalWorld.SCHEMA:
            raise ValueError('Wrong antennal outer archive schema')
        for name,h in manifest['files'].items():
            if digest(path/name)!=h:raise ValueError('Changed archive '+name)
        s=read_state(path/'prosthesis')
        if s['schema']!=cls.SCHEMA:raise ValueError('Wrong antennal sidecar schema')
        for name,h in s['sources'].items():
            if digest(ROOT/name)!=h:raise ValueError('Changed strategy '+name)
        core=_load_carrier(path/'core_carrier')
        try:
            obj=cls(core)
            for name in cls.RESTORE_FIELDS:setattr(obj,name,s[name])
            obj.adapter_source_identity=s['adapter_source_identity']
            if s['controller'] is not None:obj.controller=obj._module().Controller.from_state(obj.body.model,obj.body.data,s['controller'])
            obj._validate_adapter()
            actual=obj.state()
            if set(actual)!=set(s) or actual['sources']!=s['sources']:raise ValueError('Incomplete outer state or sources')
            if actual['world_authority']!=s['world_authority']:raise ValueError('World authority mismatch')
            if (core.time_ns!=manifest['time_ns'] or core.brain.n_neurons!=manifest['neuron_count']
                    or len(core.source_identity)!=manifest['core_runtime_sources']):raise ValueError('Outer manifest and CNS disagree')
            return obj
        except BaseException:core.close();raise
