"""Cold reconstruction of this conserved organism adapter, not a generic brain loader.

Uses the existing carrier's constructors/validators. The original loader supplies
the static brain and decoder bindings; all evolving scientific state is replaced
by the saved checkpoint before any neural or physical step is allowed.
"""
from pathlib import Path
import copy,gc,hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def require(ok,message):
    if not ok:raise ValueError(message)

def restore(obj,folder):
    from session_io import read_state
    from run_storage import verify_snapshot
    from operator_state import OperatorState,LEGACY_BINDINGS
    import antennal_runtime as ar
    from antennal_world import AntennalWorld
    storage=ar.storage
    folder=Path(folder);manifest=verify_snapshot(folder)
    require('effective_operator.npz' in manifest['files'],'Checkpoint lacks effective operator; do not reconstruct it from guessed parameters')
    saved=read_state(folder/'session');outer=read_state(folder/'prosthesis');published=read_state(folder/'published')
    operator=OperatorState.from_state(read_state(folder/'effective_operator'),expected_bindings=LEGACY_BINDINGS)
    boundary=json.loads((folder/'boundary.json').read_text(encoding='utf-8'))
    special={'schema','world','body','hybrid','eyes','light_world','muscles','plasticity','proprioception','used_light','rotor','probe'}
    require(set(saved)==storage.SCALARS|storage.CYBORG_SCALARS|special,'Incomplete carrier state')
    require(saved['schema']==ar.AntennalCarrier.SCHEMA,'Wrong carrier schema')
    require(outer['schema']==obj.SCHEMA,'Wrong outer schema')
    require(saved['source_identity']==obj.core.source_identity,'Source versions differ')
    require(set(outer)==set(obj.state()),'Outer state schema differs')
    require(outer['sources']==obj.state()['sources'],'Outer source versions differ')
    # The static graph and named reader bindings are supplied by the same frozen
    # original loader, not inferred from the observation we aim to reproduce.
    old=obj.core;brain=old.brain
    require(published['rates'].shape==brain.rates.shape,'Published brain layout differs')
    old.close();obj.core=None;del old;gc.collect()
    core=ar.AntennalCarrier();core.brain=brain
    try:
        for key in storage.SCALARS|storage.CYBORG_SCALARS:setattr(core,key,copy.deepcopy(saved[key]))
        require(core.config['numba_version']==ar.numba.__version__,'Numerical runtime changed')
        require(core.config['backend_identity']==storage.GpuVisualBrain.backend_identity(),'GPU runtime changed')
        brain.rates=np.array(published['rates'],copy=True);brain.time_ns=int(published['time_ns'])
        brain.rng.bit_generator.state=copy.deepcopy(published['rng'])
        core._index_ports();core._index_motors();core.world=AntennalWorld.from_state(saved['world'])
        core.plasticity=storage.CandidateGammaPlasticity.from_state(brain,saved['plasticity'])
        require(saved['hybrid']['schema']==core.BRAIN.SCHEMA,'Wrong neural backend')
        core.hybrid=core.BRAIN.from_state(brain,saved['hybrid']);core._build_boundary_optics()
        require(core.config['electrical_backend_identity']==core.hybrid.backend_identity(),'Electrical runtime changed')
        core.hybrid.sync_plastic_weights(core.plasticity);core.probe=copy.deepcopy(saved['probe']);core._apply_probe_mask();storage.guard_visual_session(core)
        operator.restore(core.hybrid)
        require(not operator.differences(core.hybrid),'Effective operator changed after restoration')
        core.body=storage.RHTarsalBody.from_state(saved['body']);core.world.bind(core.body)
        for name in ('_index_coxa','_index_tr','_index_rf','_index_tarsal','_index_serial','_index_rh'):getattr(core,name)()
        core.proprioception=storage.FlyBodyProprioception.from_state(brain,core.body,saved['proprioception'])
        core.rotor=storage.CyborgBoundedServo.from_state(saved['rotor'])
        core.eyes=storage.FlyBodyEye.from_state(brain,core.body,saved['eyes'],core.rotor)
        core.light_world=storage.CardinalGratingWorld.from_state(saved['light_world'])
        core.muscles=storage.CNSFlyBodyMuscles.from_state(core.body,saved['muscles']);core.body.bind_muscles(core.muscles)
        core.used_light=list(saved['used_light']);core.failed=False
        # Preserve the world-fixed Gaussian and its original installation clock.
        # The carrier serializer persists the original world, so the boundary
        # wrapper must be reconstructed from its separately frozen metadata.
        required={'schema','arm','installed_ns','source_mm','sigma_mm','geometry_sha256',
                  'sampling_order','source_authority','canonical_scale_declared',
                  'biological_transduction_calibrated'}
        require(set(boundary)==required,'Unknown boundary schema')
        require(boundary['schema']=='fixed_gaussian_intensity_v1' and
                boundary['arm']=='minus' and boundary['sampling_order']==['L','R'],
                'Wrong Gaussian arm or side order')
        from gaussiano_400 import FixedGaussian
        field=FixedGaussian.__new__(FixedGaussian)
        field.base=core.world.boundary;field.world=core.world
        field.arm=boundary['arm'];field.origin_ns=int(boundary['installed_ns'])
        field.spec={'source_mm':boundary['source_mm'],'sigma_mm':boundary['sigma_mm'],
                    'geometry_sha256':boundary['geometry_sha256']}
        field.source=np.asarray(boundary['source_mm'],dtype=float).copy()
        field.source.flags.writeable=False
        field.sigma=float(boundary['sigma_mm'])
        require(field.source.shape==(2,) and np.isfinite(field.source).all() and
                np.isfinite(field.sigma) and field.sigma>0,
                'Invalid fixed Gaussian geometry')
        require(core.time_ns==field.origin_ns+1_000_000_000 and
                core.world.next_event==len(core.world.events),
                'Wrong continuation clock or pending historical event')
        require(field.metadata()==boundary,'Boundary metadata changed')
        core.world.boundary=field
        # AntennalContactRuntime caches the body separately from its carrier.
        # Rebind that alias before constructing the controller or validating
        # the resumed world.  The prior attempt left obj.body pointing at the
        # closed pre-restore body while the world owned core.body.
        obj.core=core;obj.body=core.body
        for name in obj.RESTORE_FIELDS:setattr(obj,name,copy.deepcopy(outer[name]))
        obj.adapter_source_identity=copy.deepcopy(outer['adapter_source_identity'])
        obj.controller=obj._module().Controller.from_state(core.body.model,core.body.data,outer['controller'])
        obj.native_advance=core.body.advance
        core.body.advance=obj.body_advance
        # Constructors may publish initial rates; the saved publication is the
        # committed boundary and is restored verbatim after construction.
        brain.rates=np.array(published['rates'],copy=True);brain.time_ns=int(published['time_ns'])
        brain.rng.bit_generator.state=copy.deepcopy(published['rng'])
        for name in ('_validate','_validate_pending','_validate_afferent_pending','_validate_coxal_pending','_validate_tr_pending',
                     '_validate_rf_pending','_validate_tarsal_pending','_validate_serial_pending','_validate_cxhp8_pending','_validate_rh_pending'):
            getattr(core,name)()
        obj._validate_adapter()
        require(obj.state()['world_authority']==outer['world_authority'],'World ownership differs')
        require(core.time_ns==core.hybrid.time_ns==brain.time_ns==core.world.time_ns,'Restored clocks differ')
        return {'checkpoint_manifest_sha256':hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest(),
                'operator_identity_sha256':operator.state_dict()['identity_sha256'],
                'field_metadata_exact':True,'neural_or_physical_steps_during_restore':0,
                'scope':'Constructed and validated only; exact initial state and actual continuation must be checked separately'}
    except BaseException:
        # A partially reconstructed and then closed owner cannot publish a
        # checkpoint. This guard affects only the failure path.
        core.failed=True
        obj.core=core
        if getattr(core,'body',None) is not None:core.body.close()
        if getattr(core,'hybrid',None) is not None and hasattr(core.hybrid,'close'):core.hybrid.close()
        raise
