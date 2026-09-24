"""Bounded full-organism mirrored finite-source contrast gate; not navigation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EVENTS = ROOT/'campanas/etapa3_pn629_intervention_20260923_15'
GAUSSIAN = ROOT/'motor_nuevo/gaussiano400_pose_guard_v2_20260923'
DRIVE_OBSERVER = ROOT/'motor_nuevo/adapter_drive_observer_20260923'
sys.path[:0] = [str(OLD/'work/motor14_20260922'), str(OLD/'work/motor13_20260922'),
                str(ROOT/'motor_nuevo/pipeline_review_20260922'),str(EVENTS),
                str(GAUSSIAN),str(DRIVE_OBSERVER)]
from motor_runtime import load
from run_pipeline import RunStorage, atomic_json
from native_tap import NativeFlowTap
from gaussiano_400 import geometry, install as install_gaussian
from observer import DriveObserver

CAPTURE_EPOCHS = ('preparacion:1','preparacion:40','ensayo:1','ensayo:2',
                  'ensayo:10','ensayo:100','ensayo:300','ensayo:400')

def source_files() -> tuple[Path,...]:
    return (HERE/'PLAN.json',HERE/'run_mirror.py',HERE/'verify_mirror.py',HERE/'test_verify_mirror.py',
            HERE/'CAMPOS.json',HERE/'reference/ORN_INDEX_MAP.json',
            ROOT/'campanas/etapa4_next_20260924_25/geometry_preflight.py',
            ROOT/'campanas/etapa4_next_20260924_25/geometry_preflight_01/RESULT.json',
            *(ROOT/'campanas/etapa3_continuation_repair_20260923_19'/name for name in
              ('verify_repair19.py','REPAIR19_CONTRACT.json','RUN_LOCK.json')),
            *(ROOT/'campanas/etapa3_postclose_20260923_20'/name for name in
              ('verify_postclose.py','PLAN.json')),
            ROOT/'campanas/etapa4_diseno_20260923_17/GEOMETRY.json',
            GAUSSIAN/'gaussiano_400.py',
            DRIVE_OBSERVER/'observer.py',DRIVE_OBSERVER/'PLAN.json',
            *(EVENTS/name for name in ('native_tap.py','event_waveform.py','event_ports.py',
              'event_coupling.py','native_cell.py','device_cell.py','physical_events.cu',
              'device_cell.cu','block_runtime.hpp')),
            ROOT/'motor_nuevo/pipeline_review_20260922/runtime_session.py',
            ROOT/'motor_nuevo/pipeline_review_20260922/run_pipeline.py',
            *(ROOT/'campanas/etapa3_motor_nuevo_20260922'/name for name in
              ('organism_adapter.py','graph_core.py','pn_execution.py')),
            ROOT/'motor_nuevo/causal_runtime_20260922/operator_state.py',
            ROOT/'motor_nuevo/native_hybrid_20260922/libgraph_control_v2.so',
            *(ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources'/name for name in
              ('gpu_visual_brain.py','prosthetic_olfactory_brain.py','gpu_coefficient_layout.py')),
            OLD/'work/motor13_20260922/block_midpoint.py',
            OLD/'work/motor13_20260922/kc_adaptive.py',
            OLD/'work/motor14_20260922/motor_runtime.py',
            OLD/'work/motor14_20260922/event_coupling.py',
            OLD/'work/stage4_antennal_contact_adapter_20260915/antennal_runtime.py',
            OLD/'work/stage3_bilateral_entry_20260915/antennal_boundary.py',
            *(OLD/'src'/name for name in ('gpu_coefficient_buffers.py',
              'matrix_olfactory_diagnostic.py','sensorimotor_contact_session.py')))

def require_sources() -> dict:
    lock_path=HERE/'SOURCE_LOCK.json'
    frozen=json.loads(lock_path.read_text())
    expected={str(p.resolve()) for p in source_files()}
    if set(frozen)!=expected or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest
                                    for p,digest in frozen.items()):
        raise RuntimeError('Frozen Stage4 source identity changed before organism run')
    plan=json.loads((HERE/'PLAN.json').read_text())
    if (plan['fixed_inputs']['local_pose_guarded_helper_sha256']!=frozen[str((GAUSSIAN/'gaussiano_400.py').resolve())]
            or plan['fixed_inputs']['geometry_sha256']!=frozen[str((ROOT/'campanas/etapa4_diseno_20260923_17/GEOMETRY.json').resolve())]
            or plan['fixed_inputs']['fields_sha256']!=frozen[str((HERE/'CAMPOS.json').resolve())]
            or plan['fixed_inputs']['geometry_preflight_sha256']!=frozen[str((ROOT/'campanas/etapa4_next_20260924_25/geometry_preflight_01/RESULT.json').resolve())]
            or tuple(plan['observation']['bounded_drive_epochs'])!=CAPTURE_EPOCHS):
        raise RuntimeError('Stage4 mirrored plan, geometry, observer and source lock disagree')
    return {'source_lock_sha256':hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            'sources_count':len(frozen)}

def require_stage3_closed():
    repair=ROOT/'campanas/etapa3_continuation_repair_20260923_19'
    queue_path=repair/'QUEUE.json'
    postclose=ROOT/'campanas/etapa3_postclose_20260923_20'
    receipt_path=postclose/'POSTCLOSE_VERIFIED_01.json'
    final_path=repair/'FINAL_RAW_VERIFIED.json'
    if not all(path.is_file() and not path.is_symlink() for path in (queue_path,receipt_path,final_path)):
        raise RuntimeError('DORMANT: repair19 lacks its complete independent postclose evidence')
    queue=json.loads(queue_path.read_text());receipt=json.loads(receipt_path.read_text())
    final=json.loads(final_path.read_text())
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    rebuilt=receipt.get('rebuilt_raw_result',{})
    if (queue.get('state')!='COMPLETE' or queue.get('stage3_admission') is not True
            or queue.get('receipt')!='FINAL_RAW_VERIFIED.json'
            or receipt.get('classification')!='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA'
            or receipt.get('functional_stage3_pass') is not True
            or receipt.get('full_result_exact') is not True
            or receipt.get('schema')!='independent_stage3_postclose_receipt_v1'
            or final.get('classification')!='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA'
            or final.get('functional_stage3_pass') is not True):
        raise RuntimeError('DORMANT: repair19 has no successful scoped functional Stage3 closure')
    if (receipt.get('queue_sha256')!=digest(queue_path)
            or receipt.get('final_raw_sha256')!=digest(final_path)
            or receipt.get('verifier_sha256')!=digest(repair/'verify_repair19.py')
            or receipt.get('contract_sha256')!=digest(repair/'REPAIR19_CONTRACT.json')
            or final.get('verifier_sha256')!=digest(repair/'verify_repair19.py')
            or final.get('contract_sha256')!=digest(repair/'REPAIR19_CONTRACT.json')):
        raise RuntimeError('DORMANT: repair19 postclose receipt is stale or lacks exact provenance')
    if (json.dumps(rebuilt,sort_keys=True,separators=(',',':'),allow_nan=False)
            !=json.dumps(final,sort_keys=True,separators=(',',':'),allow_nan=False)):
        raise RuntimeError('DORMANT: independent rebuilt result differs from the final raw receipt')
    source_hashes=receipt.get('source_hashes',{})
    for path in (postclose/'verify_postclose.py',postclose/'PLAN.json'):
        if source_hashes.get(str(path))!=digest(path):
            raise RuntimeError('DORMANT: postclose receipt has different verifier provenance')
    return {'repair_queue_sha256':digest(queue_path),'postclose_sha256':digest(receipt_path),
            'final_raw_sha256':digest(final_path),'repair_verifier_sha256':digest(repair/'verify_repair19.py'),
            'scope':'Independent receipt for scoped functional Stage3; not strict historical equivalence or navigation'}


def main() -> int:
    if not __debug__:
        raise RuntimeError('Historical loader requires normal Python; -O forbidden')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--field', choices=('plus','minus'), required=True)
    p.add_argument('--engine', choices=('causal_cuda','reference_cuda'), default='causal_cuda')
    a = p.parse_args()
    prep = 40
    trial_ms = 400
    max_wall = 2050
    stage3_receipt=require_stage3_closed()
    source_receipt=require_sources()
    storage = RunStorage(a.out)
    start = time.perf_counter()
    obj = session = flow_tap = drive_observer = undo = auditor = None
    rows = []
    status = 'STARTING'
    error = None
    cleanup = []
    phase = 'loading'
    checkpoint = {'status':'NOT_ATTEMPTED'}
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise InterruptedError('Bounded process terminated')
    signal.signal(signal.SIGTERM, terminated)

    def snapshot(folder: Path) -> None:
        from session_io import write_state
        from operator_state import OperatorState, LEGACY_BINDINGS
        h = obj.core.hybrid
        if getattr(obj.core,'failed',False) or getattr(h,'_native_rebuild_required',False):
            raise RuntimeError('Failed owner cannot publish coherent state')
        write_state(folder/'session', obj.core.state_dict())
        write_state(folder/'prosthesis', obj.state())
        write_state(folder/'published', {'rates':obj.core.brain.rates,
                    'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
        write_state(folder/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
        atomic_json(folder/'boundary.json', obj.core.world.boundary.metadata())

    def save_trace() -> None:
        if rows:
            np.savez_compressed(a.out/'traces.npz',
                **{key:np.asarray([r[key] for r in rows]) for key in rows[0]})

    def preserve_rejected_interval(row,used,phase,step,exc) -> None:
        raw=a.out/'REJECTED_INTERVAL.npz'
        np.savez_compressed(raw,**{key:np.asarray(value) for key,value in row.items()},
                            used_before_step=np.asarray(used))
        atomic_json(a.out/'REJECTED_INTERVAL.json',{
            'phase':phase,'step':step,'error_type':type(exc).__name__,
            'error_message':str(exc),'raw_npz_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
            'accepted_into_trace':False})

    try:
        import cupy as cp
        import pandas as pd
        from runtime_session import RuntimeSession
        obj, d, plan, Field, field, ports = load(a.out/'preparation_inputs')
        h = obj.core.hybrid
        from kcgamma_regional_brain import _record_hash
        import copy
        original_manifest=copy.deepcopy(h.pn_online_manifest)
        if original_manifest['general_outputs']['enabled'] is not True:
            raise ValueError('Parent PN629 route was not enabled')
        h.pn_online_manifest['general_outputs']['enabled']=False
        h.pn_online_manifest['record_sha256']=_record_hash(h.pn_online_manifest)
        h.validate_online()
        atomic_json(a.out/'INTERVENTION.json',{'operation':'disable left PN629 general replacement','before_hash':original_manifest['record_sha256'],'after_hash':h.pn_online_manifest['record_sha256'],'only_changes':['general_outputs.enabled','record_sha256'],'dynamic_466_enabled':h.pn_online_manifest['electrical_outputs']['enabled'],'left_pn_id':10208,'anatomy_changed':False,'motor_decoder_changed':False})
        check=copy.deepcopy(h.pn_online_manifest)
        check['general_outputs']['enabled']=True
        check['record_sha256']=original_manifest['record_sha256']
        if _record_hash(check)!=_record_hash(original_manifest):
            raise ValueError('Unexpected additional PN intervention')
        # RuntimeSession establishes the model source path but imports event
        # modules only during instantiation. Refuse an ambiguous preloaded
        # owner rather than silently mixing historical and local operators.
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            if name in sys.modules:
                raise RuntimeError('Event owner already imported before local candidate: '+name)
        sys.path.insert(0,str(EVENTS))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(module.__file__).resolve().parent != EVENTS:
                raise RuntimeError('Wrong SET/ADD module source: '+module.__name__)
        session = RuntimeSession(h,a.engine)
        orn_rows = {side:np.asarray(obj.core.port_indices['ORN_DM1_'+side],dtype=np.int64)
                    for side in ('L','R')}
        if any(rows.ndim != 1 or len(rows) == 0 or np.any(rows < 0) or
               np.any(rows >= h.brain.n_neurons) for rows in orn_rows.values()):
            raise ValueError('Invalid bilateral ORN indices for actual drive observation')
        selected_rows = tuple(int(i) for side in ('L','R') for i in orn_rows[side][:4])
        if len(set(selected_rows)) != len(selected_rows):
            raise ValueError('Bilateral ORN index overlap')
        drive_observer = DriveObserver(session.adapter,selected_epochs=CAPTURE_EPOCHS,
                                      drive_indices=selected_rows,pn_indices=())
        atomic_json(a.out/'DRIVE_INDEX_MAP.json',{
            'selected_drive_rows':list(selected_rows),
            'ORN_DM1_L_all_rows':orn_rows['L'].tolist(),
            'ORN_DM1_R_all_rows':orn_rows['R'].tolist(),
            'drive_buffer_full_hash_at_selected_calls':True,
            'pn_buffer_full_hash_at_selected_calls':True,
            'pn_buffer_semantics':'Boundary available; general PN629 route disabled, so buffer presence is not proof of consumption',
            'capture_epochs':list(CAPTURE_EPOCHS)})
        metadata = pd.read_parquet(OLD/'data/male_v10/nodes.parquet',
                    columns=['bodyId','type','rootSide','somaSide','instance'])
        flow_tap = NativeFlowTap(h,metadata,a.out/'flow')
        undo = flow_tap.install()
        frozen = a.out/'executed_sources'
        frozen.mkdir()
        manifest = {}
        for index,source in enumerate(source_files()):
            destination = frozen/(str(index)+'__'+source.parent.name+'__'+source.name)
            shutil.copy2(source,destination)
            manifest[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
        atomic_json(a.out/'FROZEN.json',manifest)
        atomic_json(a.out/'RUN_CONTRACT.json',{
            'schema':'stage4_mirrored_source_gate_run_v1','field':a.field,'engine':a.engine,
            'observer':'native_signed_flow_and_bounded_adapter_boundary',
            'drive_capture_epochs':list(CAPTURE_EPOCHS),
            'preparation_ms':prep,'trial_ms':trial_ms,
            'stage3_eligibility':stage3_receipt,
            'source_identity':source_receipt,
            'checkpoint':plan['checkpoint'],
            'checkpoint_manifest_sha256':hashlib.sha256((OLD/plan['checkpoint']/'manifest.json').read_bytes()).hexdigest(),
            'plan_sha256':hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),
            'wall_limit_s':max_wall,'event_boundaries':True,
            'biological_parameters_changed':False,'interface_intervention':'PN629 disabled before preparation; mirrored fixed Gaussian source installed after preparation',
            'source_mode':'plus=left, minus=right; initial common matched to prior negative, signed bilateral contrast reversed',
            'stage3_admission':False,'stage4_admission':False,
            'event_representation':'physical SET(post_q) from both LIF and gamma CUDA publishers; genuine ADD retained; no state clipping',
            'native_sideband':'FP64 raw signed net before tanh in general and PN CUDA kernels',
            'resumable_gaussian':False})
        yaw0 = d.yaw_grados(obj.body.data.qpos)
        geom_path=ROOT/'campanas/etapa4_diseno_20260923_17/GEOMETRY.json'
        gaussian_specs=json.loads((HERE/'CAMPOS.json').read_text())
        preflight=json.loads((ROOT/'campanas/etapa4_next_20260924_25/geometry_preflight_01/RESULT.json').read_text())
        if (preflight['status']!='CPU_COMPLETE_DESCRIPTIVE_ONLY' or
                gaussian_specs['plus']['source_mm']!=preflight['sources']['left']['source_mm'] or
                gaussian_specs['minus']['source_mm']!=preflight['sources']['right']['source_mm'] or
                gaussian_specs['plus']['sigma_mm']!=preflight['sources']['left']['sigma_mm'] or
                gaussian_specs['minus']['sigma_mm']!=preflight['sources']['right']['sigma_mm']):
            raise ValueError('Frozen mirrored sources disagree with CPU preflight')
        atomic_json(a.out/'GAUSSIAN_SPEC.json',gaussian_specs)
        for phase, duration in [('preparacion',prep),('ensayo',trial_ms)]:
            if phase == 'ensayo':
                storage.snapshot('prepared_state',snapshot)
                first={name:field.sample(obj.body.data,spec['source_mm'],spec['sigma_mm'])['concentration']
                       for name,spec in gaussian_specs.items()}
                limits=json.loads((HERE/'PLAN.json').read_text())['preflight']
                common_error=float(abs(np.mean(first['plus'])-np.mean(first['minus'])))
                matched_error=float(abs(np.mean(first['plus'])-preflight['calibration']['old_initial_common']))
                plus_lr=float(first['plus'][0]-first['plus'][1]);minus_lr=float(first['minus'][0]-first['minus'][1])
                contrast_antisymmetry=float(abs(plus_lr+minus_lr))
                if (not np.isfinite([common_error,matched_error,plus_lr,minus_lr]).all() or
                        common_error>limits['initial_common_pair_abs_max'] or
                        matched_error>limits['initial_common_prior_abs_max'] or
                        contrast_antisymmetry>limits['initial_contrast_antisymmetry_abs_max'] or
                        plus_lr<limits['initial_signed_contrast_min'] or
                        minus_lr>-limits['initial_signed_contrast_min']):
                    raise ValueError('Mirrored source prepared input failed frozen common/contrast bounds')
                atomic_json(a.out/'GAUSSIAN_INITIAL_GUARD.json',{'plus':first['plus'].tolist(),
                            'minus':first['minus'].tolist(),'common_pair_abs':common_error,
                            'common_prior_abs':matched_error,'contrast_antisymmetry_abs':contrast_antisymmetry,
                            'plus_signed_LR':plus_lr,'minus_signed_LR':minus_lr,'limits':limits})
                auditor=install_gaussian(obj,field,a.field,gaussian_specs[a.field],rows[-1],geometry_path=geom_path)
                yaw0 = d.yaw_grados(obj.body.data.qpos)
            for k in range(duration):
                if time.perf_counter()-start > max_wall-40:
                    raise TimeoutError('Frozen process budget; saving prefix evidence')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > 18:
                    raise MemoryError('Frozen 18-GiB RAM budget')
                used = obj.core.pending_sensors.copy() if phase == 'preparacion' else auditor.before(obj)
                if drive_observer:
                    drive_observer.begin_epoch(f'{phase}:{k+1}')
                if flow_tap:
                    flow_tap.set_interval(phase,k+1)
                t = time.perf_counter()
                obj.step()
                if drive_observer:
                    drive_observer.end_epoch()
                if flow_tap:
                    flow_tap.sample_last(session.adapter)
                cp.cuda.runtime.deviceSynchronize()
                row = d.captura(obj,ports,phase,k+1,used,yaw0,cp)
                if phase == 'ensayo':
                    try:auditor.after(obj,row)
                    except BaseException as exc:
                        try:preserve_rejected_interval(row,used,phase,k+1,exc)
                        except BaseException as save_exc:
                            cleanup.append('rejected interval preservation: '+repr(save_exc))
                        raise
                rows.append(row)
                record = {'phase':phase,'step':k+1,'clock_ns':int(h.time_ns),
                          'step_wall_s':time.perf_counter()-t,
                          'elapsed_s':time.perf_counter()-start,'yaw_delta_deg':row['yaw_delta_deg'],
                          'rss_peak_sample_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                          'gpu_pool_used_sample_gib':cp.get_default_memory_pool().used_bytes()/1024**3}
                gpu_free,gpu_total=cp.cuda.runtime.memGetInfo()
                record.update(gpu_device_total_gib=gpu_total/1024**3,
                              gpu_device_used_sample_gib=(gpu_total-gpu_free)/1024**3)
                with (a.out/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                if phase == 'preparacion' and np.any(used != 0):
                    raise ValueError('Preparation consumed odor')
                if phase == 'ensayo' and k+1 == 100:
                    storage.snapshot('state_100ms',snapshot)
                if (k+1)%10 == 0:
                    print(json.dumps(record),flush=True)
        status = 'COMPLETE'
    except BaseException as exc:
        error = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                 'numerical_diagnostic':getattr(exc,'numerical_diagnostic',None)}
        status = 'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'INCOMPLETE'
    finally:
        if auditor is not None:
            try:auditor.export(a.out/'GAUSSIAN_INTERVALS.json')
            except BaseException as exc:cleanup.append('gaussian auditor: '+repr(exc));status='INCOMPLETE'
        try:
            save_trace()
        except BaseException as exc:
            cleanup.append('trace: '+repr(exc))
            status = 'INCOMPLETE'
        if session is not None and session.events is not None:
            try:
                atomic_json(a.out/'EVENT_AUDIT.json',{'schema':'stage4_event_owner_log_v1',
                              'profile':a.engine,'field':a.field,'blocks':session.events.audit})
            except BaseException as exc:
                cleanup.append('event audit: '+repr(exc))
                status = 'INCOMPLETE'
        if drive_observer:
            try:
                drive_observer.save(a.out/'DRIVE_OBSERVATION.json')
            except BaseException as exc:
                cleanup.append('drive observer report: '+repr(exc))
                status = 'INCOMPLETE'
            try:drive_observer.close()
            except BaseException as exc:
                cleanup.append('drive observer detach: '+repr(exc))
                status = 'INCOMPLETE'
        if flow_tap:
            try:
                flow_tap.save()
            except BaseException as exc:
                cleanup.append('flow tap report: '+repr(exc))
                status = 'INCOMPLETE'
            try:flow_tap.close()
            except BaseException as exc:
                cleanup.append('flow tap close: '+repr(exc))
                status = 'INCOMPLETE'
        if obj is not None:
            try:
                checkpoint = storage.snapshot('final_state',snapshot)
            except BaseException as exc:
                checkpoint = {'status':'FAILED_NOT_RESUMABLE','message':repr(exc)}
                cleanup.append('checkpoint: '+repr(exc))
                status = 'INCOMPLETE'
        if undo:
            try:undo()
            except BaseException as exc:cleanup.append('flow tap undo: '+repr(exc))
        try:runtime = None if session is None else session.report()
        except BaseException as exc:
            runtime=None;cleanup.append('runtime report: '+repr(exc))
        for closer in ([session.close] if session else [])+([obj.close] if obj else []):
            try:closer()
            except BaseException as exc:cleanup.append('close: '+repr(exc))
        if cleanup and status=='COMPLETE':status='INCOMPLETE_CLEANUP'
        signal.signal(signal.SIGTERM,old_sigterm)
        trial = [r for r in rows if r['fase']=='ensayo']
        atomic_json(a.out/'RESULT.json',{'status':status,'error':error,'cleanup_errors':cleanup,
            'field':a.field,'engine':a.engine,'observer':'on',
            'requested_trial_ms':trial_ms,'completed_trial_ms':len(trial),
            'completed_preparation_ms':len(rows)-len(trial),'wall_total_s':time.perf_counter()-start,
            'checkpoint':checkpoint,'runtime':runtime,'stage3_pass':None,'stage4_pass':None,
            'reason':'Mirrored finite-source input-to-command diagnostic only; not source navigation or biological equivalence.'})
    return 0 if status == 'COMPLETE' and not cleanup else 2


if __name__ == '__main__':
    raise SystemExit(main())
