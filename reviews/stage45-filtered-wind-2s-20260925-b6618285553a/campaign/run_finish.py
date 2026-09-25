"""Cold continuation of the saved 1000-ms organism to 2000 ms and wind."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EVENTS = ROOT/'campanas/etapa3_pn629_intervention_20260923_15'
PREFIX = HERE/'navigation_minus_filtered_wind_03'
sys.path[:0] = [str(OLD/'work/motor14_20260922'), str(OLD/'work/motor13_20260922'),
                str(ROOT/'motor_nuevo/pipeline_review_20260922'),
                str(ROOT/'campanas/etapa3_continuation_repair_20260923_19'),
                str(EVENTS), str(HERE)]
from motor_runtime import load
from run_storage import RunStorage, atomic_json, verify_snapshot
from restore_gaussian import restore
from motor_wind import MotorWind


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def lock_sources():
    original = json.loads((HERE/'SOURCE_LOCK.json').read_text())
    need(all(Path(path).is_file() and sha(path) == digest
             for path, digest in original.items()), 'Original frozen source changed')
    own = json.loads((HERE/'FINISH_SOURCE_LOCK.json').read_text())
    need(all(Path(path).is_file() and sha(path) == digest
             for path, digest in own.items()), 'Continuation source changed')
    return dict(original_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
                finish_lock_sha256=sha(HERE/'FINISH_SOURCE_LOCK.json'),
                original_source_count=len(original), finish_source_count=len(own))


def snapshot(obj, folder):
    from session_io import write_state
    from operator_state import OperatorState, LEGACY_BINDINGS
    h = obj.core.hybrid
    need(not obj.core.failed and not getattr(h, '_native_rebuild_required', False),
         'Failed owner cannot publish coherent state')
    write_state(folder/'session', obj.core.state_dict())
    write_state(folder/'prosthesis', obj.state())
    write_state(folder/'published', {'rates':obj.core.brain.rates,
                                    'time_ns':obj.core.brain.time_ns,
                                    'rng':obj.core.brain.rng.bit_generator.state})
    write_state(folder/'effective_operator', OperatorState(h, LEGACY_BINDINGS).state_dict())
    atomic_json(folder/'boundary.json', obj.core.world.boundary.metadata())


def main():
    if not __debug__:
        raise RuntimeError('Historical loader requires normal Python; -O forbidden')
    if len(sys.argv) != 2:
        raise SystemExit('Usage: run_finish.py NEW_OUTPUT_DIRECTORY')
    out = Path(sys.argv[1]).resolve()
    source = PREFIX/'final_state'
    lock = lock_sources()
    verify_snapshot(source)
    prefix_result = json.loads((PREFIX/'RESULT.json').read_text())
    need(prefix_result['status'] == 'INCOMPLETE' and
         prefix_result['completed_trial_ms'] == 1000 and
         prefix_result['completed_preparation_ms'] == 40 and
         prefix_result['error']['message'] == 'Unregistered snapshot name',
         'Prefix is not the exact 1000-ms storage-only failure')
    from checkpoint_compare_utf8 import compare
    from runtime_session import RuntimeSession
    storage = RunStorage(out)
    started = time.perf_counter()
    obj = session = motor = None
    rows = []
    cleanup = []
    error = None
    status = 'STARTING'
    restoration = None
    checkpoint = {'status':'NOT_ATTEMPTED'}
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def terminated(*_):
        raise InterruptedError('Bounded continuation terminated')
    signal.signal(signal.SIGTERM, terminated)
    try:
        import cupy as cp
        from pn_cns_ports import PnCnsPorts
        obj, d, plan, Field, base, old_ports = load(out/'static_preparation_inputs')
        del Field, base, old_ports
        restoration = restore(obj, source)
        storage.snapshot('prepared_state', lambda folder: snapshot(obj, folder))
        initial = compare(source, out/'prepared_state')
        atomic_json(out/'RESTORATION.json', {'construction':restoration,
                                            'initial':initial})
        need(initial['exact'], 'Restored complete state differs before next step')
        h = obj.core.hybrid
        ports = PnCnsPorts(h, 10208)
        with np.load(PREFIX/'traces.npz', allow_pickle=False) as z:
            phases = z['fase']
            trial = np.flatnonzero(phases == 'ensayo')
            need(len(trial) == 1000 and len(phases) == 1040,
                 'Wrong prefix trace length')
            last = int(trial[-1])
            yaw0 = d.yaw_grados(z['qpos'][39])
            need(np.array_equal(obj.body.data.qpos, z['qpos'][last]) and
                 np.array_equal(obj.core.pending_sensors, z['sensores_pendientes'][last]),
                 'Restored body/sensor disagrees with exposed 1000-ms trace')
            filter_state = float(z['motor_filter_state_rad_s'][last])
            last_raw = float(z['neural_command_raw_rad_s'][last])
            last_applied = float(z['motor_filter_applied_rad_s'][last])
            last_forward = float(z['command_forward_mm_s'][last])
            need(np.isfinite([filter_state, last_raw, last_applied, last_forward]).all()
                 and obj.controller.yaw_rate_rad_s == last_applied,
                 'Filter/controller boundary differs')
        # RuntimeSession owns the event kernels. Import its local event model
        # before construction, exactly as in the interrupted prefix.
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            need(name not in sys.modules, 'Ambiguous preloaded event owner '+name)
        sys.path.insert(0, str(EVENTS))
        import event_waveform, event_ports, event_coupling, native_cell, device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            need(Path(module.__file__).resolve().parent == EVENTS,
                 'Wrong event owner '+module.__name__)
        session = RuntimeSession(h, 'causal_cuda')
        motor = MotorWind(obj)
        motor.filtered = filter_state
        motor.raw = last_raw
        motor.applied = last_applied
        motor.forward = last_forward
        motor.trial_step = 1000
        motor.body_calls = 40000
        atomic_json(out/'FILTER_RESTORE.json',
                    {'source_trace_sha256':sha(PREFIX/'traces.npz'),
                     'trial_step':motor.trial_step,'body_calls':motor.body_calls,
                     'filtered_rad_s':filter_state,'last_raw_rad_s':last_raw,
                     'last_applied_rad_s':last_applied,
                     'last_forward_mm_s':last_forward,
                     'zero_wind_substeps_before_continuation':True})
        atomic_json(out/'GAUSSIAN_SPEC.json', json.loads((PREFIX/'GAUSSIAN_SPEC.json').read_text()))
        atomic_json(out/'RUN_CONTRACT.json',
                    {'schema':'stage45_cold_continue_1000_to_2000_v1',
                     'source_checkpoint_manifest_sha256':sha(source/'MANIFEST.json'),
                     'source_result_sha256':sha(PREFIX/'RESULT.json'),
                     'source_trace_sha256':sha(PREFIX/'traces.npz'),
                     'source_identity':lock,'engine':'causal_cuda',
                     'first_trial_step':1001,'last_trial_step':2000,
                     'wind_first_step':MotorWind.WIND_FIRST_STEP,
                     'wind_last_step':MotorWind.WIND_LAST_STEP,
                     'wall_limit_s':4500,'stage4_admission':False,
                     'stage5_admission':False})
        for step in range(1001, 2001):
            if time.perf_counter()-started > 4460:
                raise TimeoutError('Frozen continuation wall budget')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > 18:
                raise MemoryError('Frozen 18-GiB RAM budget')
            before = obj.core.pending_sensors.copy()
            tick = time.perf_counter()
            obj.step()
            cp.cuda.runtime.deviceSynchronize()
            row = d.captura(obj, ports, 'ensayo', step, before, yaw0, cp)
            need(motor.trial_step == step and motor.body_calls == step*40,
                 'Filter/body clocks disagree')
            row['neural_command_raw_rad_s'] = motor.raw
            row['motor_filter_applied_rad_s'] = motor.applied
            row['motor_filter_state_rad_s'] = motor.filtered
            row['wind_torque_native'] = (MotorWind.WIND_TORQUE_NATIVE if
                                          MotorWind.WIND_FIRST_STEP <= step <= MotorWind.WIND_LAST_STEP
                                          else 0.0)
            dq = row['DN_q_usada']-row['DN_baseline']
            expected_raw = float(np.tanh(250*(dq[2]-dq[3])) * np.deg2rad(5.))
            need(expected_raw == row['neural_command_raw_rad_s'] and
                 row['command_yaw_rate_rad_s'] == row['motor_filter_applied_rad_s'],
                 'Raw/applied motor differs from actual DN/controller')
            geom = obj.core.world.boundary.sample(obj.body.data,
                       obj.core.world.source_mm, obj.core.world.sigma_mm)
            need(np.array_equal(row['antenas_mm'], geom['antennae_mm']) and
                 np.array_equal(row['concentracion_campo'], geom['concentration']) and
                 np.array_equal(obj.core.pending_sensors,
                                np.r_[geom['concentration'], 0.]),
                 'Gaussian pending/body sample differs')
            rows.append(row)
            record = {'phase':'ensayo','step':step,'clock_ns':int(h.time_ns),
                      'step_wall_s':time.perf_counter()-tick,
                      'elapsed_s':time.perf_counter()-started,
                      'yaw_delta_deg':row['yaw_delta_deg']}
            with (out/'PROGRESS.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False)+'\n')
            if step % 10 == 0:
                print(json.dumps(record), flush=True)
        need(motor.wind_substeps == 800, 'Wind did not occupy exactly 800 physical steps')
        status = 'COMPLETE'
    except BaseException as exc:
        error = {'type':type(exc).__name__,'message':str(exc),
                 'traceback':traceback.format_exc(),
                 'numerical_diagnostic':getattr(exc,'numerical_diagnostic',None)}
        status = 'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'INCOMPLETE'
    finally:
        if rows:
            try:
                np.savez_compressed(out/'traces.npz',
                                    **{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
            except BaseException as exc:
                cleanup.append('trace: '+repr(exc));status='INCOMPLETE'
        if motor is not None:
            try:atomic_json(out/'MOTOR_WIND_AUDIT.json', motor.audit())
            except BaseException as exc:cleanup.append('motor audit: '+repr(exc));status='INCOMPLETE'
        if session is not None and session.events is not None:
            try:atomic_json(out/'EVENT_AUDIT.json',
                            {'profile':'causal_cuda','blocks':session.events.audit})
            except BaseException as exc:cleanup.append('events: '+repr(exc));status='INCOMPLETE'
        if obj is not None and obj.core is not None:
            try:checkpoint=storage.snapshot('final_state',lambda folder:snapshot(obj,folder))
            except BaseException as exc:
                checkpoint={'status':'FAILED_NOT_RESUMABLE','message':repr(exc)}
                cleanup.append('checkpoint: '+repr(exc));status='INCOMPLETE'
        try:runtime=None if session is None else session.report()
        except BaseException as exc:runtime=None;cleanup.append('runtime: '+repr(exc));status='INCOMPLETE'
        for close in ([session.close] if session else [])+([obj.close] if obj and obj.core else []):
            try:close()
            except BaseException as exc:cleanup.append('close: '+repr(exc));status='INCOMPLETE'
        signal.signal(signal.SIGTERM,old_sigterm)
        atomic_json(out/'RESULT.json',
                    {'schema':'stage45_cold_continue_1000_to_2000_v1',
                     'status':status,'error':error,'cleanup_errors':cleanup,
                     'completed_continuation_ms':len(rows),
                     'first_trial_step':1001,'last_trial_step':1000+len(rows),
                     'wall_total_s':time.perf_counter()-started,
                     'restoration':restoration,'checkpoint':checkpoint,
                     'runtime':runtime,'stage4_admission':False,'stage5_admission':False})
    return 0 if status=='COMPLETE' and not cleanup else 2


if __name__ == '__main__':
    raise SystemExit(main())
