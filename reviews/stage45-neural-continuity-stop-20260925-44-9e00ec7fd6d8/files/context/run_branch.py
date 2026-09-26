"""Bounded original-engine continuation with sensory and physical evaluator tapes."""
from __future__ import annotations

import argparse
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
DONOR = ROOT / 'campanas/etapa45_navigation_wind_20260925_40'
EVENTS = ROOT / 'campanas/etapa3_pn629_intervention_20260923_15'
SOURCE = DONOR / 'navigation_minus_filtered_wind_03/final_state'
TRACE = DONOR / 'merged_full_01/traces.npz'
TAPES = ROOT / 'campanas/etapa45_orientation_observability_20260925_43/tapes_01/TAPES.npz'
sys.path.insert(0, str(DONOR))
import run_finish as original
from interventions import TapeBoundary, make_frozen_motor, need
from run_storage import RunStorage, atomic_json, verify_snapshot


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def lock():
    original.lock_sources()
    plan = json.loads((HERE / 'PLAN.json').read_text())
    for rel, digest in plan['fixed_sha256'].items():
        need(sha(ROOT / rel) == digest, 'Changed prospective input: ' + rel)
    own = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    for rel, digest in own.items():
        need(sha(ROOT / rel) == digest, 'Changed campaign source: ' + rel)
    return plan


def main():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError('Conserved loader needs normal Python; -O is forbidden')
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['sham','common','virtual'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = lock()
    if args.arm != 'sham':
        verdict = json.loads((HERE/'sham_01/VERIFIED.json').read_text())
        need(verdict['classification'] == 'SHAM_EXACT', 'Sham not admitted')
        need(verdict['plan_sha256'] == sha(HERE/'PLAN.json') and
             verdict['source_lock_sha256'] == sha(HERE/'SOURCE_LOCK.json'), 'Sham contract changed')
    verify_snapshot(SOURCE)
    storage = RunStorage(args.out)
    out = storage.out
    started = time.perf_counter()
    rows, inputs, witness = [], [], []
    attempted_ms = 0
    obj = session = motor = None
    error = None
    checkpoint = {'status':'NOT_ATTEMPTED'}
    status = 'STARTING'
    old_signal = signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise InterruptedError('Finite branch terminated')
    signal.signal(signal.SIGTERM, terminate)
    try:
        import cupy as cp
        from checkpoint_compare_utf8 import compare
        from runtime_session import RuntimeSession
        from pn_cns_ports import PnCnsPorts
        from kcgamma_regional_brain import _record_hash
        obj, diagnostic, *_ = original.load(out/'static_preparation_inputs')
        restored = original.restore(obj, SOURCE)
        storage.snapshot('prepared_state', lambda folder: original.snapshot(obj, folder))
        initial = compare(SOURCE, out/'prepared_state')
        atomic_json(out/'RESTORATION.json', {'construction':restored,'initial':initial})
        need(initial['exact'], 'Complete initial state differs')
        with np.load(TRACE, allow_pickle=False) as z:
            donor = {k:z[k].copy() for k in z.files}
        trial_rows = np.flatnonzero(donor['fase'] == 'ensayo')
        donor['row_by_step'] = {int(donor['paso'][i]):int(i) for i in trial_rows}
        with np.load(TAPES, allow_pickle=False) as z:
            tapes = {k:z[k].copy() for k in z.files}
        last = donor['row_by_step'][1000]
        need(np.array_equal(obj.body.data.qpos, donor['qpos'][last]) and
             np.array_equal(obj.core.pending_sensors, donor['sensores_pendientes'][last]),
             'Original body or pending sensor changed')
        yaw0 = diagnostic.yaw_grados(donor['qpos'][39])
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            need(name not in sys.modules, 'Ambiguous event owner ' + name)
        sys.path.insert(0, str(EVENTS))
        import event_waveform, event_ports, event_coupling, native_cell, device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            need(Path(module.__file__).resolve().parent == EVENTS, 'Wrong event owner')
        h = obj.core.hybrid
        ports = PnCnsPorts(h, 10208)
        session = RuntimeSession(h, 'causal_cuda')
        motor = make_frozen_motor(original.MotorWind)(obj, donor)
        motor.filtered = float(donor['motor_filter_state_rad_s'][last])
        motor.raw = float(donor['neural_command_raw_rad_s'][last])
        motor.applied = float(donor['motor_filter_applied_rad_s'][last])
        motor.forward = float(donor['command_forward_mm_s'][last])
        motor.trial_step, motor.body_calls = 1000, 40000
        field = obj.core.world.boundary
        obj.core.world.boundary = TapeBoundary(field, obj.core.world, args.arm, tapes)
        original_advance = h.advance
        current_inputs = []
        def observed_advance(ns, drive, light):
            current_inputs.append((int(ns),np.asarray(drive).copy(),np.asarray(light).copy()))
            return original_advance(ns, drive, light)
        h.advance = observed_advance
        orn_left = obj.core.port_indices['ORN_DM1_L'].copy()
        orn_right = obj.core.port_indices['ORN_DM1_R'].copy()
        atomic_json(out/'INPUT_SCHEMA.json', {'node_ids_sha256':_record_hash(h.brain.node_ids),
                    'ORN_L_indices':orn_left.tolist(),'ORN_R_indices':orn_right.tolist(),
                    'odor_drive':obj.core.config['odor_drive'],
                    'input_clock':'Step k consumes sample committed after k-1; world clock includes original origin.',
                    'rng_initial':obj.core.brain.rng.bit_generator.state,
                    'neural_advance_owner':str(type(h)),
                    'neural_step_noise_scope':'Record full existing brain RNG before/after and actual drive/light; do not draw new diagnostic samples.'})
        atomic_json(out/'RUN_CONTRACT.json', {'arm':args.arm,'plan_sha256':sha(HERE/'PLAN.json'),
                    'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json'),'tape_sha256':sha(TAPES),
                    'checkpoint_manifest_sha256':sha(SOURCE/'MANIFEST.json'),
                    'start_ns':obj.core.time_ns,'steps':[1001,1120],
                    'body_commands_from_donor':True,'neural_output_delivered':False})
        for step in range(1001,1121):
            need(time.perf_counter()-started < 850, 'Branch wall budget exhausted before checkpoint reserve')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 <= 18, 'RAM budget exceeded')
            need(cp.get_default_memory_pool().total_bytes()/1024**3 <= 12, 'GPU pool budget exceeded')
            before = obj.core.pending_sensors.copy()
            sample_time = int(obj.core.world.time_ns)
            rng_before = _record_hash(obj.core.brain.rng.bit_generator.state)
            proprio_before = _record_hash(obj.core.pending_proprioception)
            light_before = _record_hash(obj.core.pending_light)
            current_inputs.clear()
            tick = time.perf_counter()
            attempted_ms += 1
            obj.step()
            cp.cuda.runtime.deviceSynchronize()
            need(len(current_inputs) == 1 and current_inputs[0][0] == 1_000_000,
                 'Unexpected actual CNS input calls')
            _, drive, light = current_inputs[0]
            row = diagnostic.captura(obj,ports,'ensayo',step,before,yaw0,cp)
            row.update(neural_command_raw_rad_s=motor.raw,
                       motor_filter_applied_rad_s=motor.applied,
                       motor_filter_state_rad_s=motor.filtered,
                       neural_forward_shadow_mm_s=motor.forward,
                       wind_torque_native=(motor.WIND_TORQUE_NATIVE if 1001<=step<=1020 else 0.0))
            need(motor.trial_step == step and motor.body_calls == step*40, 'Motor/body clock differs')
            rows.append(row)
            inputs.append({'drive':drive,'light':light})
            witness.append({'step':step,'sampling_ns':sample_time,'consuming_ns':int(obj.core.time_ns),
                            'rng_before':rng_before,'rng_after':_record_hash(obj.core.brain.rng.bit_generator.state),
                            'pending_proprio_before':proprio_before,'pending_proprio_after':_record_hash(obj.core.pending_proprioception),
                            'pending_light_before':light_before,'pending_light_after':_record_hash(obj.core.pending_light)})
            di = donor['row_by_step'][step]
            natural = field.sample(obj.body.data,obj.core.world.source_mm,obj.core.world.sigma_mm)
            row['natural_concentration'] = natural['concentration'].copy()
            for key in ('qpos','qvel','position_mm','antenas_mm','command_forward_mm_s',
                        'command_yaw_rate_rad_s','contact_active','normal_force_N','contact_force_N',
                        'generalized_force_native','energy_motor_J'):
                need(np.array_equal(row[key],donor[key][di]), 'Yoked body differs: ' + key)
            progress={'step':step,'elapsed_s':time.perf_counter()-started,'step_wall_s':time.perf_counter()-tick}
            with (out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            if step%10==0:print(json.dumps(progress),flush=True)
        need(motor.wind_substeps == 800 and len(motor.substep_forces) == 4800, 'Wrong wind/body dose')
        status = 'COMPLETE'
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        status='INCOMPLETE'
    finally:
        if rows:
            np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
            np.savez_compressed(out/'actual_inputs.npz',**{k:np.asarray([r[k] for r in inputs]) for k in inputs[0]})
            atomic_json(out/'EXOGENOUS_WITNESS.json',witness)
        if motor is not None:
            np.savez_compressed(out/'substep_body.npz',force=np.asarray(motor.substep_forces),
                                command=np.asarray(motor.substep_commands),clock=np.asarray(motor.substep_clock))
        cleanup=[]
        if obj is not None and obj.core is not None:
            try:
                checkpoint=storage.snapshot('final_state',lambda folder:original.snapshot(obj,folder))
            except BaseException as exc:
                cleanup.append('checkpoint: '+repr(exc));status='INCOMPLETE'
        runtime=None
        if session is not None:
            try:runtime=session.report()
            except BaseException as exc:cleanup.append('runtime: '+repr(exc));status='INCOMPLETE'
        for close in ([session.close] if session else [])+([obj.close] if obj and obj.core else []):
            try:close()
            except BaseException as exc:cleanup.append('close: '+repr(exc));status='INCOMPLETE'
        signal.signal(signal.SIGTERM,old_signal)
        atomic_json(out/'RESULT.json', {'schema':'stage45_neural_branch44_v1','status':status,'arm':args.arm,
                    'completed_ms':len(rows),'attempted_ms':attempted_ms,'error':error,'cleanup_errors':cleanup,'checkpoint':checkpoint,
                    'runtime':runtime,'wall_s':time.perf_counter()-started,
                    'cpu_s':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                    'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                    'stage4_admission':False,'stage5_admission':False})
    need(status == 'COMPLETE','Branch did not complete: '+str(error))


if __name__ == '__main__':
    main()
