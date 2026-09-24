"""Body-only prospective wind-pulse screen on the current Stage4 physical prep.

The saved neural commands are replayed verbatim; no CNS, PN, GPU, or odor sensor
is advanced. Historical sources are imported read-only, and all new intervention
code lives in this campaign directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import resource
import time
from pathlib import Path

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT/'campanas/etapa3_causal_controls_20260923_11/body_replay.py'
SOURCE = ROOT/'campanas/etapa4_reference_budget_20260924_27/reference_plus_01'
PREP = SOURCE/'prepared_state'

spec = importlib.util.spec_from_file_location('preserved_body_replay_29', OLD)
if spec is None or spec.loader is None:
    raise RuntimeError('Historical body replay unavailable')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def source_lock():
    manifest = json.loads((PREP/'MANIFEST.json').read_text())
    files = [PREP/name for name in ('MANIFEST.json', 'session.json', 'session.npz',
                                   'prosthesis.json', 'prosthesis.npz')]
    files += [SOURCE/'traces.npz', SOURCE/'GAUSSIAN_SPEC.json', OLD,
              ROOT/'campanas/etapa45_pulse_preflight_20260924_29/PLAN.json', Path(__file__),
              Path(replay.CONTACT_SOURCE),
              ROOT/'campanas/etapa3_funcional_20260923_16/body_support.py']
    for name in ('session.json', 'session.npz', 'prosthesis.json', 'prosthesis.npz'):
        need(sha(PREP/name)==manifest['files'][name]['sha256'], 'Prepared source differs: '+name)
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in files}


def load_inputs():
    session = replay.read_state(PREP/'session')
    body_state = session['body']
    del session
    controller_state = replay.read_state(PREP/'prosthesis')['controller']
    need(body_state['schema']==replay.RHTarsalBody.SCHEMA, 'Unexpected body schema')
    need(controller_state['schema']==replay.Controller.SCHEMA, 'Unexpected controller schema')
    need(controller_state['units']['torque_Nm_per_native']==1e-7, 'Torque unit mismatch')
    with np.load(SOURCE/'traces.npz', allow_pickle=False) as z:
        trial = np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==400 and np.array_equal(z['paso'][trial], np.arange(1,401)), 'Wrong trial clock')
        chosen = trial[:200]
        command = np.stack((z['command_forward_mm_s'][chosen], z['command_yaw_rate_rad_s'][chosen]), axis=1)
        reference = {k:z[k][chosen].copy() for k in ('qpos','qvel','yaw_delta_deg','contact_active','normal_force_N','upright')}
    need(command.shape==(200,2) and np.isfinite(command).all(), 'Command domain')
    return body_state, controller_state, command, reference


def nominal_torque(body_state, controller_state, plan):
    body = replay.RHTarsalBody.from_state(body_state)
    try:
        ctrl = replay.Controller.from_state(body.model, body.data, controller_state)
        need(body.dt==25e-6 and ctrl.active, 'Wrong prepared controller/body')
        scratch = mj.MjData(body.model)
        mj.mj_setState(body.model, scratch, body.integration_state(), body.spec)
        mj.mj_forward(body.model, scratch)
        full = np.empty((body.model.nv,body.model.nv))
        mj.mj_fullM(body.model, full, scratch.qM)
        inertia_native = float(full[5,5])
        need(np.isfinite(inertia_native) and inertia_native>0, 'Root yaw inertia')
        pulse_s=(plan['model']['pulse_end_ms_exclusive']-plan['model']['pulse_start_ms'])*.001
        follow_s=(plan['model']['duration_ms']-plan['model']['pulse_start_ms'])*.001
        native_to_Nm=controller_state['units']['torque_Nm_per_native']
        torque_Nm = plan['model']['torque_sign'] * inertia_native * native_to_Nm * math.radians(
            plan['model']['nominal_free_body_target_deg_at_200ms'])/(pulse_s*(follow_s-.5*pulse_s))
        torque_native = torque_Nm/native_to_Nm
        need(np.isfinite(torque_Nm) and 0<abs(torque_Nm)<1, 'Pulse torque outside conservative physical domain')
        # A diagnostic comparison at one prepared pose. The physical run uses mj_applyFT.
        target = np.zeros(body.model.nv)
        mj.mj_applyFT(body.model, scratch, np.zeros(3), np.array([0.,0.,torque_native]),
                      scratch.xpos[ctrl.thorax].copy(), ctrl.thorax, target)
        direct = np.zeros(body.model.nv);direct[5]=torque_native
        return {'M_native_root_yaw': inertia_native, 'torque_Nm_per_native':native_to_Nm,
                'torque_native':torque_native, 'torque_Nm':torque_Nm,
                'equivalence_root5_max_abs_native':float(np.max(abs(target-direct))),
                'equivalence_root5_rel':float(np.max(abs(target-direct))/abs(torque_native)),
                'applyFT_generalized_nonzero_indices':np.flatnonzero(abs(target)>abs(torque_native)*1e-12).tolist(),
                'source_model_nq':int(body.model.nq), 'source_model_nv':int(body.model.nv),
                'prepared_time_s':float(body.data.time)}
    finally:
        body.close()


def run_arm(name, body_state, controller_state, commands, plan, torque, dt_ns, root):
    out=root/name
    need(not out.exists(), 'Unique arm output already exists: '+name)
    out.mkdir()
    start=time.monotonic()
    body=None
    try:
        body=replay.RHTarsalBody.from_state(body_state)
        ctrl=replay.Controller.from_state(body.model,body.data,controller_state)
        need(body.dt==25e-6 and ctrl.active and not np.any(body.data.xfrc_applied), 'Physical preparation')
        if dt_ns==12500:
            body.dt=12.5e-6
            body.model.opt.timestep=12.5e-6
            body.steps*=2
            body.validate_coxa()
        else:
            need(dt_ns==25000, 'Unknown physical step')
        owner=replay.PhysicalOwner(body)
        start_integral_hash=hashlib.sha256(np.ascontiguousarray(body.integration_state()).tobytes()).hexdigest()
        initial_yaw=replay.yaw_deg(body.data.qpos)
        thorax=ctrl.thorax
        n_per_ms=1000000//dt_ns
        need(n_per_ms*dt_ns==1000000, 'Milliseconds not exact')
        values={k:[] for k in ('qpos','qvel','yaw_deg','contacts','normal_N','upright','pulse_native')}
        pulse_count=0
        for ms, command in enumerate(commands):
            forward,yaw=map(float,command)
            need(np.isfinite([forward,yaw]).all(), 'Nonfinite command')
            for _ in range(n_per_ms):
                body.validate_coxa()
                need(not np.any(body.data.xfrc_applied), 'Undeclared force')
                owner._shadow_step()
                ctrl.set_command(forward_mm_s=forward,yaw_rate_rad_s=yaw)
                force=ctrl.torque(body.data,body.dt)
                need(force.shape==(body.model.nv,) and np.isfinite(force).all(), 'Invalid controller force')
                body.data.ctrl[:]=0
                body.data.qfrc_applied[:]=force
                pulse_now = (name!='zero_25us' and plan['model']['pulse_start_ms']<=ms<plan['model']['pulse_end_ms_exclusive'])
                if pulse_now:
                    addition=np.zeros(body.model.nv)
                    mj.mj_applyFT(body.model,body.data,np.zeros(3),np.array([0.,0.,torque['torque_native']]),
                                  body.data.xpos[thorax].copy(),thorax,addition)
                    need(np.isfinite(addition).all(), 'Nonfinite pulse')
                    body.data.qfrc_applied[:]+=addition
                    pulse_count+=1
                mj.mj_step(body.model,body.data)
                body.steps+=1
                body.data.qfrc_applied[:]=0
                body.data.ctrl[:]=0
                need(not np.any(body.data.qfrc_applied) and not np.any(body.data.ctrl), 'Force cleanup failed')
                need(not np.any(body.data.warning.number) and np.isfinite(body.data.qpos).all()
                     and np.isfinite(body.data.qvel).all(), 'MuJoCo warning/nonfinite')
                body.validate_coxa()
            q=body.data.qpos.copy()
            values['qpos'].append(q)
            values['qvel'].append(body.data.qvel.copy())
            values['yaw_deg'].append(replay.angle_delta(replay.yaw_deg(q),initial_yaw))
            values['contacts'].append(int(np.count_nonzero(ctrl.last_contact_active)))
            values['normal_N'].append(float(np.sum(ctrl.last_normal_N)))
            values['upright'].append(float(1-2*(q[4]*q[4]+q[5]*q[5])))
            values['pulse_native'].append(float(torque['torque_native'] if pulse_now else 0.))
        need(pulse_count==(plan['model']['pulse_end_ms_exclusive']-plan['model']['pulse_start_ms'])*n_per_ms if name!='zero_25us' else pulse_count==0, 'Pulse clock/count')
        for key in values: values[key]=np.asarray(values[key])
        need(all(np.isfinite(x).all() for x in values.values()), 'Trace nonfinite')
        np.savez_compressed(out/'trace.npz', **values)
        result={'name':name,'status':'COMPLETE','body_ms':len(commands),'step_ns':dt_ns,
                'initial_integration_sha256':start_integral_hash,
                'final_yaw_deg':float(values['yaw_deg'][-1]),
                'min_contacts':int(np.min(values['contacts'])),
                'min_total_normal_N':float(np.min(values['normal_N'])),
                'min_upright':float(np.min(values['upright'])),
                'pulse_substeps':pulse_count,
                'wall_s':time.monotonic()-start,
                'scope':'Saved neural commands; body/controller/contact only; no olfactory or CNS feedback.'}
        write_new(out/'RESULT.json',result)
        return result,values
    except BaseException as error:
        write_new(out/'ERROR.json',{'name':name,'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if body is not None:body.close()


def main():
    if not __debug__:
        raise RuntimeError('Python -O forbidden for this historical body owner')
    plan=json.loads((HERE/'PLAN.json').read_text())
    need(plan['schema']=='stage45_body_pulse_preflight_v1' and plan['budget']['body_replays_max']==3, 'Plan changed')
    need(not (HERE/'SOURCE_LOCK.json').exists(), 'This preflight has already been attempted')
    lock=source_lock()
    write_new(HERE/'SOURCE_LOCK.json',lock)
    started=time.monotonic()
    state,controller,commands,reference=load_inputs()
    torque=nominal_torque(state,controller,plan)
    write_new(HERE/'PULSE_SPEC.json',torque)
    results={}
    traces={}
    for name,dt in (('zero_25us',25000),('pulse_25us',25000),('pulse_12p5us',12500)):
        need(time.monotonic()-started < plan['budget']['CPU_wall_total_s_max']-15, 'Remaining wall budget')
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        need(peak<plan['budget']['RAM_GiB_max'], 'Memory budget')
        result,trace=run_arm(name,state,controller,commands,plan,torque,dt,HERE)
        results[name]=result;traces[name]=trace
        write_new(HERE/f'PROGRESS_{name}.json',{'completed':name,'elapsed_s':time.monotonic()-started,'result_sha256':sha(HERE/name/'RESULT.json')})
        if name=='zero_25us':
            err={'qpos_max_abs':float(np.max(abs(trace['qpos']-reference['qpos']))),
                 'qvel_max_abs':float(np.max(abs(trace['qvel']-reference['qvel']))),
                 'yaw_max_abs_deg':float(np.max(abs(trace['yaw_deg']-reference['yaw_delta_deg'])))}
            results[name]['reference_errors']=err
            bounds=plan['checks']
            need(err['qpos_max_abs']<=bounds['zero_qpos_max_abs'] and
                 err['qvel_max_abs']<=bounds['zero_qvel_max_abs'] and
                 err['yaw_max_abs_deg']<=bounds['zero_yaw_deg_max_abs'], 'Historical body replay parity failed')
    need(all(r['initial_integration_sha256']==results['zero_25us']['initial_integration_sha256'] for r in results.values()),'Different physical starts')
    c=plan['checks']; z=traces['zero_25us']; p=traces['pulse_25us']; fine=traces['pulse_12p5us']
    effect=float(p['yaw_deg'][-1]-z['yaw_deg'][-1])
    refine_yaw=float(np.max(abs(p['yaw_deg']-fine['yaw_deg'])))
    refine_xy=float(np.max(abs(p['qpos'][:,:2]-fine['qpos'][:,:2]))*10)
    normal_ratio=float(np.min(np.minimum(p['normal_N'],fine['normal_N'])/z['normal_N']))
    science_checks={
        'effect_signed': effect*plan['model']['torque_sign']>0,
        'effect_size':c['pulse_final_yaw_difference_deg_min']<=abs(effect)<=c['pulse_final_yaw_difference_deg_max'],
        'refined_yaw':refine_y<=c['pulse_refinement_yaw_deg_max_abs'],
        'refined_xy':refine_xy<=c['pulse_refinement_xy_mm_max_abs'],
        'contact':min(int(np.min(p['contacts'])),int(np.min(fine['contacts'])))>=c['contact_count_each_ms_min'],
        'support':normal_ratio>=c['normal_force_ratio_vs_zero_min'],
        'upright':min(float(np.min(p['upright'])),float(np.min(fine['upright'])))>=c['upright_each_ms_min']}
    total=time.monotonic()-started
    science_checks['budget']=total<=plan['budget']['CPU_wall_total_s_max']
    result={'schema':'stage45_body_pulse_preflight_result_v1',
            'classification':'PROMETEDOR_NO_CONFIRMADO' if all(science_checks.values()) else 'DESCARTADO_EN_ESTE_CONTRATO',
            'body_only':True,'organism_runs':0,'stage4_admission':False,'stage5_admission':False,
            'torque':torque,'arms':results,'effect_final_deg':effect,
            'refinement_yaw_sup_deg':refine_y,'refinement_xy_sup_mm':refine_xy,
            'normal_force_ratio_min':normal_ratio,'checks':science_checks,'wall_total_s':total,
            'plan_sha256':lock[str((HERE/'PLAN.json').relative_to(ROOT))],
            'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json')}
    write_new(HERE/'RESULT.json',result)
    print(json.dumps({'classification':result['classification'],'effect_final_deg':effect,
                      'checks':science_checks,'wall_total_s':total},allow_nan=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_new(HERE/'FAILURE.json',{'type':type(error).__name__,'message':str(error),
                                        'stage4_admission':False,'stage5_admission':False})
        raise
