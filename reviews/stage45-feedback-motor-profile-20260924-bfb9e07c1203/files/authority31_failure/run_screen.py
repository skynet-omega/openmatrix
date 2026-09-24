"""Physical authority screen: original commands and two saturated yaw commands.

The brain is never loaded. Odor is evaluated on the resulting antenna poses,
but is never delivered to the body controller. This cannot test navigation.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'campanas/etapa4_reference_budget_20260924_27/reference_plus_01'
PREP = SOURCE/'prepared_state'
FIELDS = ROOT/'campanas/etapa4_mirrored_source_20260924_26/CAMPOS.json'
REPLAY = ROOT/'campanas/etapa3_causal_controls_20260923_11/body_replay.py'
ANTENNA = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_bilateral_entry_20260915/antennal_boundary.py')


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Unavailable source: '+str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_module('preserved_body_replay_31', REPLAY)
antenna_module = load_module('preserved_antenna_boundary_31', ANTENNA)


def need(ok, reason: str):
    if not ok:
        raise ValueError(reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path: Path, obj: dict):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sources():
    manifest = json.loads((PREP/'MANIFEST.json').read_text())
    paths = [PREP/'MANIFEST.json']
    for name in ('session.json','session.npz','prosthesis.json','prosthesis.npz'):
        path = PREP/name
        need(digest(path)==manifest['files'][name]['sha256'], 'Prepared manifest mismatch: '+name)
        paths.append(path)
    paths.extend((SOURCE/'traces.npz', FIELDS, REPLAY, Path(replay.CONTACT_SOURCE),
                  ANTENNA, HERE/'PLAN.json', Path(__file__)))
    return {str(p): digest(p) for p in paths}


def load_inputs():
    plan = json.loads((HERE/'PLAN.json').read_text())
    need(plan['schema']=='stage45_body_authority_screen_v1' and
         plan['budget']['body_replays_max']==3, 'Wrong plan')
    field = json.loads(FIELDS.read_text())['minus']
    source = np.asarray(field['source_mm'], dtype=float)
    sigma = float(field['sigma_mm'])
    need(source.shape==(2,) and np.isfinite(source).all() and sigma>0, 'Source geometry')
    body_state = replay.read_state(PREP/'session')['body']
    controller_state = replay.read_state(PREP/'prosthesis')['controller']
    need(body_state['schema']==replay.RHTarsalBody.SCHEMA and
         controller_state['schema']==replay.Controller.SCHEMA, 'Body/controller schema')
    with np.load(SOURCE/'traces.npz', allow_pickle=False) as z:
        trial = np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==400 and np.array_equal(z['paso'][trial],np.arange(1,401)), 'Donor clock')
        commands = np.stack((z['command_forward_mm_s'][trial],
                             z['command_yaw_rate_rad_s'][trial]),axis=1)
        reference = {key:z[key][trial].copy() for key in ('qpos','qvel','yaw_delta_deg')}
    need(commands.shape==(400,2) and np.isfinite(commands).all(), 'Donor commands')
    return plan,source,sigma,body_state,controller_state,commands,reference


def run_arm(name, body_state, controller_state, commands, source, sigma, started, budget):
    out = HERE/name
    need(not out.exists(), 'Arm already exists: '+name)
    out.mkdir()
    body = None
    arm_start = time.monotonic()
    try:
        body = replay.RHTarsalBody.from_state(body_state)
        ctrl = replay.Controller.from_state(body.model,body.data,controller_state)
        need(body.dt==25e-6 and ctrl.active and not np.any(body.data.xfrc_applied), 'Prepared body')
        owner = replay.PhysicalOwner(body)
        boundary = antenna_module.AntennalBoundary(body.model)
        initial_hash = hashlib.sha256(np.ascontiguousarray(body.integration_state()).tobytes()).hexdigest()
        yaw0 = replay.yaw_deg(body.data.qpos)
        values = {k:[] for k in ('qpos','qvel','yaw_deg','antennae_mm','minus_concentration',
                                 'contacts','upright','normal_N')}
        for ms,(forward,yaw_rate) in enumerate(commands, start=1):
            need(time.monotonic()-started < budget['wall_total_s_max']-10, 'Aggregate wall budget')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 < budget['RAM_GiB_max'],
                 'Physical RAM budget')
            need(np.isfinite([forward,yaw_rate]).all() and abs(float(yaw_rate))<=math.radians(5)+1e-15,
                 'Invalid or over-limit command')
            for _ in range(40):
                body.validate_coxa()
                need(not np.any(body.data.xfrc_applied), 'Undeclared external wrench')
                owner._shadow_step()
                ctrl.set_command(forward_mm_s=float(forward),yaw_rate_rad_s=float(yaw_rate))
                force = ctrl.torque(body.data,body.dt)
                need(force.shape==(body.model.nv,) and np.isfinite(force).all(), 'Contact force')
                body.data.ctrl[:]=0.
                body.data.qfrc_applied[:]=force
                try:
                    mj.mj_step(body.model,body.data)
                    body.steps+=1
                finally:
                    body.data.qfrc_applied[:]=0.
                    body.data.ctrl[:]=0.
                need(not np.any(body.data.warning.number) and np.isfinite(body.data.qpos).all() and
                     np.isfinite(body.data.qvel).all(), 'MuJoCo warning/nonfinite state')
                body.validate_coxa()
            q = body.data.qpos.copy()
            sampled = boundary.sample(body.data,source,sigma)
            values['qpos'].append(q)
            values['qvel'].append(body.data.qvel.copy())
            values['yaw_deg'].append(replay.angle_delta(replay.yaw_deg(q),yaw0))
            values['antennae_mm'].append(sampled['antennae_mm'])
            values['minus_concentration'].append(sampled['concentration'])
            values['contacts'].append(int(np.count_nonzero(ctrl.last_contact_active)))
            values['upright'].append(float(1-2*(q[4]**2+q[5]**2)))
            values['normal_N'].append(float(np.sum(ctrl.last_normal_N)))
        values={key:np.asarray(value) for key,value in values.items()}
        need(all(np.isfinite(value).all() for value in values.values()), 'Trace nonfinite')
        np.savez_compressed(out/'trace.npz', **values)
        result={'schema':'stage45_body_authority_arm_v1','name':name,'status':'COMPLETE',
                'initial_integration_sha256':initial_hash,'ms':len(commands),
                'final_yaw_deg':float(values['yaw_deg'][-1]),
                'contact_count_min':int(np.min(values['contacts'])),
                'upright_min':float(np.min(values['upright'])),
                'wall_s':time.monotonic()-arm_start,
                'trace_sha256':digest(out/'trace.npz'),
                'body_only':True,'stage4_admission':False,'stage5_admission':False}
        save_new(out/'RESULT.json',result)
        return result,values
    except BaseException as error:
        save_new(out/'ERROR.json',{'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if body is not None:
            body.close()


def main():
    if not __debug__:
        raise RuntimeError('Do not optimize away historical body-owner assertions')
    need(not (HERE/'SOURCE_LOCK.json').exists(), 'Screen already attempted')
    plan,source,sigma,body,ctrl,donor,reference = load_inputs()
    save_new(HERE/'SOURCE_LOCK.json',sources())
    started=time.monotonic()
    arms={}
    traces={}
    for name,sign in (('donor',0),('positive_ceiling',1),('negative_ceiling',-1)):
        commands=donor.copy()
        if sign:
            commands[100:,1]=sign*math.radians(plan['criteria']['max_command_abs_deg_s'])
        arm,trace=run_arm(name,body,ctrl,commands,source,sigma,started,plan['budget'])
        arms[name]=arm;traces[name]=trace
        save_new(HERE/f'PROGRESS_{name}.json',{'arm':name,'arm_result_sha256':digest(HERE/name/'RESULT.json'),
                                               'elapsed_s':time.monotonic()-started})
        if name=='donor':
            errors={'qpos':float(np.max(abs(trace['qpos']-reference['qpos']))),
                    'qvel':float(np.max(abs(trace['qvel']-reference['qvel']))),
                    'yaw':float(np.max(abs(trace['yaw_deg']-reference['yaw_delta_deg'])))}
            bounds=plan['criteria']
            need(errors['qpos']<=bounds['baseline_qpos_sup_native_max'] and
                 errors['qvel']<=bounds['baseline_qvel_sup_native_max'] and
                 errors['yaw']<=bounds['baseline_yaw_sup_deg_max'], 'Donor body parity failed')
            save_new(HERE/'DONOR_PARITY.json',errors)
    need(len({arm['initial_integration_sha256'] for arm in arms.values()})==1, 'Physical starts differ')
    base=traces['donor'];pos=traces['positive_ceiling'];neg=traces['negative_ceiling']
    def signed_lr(trace):
        c=trace['minus_concentration']
        return c[:,0]-c[:,1]
    def bearing(trace):
        q=trace['qpos']
        theta=np.arctan2(source[1]-10*q[:,1],source[0]-10*q[:,0])
        quaternion=q[:,3:7]
        w,x,y,z=quaternion.T
        head=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        return np.rad2deg(np.arctan2(np.sin(theta-head),np.cos(theta-head)))
    late=slice(200,400)
    effects={'final_yaw_gap_extremes_deg':float(abs(pos['yaw_deg'][-1]-neg['yaw_deg'][-1])),
             'late_mean_abs_signed_LR_gap_extremes':float(np.mean(abs(signed_lr(pos)[late]-signed_lr(neg)[late]))),
             'max_signed_LR_gap_donor_vs_either':float(max(np.max(abs(signed_lr(base)[100:]-signed_lr(pos)[100:])),
                                                        np.max(abs(signed_lr(base)[100:]-signed_lr(neg)[100:])))),
             'final_bearing_error_deg':{name:float(abs(bearing(trace)[-1])) for name,trace in traces.items()},
             'bearing_error_at_switch_deg':float(abs(bearing(base)[99])),
             'final_minus_LR':{name:float(signed_lr(trace)[-1]) for name,trace in traces.items()},
             'donor_final_command_integral_deg':float(np.rad2deg(np.sum(donor[:,1])*.001))}
    c=plan['criteria']
    checks={'donor_parity':True,
            'same_initial_state':len({arm['initial_integration_sha256'] for arm in arms.values()})==1,
            'extreme_yaw_gap':effects['final_yaw_gap_extremes_deg']>=c['extreme_pair_final_yaw_gap_deg_min'],
            'extreme_field_gap':effects['late_mean_abs_signed_LR_gap_extremes']>=c['extreme_pair_late_mean_abs_signed_LR_gap_min'],
            'baseline_field_gap':effects['max_signed_LR_gap_donor_vs_either']>=c['baseline_vs_extreme_max_signed_LR_gap_min'],
            'contacts':min(a['contact_count_min'] for a in arms.values())>=c['contact_count_each_ms_min'],
            'upright':min(a['upright_min'] for a in arms.values())>=c['upright_each_ms_min'],
            'budget':time.monotonic()-started<=plan['budget']['wall_total_s_max']}
    result={'schema':'stage45_body_authority_screen_result_v1',
            'classification':'PHYSICAL_ENVELOPE_MATERIAL' if all(checks.values()) else 'ENVELOPE_NOT_ESTABLISHED',
            'checks':checks,'effects':effects,'arms':arms,'body_only':True,'organism_runs':0,
            'stage4_admission':False,'stage5_admission':False,
            'source_lock_sha256':digest(HERE/'SOURCE_LOCK.json'),
            'plan_sha256':digest(HERE/'PLAN.json'),'code_sha256':digest(Path(__file__)),
            'wall_total_s':time.monotonic()-started,
            'interpretation':'Saturated-command physical counterfactual, not an upper bound on passive MuJoCo yaw or actual CNS response.'}
    output_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    need(output_bytes <= plan['budget']['output_MiB_max']*1024**2, 'Output budget')
    result['output_bytes']=output_bytes
    save_new(HERE/'RESULT.json',result)
    print(json.dumps({'classification':result['classification'],'effects':effects,'checks':checks,
                      'wall_s':result['wall_total_s']},indent=2,allow_nan=False))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save_new(HERE/'FAILURE.json',{'type':type(error).__name__,'message':str(error),
                                      'stage4_admission':False,'stage5_admission':False})
        raise
