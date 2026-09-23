"""Replay the original contact body, then test mirrored open-loop commands.

This deliberately does not load or step the 166700-neuron CNS. The saved
controller, MuJoCo body, and contact law are restored from the 40-ms sham
snapshot. Interpretation is allowed only after the original commanded body
trajectory is reproduced. This is a device diagnostic, never a biological
intervention or stage-3 admission.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
SOURCE = ROOT / 'campanas/etapa3_largo_diagnostico_20260923_10/full_sham_01'
PREPARED = SOURCE / 'prepared_state'
sys.path[:0] = [str(OLD/'src'), str(OLD/'work/stage2_prosthesis_cns_20260915')]
from session_io import read_state, sha256
from rh_tarsal_body import RHTarsalBody
from runtime import Runtime as SyntheticRuntime

CONTACT_SOURCE = OLD/'work/stage2_contact_prosthesis_20260915/controller.py'
spec = importlib.util.spec_from_file_location('body_replay_contact_controller', CONTACT_SOURCE)
contact_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(contact_module)
Controller = contact_module.Controller


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def yaw_deg(qpos: np.ndarray) -> float:
    w,x,y,z = np.asarray(qpos)[3:7]
    return float(np.rad2deg(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def angle_delta(a: float,b: float) -> float:
    return float((a-b+180.)%360.-180.)


def source_receipt() -> dict:
    manifest = json.loads((PREPARED/'MANIFEST.json').read_text())
    for name in ('session.json','session.npz','prosthesis.json','prosthesis.npz'):
        entry = manifest['files'][name]
        path = PREPARED/name
        require(path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256'],
                'Prepared state differs from frozen manifest: '+name)
    return {'prepared_manifest_sha256':sha256(PREPARED/'MANIFEST.json'),
            'source_trace_sha256':sha256(SOURCE/'traces.npz'),
            'controller_sha256':sha256(CONTACT_SOURCE),
            'body_replay_source_sha256':sha256(__file__),
            'plan_sha256':sha256(HERE/'PLAN.json')}


class PhysicalOwner:
    _shadow_step = SyntheticRuntime._shadow_step
    def __init__(self,body):
        self.body=body


def run_case(name: str, body_state: dict, controller_state: dict,
             commands: np.ndarray, out: Path, source: dict, reference=None) -> dict:
    require(not out.exists(),'Unique result directory already exists: '+str(out))
    out.mkdir()
    start=time.perf_counter()
    body=None
    try:
        body=RHTarsalBody.from_state(body_state)
        ctrl=Controller.from_state(body.model,body.data,controller_state)
        require(body.dt==25e-6 and ctrl.active and
                (ctrl.activation_is_probe or ctrl.elapsed_ns>=600000000),
                'Prepared body/controller mode differs')
        owner=PhysicalOwner(body)
        q0=body.data.qpos.copy(); yaw0=yaw_deg(q0)
        initial=hashlib.sha256(np.ascontiguousarray(body.integration_state()).tobytes()).hexdigest()
        qposes=[];qvels=[];yaws=[];contact_counts=[];work=[]
        for command in commands:
            forward,yaw_rate=map(float,command)
            require(np.isfinite([forward,yaw_rate]).all(),'Nonfinite command')
            for _ in range(40):
                body.validate_coxa()
                require(not np.any(body.data.xfrc_applied),'Undeclared body wrench')
                owner._shadow_step()
                ctrl.set_command(forward_mm_s=forward,yaw_rate_rad_s=yaw_rate)
                force=ctrl.torque(body.data,body.dt)
                require(force.shape==(body.model.nv,) and np.isfinite(force).all(),
                        'Invalid contact force')
                body.data.ctrl[:]=0.
                body.data.qfrc_applied[:]=force
                mj.mj_step(body.model,body.data)
                body.steps+=1
                body.data.qfrc_applied[:]=0.
                body.data.ctrl[:]=0.
                require(not np.any(body.data.warning.number) and
                        np.isfinite(body.data.qpos).all() and np.isfinite(body.data.qvel).all(),
                        'MuJoCo warning or nonfinite state')
                body.validate_coxa()
            qposes.append(body.data.qpos.copy())
            qvels.append(body.data.qvel.copy())
            yaws.append(angle_delta(yaw_deg(body.data.qpos),yaw0))
            contact_counts.append(int(np.count_nonzero(ctrl.last_contact_active)))
            work.append(float(ctrl.energy_motor_J))
        qposes=np.stack(qposes);qvels=np.stack(qvels);yaws=np.asarray(yaws)
        metrics={'max_qpos_abs':None,'max_qvel_abs':None,'max_yaw_deg_abs':None}
        if reference is not None:
            metrics={'max_qpos_abs':float(np.max(np.abs(qposes-reference['qpos']))),
                     'max_qvel_abs':float(np.max(np.abs(qvels-reference['qvel']))),
                     'max_yaw_deg_abs':float(np.max(np.abs(yaws-reference['yaw_delta_deg'])))}
        np.savez_compressed(out/'trace.npz',qpos=qposes,qvel=qvels,yaw_delta_deg=yaws,
                            command_forward_mm_s=commands[:,0],
                            command_yaw_rate_rad_s=commands[:,1],
                            active_contacts=np.asarray(contact_counts),
                            controller_energy_motor_J=np.asarray(work))
        result={'schema':'body_only_command_replay_v1','name':name,'status':'COMPLETE',
                'scope':'Saved body and synthetic contact prosthesis only; no CNS, odor or biological intervention',
                'prepared_source':source,'initial_integration_sha256':initial,
                'dt_us':25,'ms':len(commands),'final_yaw_deg':float(yaws[-1]),
                'command_integral_deg':float(np.rad2deg(np.sum(commands[:,1])*.001)),
                'contact_count_min':int(min(contact_counts)),
                'contact_count_max':int(max(contact_counts)),
                'reference_errors':metrics,'wall_s':time.perf_counter()-start,
                'stage3_admission':False}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    except BaseException as error:
        (out/'ERROR.json').write_text(json.dumps({'type':type(error).__name__,
                                                  'message':str(error)},indent=2)+'\n')
        raise
    finally:
        if body is not None:
            body.close()


def main() -> None:
    if not __debug__:
        raise RuntimeError('Python -O is not allowed for this diagnostic')
    plan=json.loads((HERE/'PLAN.json').read_text())
    require(plan['budgets']['body_replay_400ms_max']==4,'Frozen budget changed')
    source=source_receipt()
    state=read_state(PREPARED/'session')
    require(state['body']['schema']==RHTarsalBody.SCHEMA,'Wrong saved body type')
    body_state=state['body']
    del state
    prosthesis=read_state(PREPARED/'prosthesis')
    controller_state=prosthesis['controller']
    require(controller_state['schema']==Controller.SCHEMA,'Wrong saved controller type')
    with np.load(SOURCE/'traces.npz',allow_pickle=False) as z:
        trial=np.asarray(z['fase']=='ensayo')
        require(int(np.count_nonzero(trial))==400,'Wrong sham horizon')
        command=np.stack([z['command_forward_mm_s'][trial],
                          z['command_yaw_rate_rad_s'][trial]],axis=1)
        reference={k:z[k][trial].copy() for k in ('qpos','qvel','yaw_delta_deg')}
    require(command.shape==(400,2),'Wrong command shape')
    report={'schema':'body_only_causal_control_close_v1','source':source,
            'commands':{'sham_replay':'saved complete-organism commands',
                        'zero':'forward 0.2 mm/s; yaw 0',
                        'plus':'forward 0.2 mm/s; yaw +0.2 deg/s',
                        'minus':'forward 0.2 mm/s; yaw -0.2 deg/s'},
            'cases':{},'stage3_admission':False}
    all_start=time.perf_counter()
    replay=run_case('sham_replay',body_state,controller_state,command,
                    HERE/'body_sham_replay_02',source,reference)
    report['cases']['sham_replay']=replay
    err=replay['reference_errors']
    report['replay_gate_passed']=bool(err['max_qpos_abs']<=1e-8 and
                                      err['max_qvel_abs']<=1e-8 and
                                      err['max_yaw_deg_abs']<=1e-6)
    (HERE/'CLOSE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    require(report['replay_gate_passed'],'Sham body replay failed frozen gate; no open-loop arms')
    for name,rate in [('zero',0.),('plus',math.radians(.2)),('minus',-math.radians(.2))]:
        commands=np.tile([.2,rate],(400,1))
        result=run_case(name,body_state,controller_state,commands,
                        HERE/('body_'+name+'_02'),source)
        require(result['initial_integration_sha256']==replay['initial_integration_sha256'],
                'Physical arms did not start from same state')
        report['cases'][name]=result
        (HERE/'CLOSE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    zero=report['cases']['zero']['final_yaw_deg']
    plus=report['cases']['plus']['final_yaw_deg']
    minus=report['cases']['minus']['final_yaw_deg']
    report['open_loop']={'zero_yaw_deg':zero,'plus_minus_zero_deg':plus-zero,
                         'minus_minus_zero_deg':minus-zero,
                         'mirror_sum_deg':plus+minus-2*zero,
                         'sham_replay_minus_zero_deg':replay['final_yaw_deg']-zero}
    report['wall_all_s']=time.perf_counter()-all_start
    require(report['wall_all_s']<=plan['budgets']['body_replay_wall_total_s_max'],
            'Physical replay exceeded frozen aggregate wall budget')
    (HERE/'CLOSE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'replay_gate_passed':True,'open_loop':report['open_loop'],
                      'wall_all_s':report['wall_all_s']},allow_nan=False))


if __name__=='__main__':
    main()
