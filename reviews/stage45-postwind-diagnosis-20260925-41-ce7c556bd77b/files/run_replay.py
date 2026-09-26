"""Portable intervention on the actual recorded physical command sequence."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from state_io import read_state, write_state


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def yaw(q):
    w,x,y,z=np.asarray(q)[...,3:7].T
    return np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def load_inputs():
    plan=json.loads((HERE/'PLAN.json').read_text())
    env=json.loads((HERE/'ENVIRONMENT.json').read_text())
    need(mj.__version__==env['mujoco'] and np.__version__==env['numpy'],'Pinned numerical runtime differs')
    provenance=json.loads((HERE/'INPUT_PROVENANCE.json').read_text())
    for name,digest in provenance['exported'].items():
        need(sha(HERE/name)==digest,'Input hash: '+name)
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for name,digest in lock.items():
        need(sha(HERE/name)==digest,'Frozen source hash: '+name)
    path=HERE/'vendor/work/stage2_contact_prosthesis_20260915/controller.py'
    spec=importlib.util.spec_from_file_location('portable_contact41',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with np.load(HERE/'inputs/donor_traces.npz',allow_pickle=False) as z:
        trial=np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==2000 and np.array_equal(z['paso'][trial],np.arange(1,2001)),'Donor clock')
        donor={k:z[k][trial].copy() for k in ('qpos','qvel','command_forward_mm_s','command_yaw_rate_rad_s','wind_torque_native')}
    return plan,mod.Controller,donor


def run_arm(name,out,plan,Controller,donor,started,cpu_start):
    out.mkdir(exist_ok=False)
    begin=time.monotonic()
    state=read_state(HERE/'inputs/physical')
    model=mj.MjModel.from_binary_path(str(HERE/'inputs/body.mjb'))
    data=mj.MjData(model)
    mj.mj_setState(model,data,state['integration'],state['spec'])
    ctrl=Controller.from_state(model,data,state['controller'])
    # No mj_forward here: the controller preserves the exact contact cache
    # observation needed for the first force after restoration.
    forward=donor['command_forward_mm_s'].copy()
    command=donor['command_yaw_rate_rad_s'].copy()
    wind=donor['wind_torque_native'].copy()
    switch=plan['criteria']['postwind_first_ms']-1
    if name=='no_wind':wind[:]=0.
    elif name=='zero_yaw_after_wind':command[switch:]=0.
    elif name=='zero_forward_after_wind':forward[switch:]=0.
    elif name!='identity':raise ValueError('Unregistered arm')
    need(np.isfinite(forward).all() and np.isfinite(command).all() and np.isfinite(wind).all(),'Nonfinite command')
    need(abs(model.opt.timestep-25e-6)<1e-20 and state['dt']==25e-6,'Physical grid')
    values={k:[] for k in ('qpos','qvel','time_s','contact_active','upright','energy_motor_J')}
    wind_substeps=0
    try:
        for i in range(2000):
            if i==1000:
                resumed=read_state(HERE/'inputs/restart_physical')
                need(np.array_equal(data.qpos,donor['qpos'][999]) and
                     np.array_equal(data.qvel,donor['qvel'][999]),'Replay prefix changed before historical cold restart')
                data=mj.MjData(model)
                mj.mj_setState(model,data,resumed['integration'],resumed['spec'])
                ctrl=Controller.from_state(model,data,resumed['controller'])
            need(time.monotonic()-started<plan['budget']['wall_s_max'],'Aggregate wall budget')
            need(time.process_time()-cpu_start<plan['budget']['CPU_s_max'],'Aggregate CPU budget')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<plan['budget']['RAM_GiB_max']*1024**3,'RAM budget')
            for j in range(40):
                need(not np.any(data.xfrc_applied),'Undeclared body wrench')
                ctrl.set_command(forward_mm_s=float(forward[i]),yaw_rate_rad_s=float(command[i]))
                force=ctrl.torque(data,model.opt.timestep)
                if wind[i]:
                    extra=np.zeros(model.nv)
                    mj.mj_applyFT(model,data,np.zeros(3),np.array([0.,0.,wind[i]]),data.xpos[ctrl.thorax].copy(),ctrl.thorax,extra)
                    force=force+extra;wind_substeps+=1
                need(np.isfinite(force).all(),'Nonfinite applied force')
                data.ctrl[:]=0.;data.qfrc_applied[:]=force
                try:mj.mj_step(model,data)
                finally:data.ctrl[:]=0.;data.qfrc_applied[:]=0.
                need(not np.any(data.warning.number) and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(),'MuJoCo warning or nonfinite')
            values['qpos'].append(data.qpos.copy());values['qvel'].append(data.qvel.copy())
            values['time_s'].append(float(data.time));values['contact_active'].append(ctrl.last_contact_active.copy())
            values['upright'].append(float(1-2*(data.qpos[4]**2+data.qpos[5]**2)))
            values['energy_motor_J'].append(float(ctrl.energy_motor_J))
        arrays={k:np.asarray(v) for k,v in values.items()}
        arrays.update(forward_mm_s=forward,yaw_rad_s=command,wind_torque_native=wind,step=np.arange(1,2001,dtype=np.int64))
        np.savez_compressed(out/'trace.npz',**arrays)
        final=np.empty(mj.mj_stateSize(model,state['spec']))
        mj.mj_getState(model,data,final,state['spec'])
        write_state(out/'final_physical',dict(integration=final,spec=state['spec'],steps=state['steps']+80000,dt=state['dt'],controller=ctrl.state_dict()))
        errors={k:float(np.max(np.abs(arrays[k]-donor[k]))) for k in ('qpos','qvel')}
        errors['yaw_deg']=float(np.max(np.abs(yaw(arrays['qpos'])-yaw(donor['qpos']))))
        result=dict(arm=name,status='COMPLETE',wall_s=time.monotonic()-begin,wind_substeps=wind_substeps,
                    source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),input_sha256=sha(HERE/'inputs/physical.npz'),
                    trace_sha256=sha(out/'trace.npz'),identity_errors=errors,
                    body_only=True,stage4_admission=False,stage5_admission=False)
        save(out/'RESULT.json',result)
        if name=='identity':
            for key in ('qpos','qvel','yaw_deg'):
                need(errors[key]<=plan['criteria']['identity_'+key+'_sup_max'],'Identity failed: '+key)
        print(json.dumps({'arm':name,'wall_s':result['wall_s'],'identity_errors':errors if name=='identity' else None}),flush=True)
        return result
    except BaseException as error:
        save(out/'ERROR.json',dict(type=type(error).__name__,message=str(error),completed_ms=len(values['qpos'])))
        raise


def main():
    need(__debug__,'Original controller assertions require normal Python; verification supports -O')
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--identity-only',action='store_true')
    a=parser.parse_args();a.out.mkdir(exist_ok=False)
    started=time.monotonic();cpu_start=time.process_time()
    try:
        plan,Controller,donor=load_inputs()
        repair=json.loads((HERE/'REPAIR_EXECUTION.json').read_text())
        names=('identity',) if a.identity_only else tuple(repair['executed_arms'])
        result={name:run_arm(name,a.out/name,plan,Controller,donor,started,cpu_start) for name in names}
        save(a.out/'QUEUE.json',dict(status='COMPLETE',arms=result,wall_s=time.monotonic()-started,
              CPU_s=time.process_time()-cpu_start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              body_only=True,organism_runs=0,previous_attempts=repair['previous_attempts'],
              previous_wall_s=repair['previous_wall_s'],source_lock_sha256=sha(HERE/'SOURCE_LOCK.json')))
    except BaseException as error:
        save(a.out/'FAILURE.json',dict(type=type(error).__name__,message=str(error),wall_s=time.monotonic()-started))
        raise


if __name__=='__main__':main()
