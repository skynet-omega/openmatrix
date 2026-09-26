"""Explicit synthetic leg prosthesis, detached from CNS and native muscles.

The unchanged RHTarsalBody model advances with mj_step. Only leg generalized
forces are applied; root, body xfrc, source motors and adhesion remain unforced.
No live-body pose assignment after declared initialization.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import h5py
import numpy as np
import mujoco as mj

D=Path(__file__).resolve().parent
R=D.parents[1]
spec=importlib.util.spec_from_file_location('prosthesis_body_bank',R/'work/stage2_native_capacity_20260915/bank.py')
bankmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(bankmod)
H5=R/'data/flybody_walking_reference_20260914/walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5'
POSE=R/'work/stage2_unblocking_20260914/selected_posture.npz'
LEGS=[f'T{i}_{side}' for i in (1,2,3) for side in ('left','right')]

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

class LegServo:
    def __init__(self,m,k_scale=1.,limit=.15,kd=.002):
        self.m=m;self.names=[];rows=[];self.k=[];self.bounds=[]
        for leg in LEGS:
            for joint in ('coxa_abduct','coxa_twist','coxa','femur_twist','femur','tibia','tarsus','tarsus2'):
                name=f'{joint}_{leg}';row=np.zeros(m.nv)
                if joint=='tarsus2':
                    for part,weight in [(2,1.),(3,.5),(4,.5),(5,.5)]:
                        row[m.joint(f'tarsus{part}_{leg}').dofadr[0]]=weight
                    self.bounds.append((-.9,.9))
                else:
                    j=m.joint(name);row[j.dofadr[0]]=1.;self.bounds.append(tuple(j.range))
                rows.append(row);self.names.append(name);self.k.append((.8 if joint.startswith(('coxa','femur')) else .4)*k_scale)
        self.J=np.array(rows);self.k=np.array(self.k);self.bounds=np.array(self.bounds)
        self.qadr=np.array([m.jnt_qposadr[m.dof_jntid[v]] for v in range(6,m.nv)])
        self.limit=float(limit);self.kd=float(kd)
        assert not np.any(self.J[:,:6])
        self.allowed=np.any(self.J!=0,axis=0)

    def coordinate(self,qpos):return self.J[:,6:]@qpos[self.qadr]

    def force(self,d,target,target_v=None):
        target=np.clip(target,self.bounds[:,0],self.bounds[:,1])
        v=np.zeros(48) if target_v is None else target_v
        raw=self.k*(target-self.coordinate(d.qpos))-self.kd*(self.J@d.qvel-v)
        actuator=np.clip(raw,-self.limit,self.limit)
        return self.J.T@actuator,actuator,np.abs(raw)>self.limit

def yaw(q):
    w,x,y,z=q[3:7];return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

def up(q):
    w,x,y,z=q[3:7];return 1-2*(x*x+y*y)

def load_reference(m,servo):
    with h5py.File(H5) as h:
        names=[v.decode() for v in h['id2name/qpos'][()]]
        adr=np.array([m.joint(n).qposadr[0] for n in names])
        t=h['trajectories/002'];values=t['qpos'][()];dt=float(h['timestep_seconds'][()]);roots=t['root_qpos'][()]
    refs=np.repeat(m.qpos0[None,:],len(values),axis=0);refs[:,adr]=values
    commands=np.array([servo.coordinate(q) for q in refs])
    return commands,dt,roots

def segment_metrics(rows,mg,start,end):
    a=max(0,start);b=min(end,len(rows)-1)
    if b<=a:return {}
    dur=(rows[b]['elapsed_ms']-rows[a]['elapsed_ms'])*.001
    imp=rows[b]['impulse_Ns']-rows[a]['impulse_Ns'];slip=rows[b]['slip_um']-rows[a]['slip_um']
    delta=(rows[b]['qpos'][:3]-rows[a]['qpos'][:3])*10
    yaws=np.unwrap([yaw(r['qpos']) for r in rows[a:b+1]])
    out=dict(duration_s=dur,load_fraction=imp/(mg*dur),slip_um=slip,displacement_mm=delta,
             horizontal_distance_mm=float(np.linalg.norm(delta[:2])),path_length_mm=float(sum(np.linalg.norm(rows[i]['qpos'][:2]-rows[i-1]['qpos'][:2])*10 for i in range(a+1,b+1))),
             yaw_change_deg=float(np.rad2deg(yaws[-1]-yaws[0])),upright_min=float(min(up(r['qpos']) for r in rows[a:b+1])),
             final_vz_mm_s=float(rows[b]['qvel'][2]*10),max_abs_vz_mm_s=float(max(abs(r['qvel'][2]*10) for r in rows[a:b+1])))
    out['support_checks']={'abdomen':bool(out['load_fraction'][0]<=.01),'legs':bool(out['load_fraction'][1]>=.99),'upright':out['upright_min']>=.9,
                           'height':delta[2]>=-.005,'final_vz':out['final_vz_mm_s']>=-.05,'slip':bool(np.all(slip<=5.))}
    out['static_support_pass']=all(out['support_checks'].values())
    return out

def run(args):
    out=D/args.name;out.mkdir(parents=True,exist_ok=False)
    bank=bankmod.BodyBank(ns=args.dt_ns);b=bank.b;m=b.model;d=b.data
    servo=LegServo(m,args.k_scale,args.limit,args.kd)
    initial_time=float(d.time)
    if args.initial=='selected':
        mj.mj_resetData(m,d)
        with np.load(POSE) as a:d.qpos[:]=a['qpos']
        d.qvel[:]=0
        mj.mj_forward(m,d)
    d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
    with np.load(POSE) as a:hold=servo.coordinate(a['qpos'])
    ref,ref_dt,ref_roots=load_reference(m,servo)
    first=servo.coordinate(d.qpos)
    start_time=float(d.time);bank.ob.reset();rows=[];failure=None;start_wall=time.monotonic()
    max_root_force=0.;max_xforce=0.;saturated=np.zeros(48,dtype=int);max_torque=np.zeros(m.nv);physical_steps=0
    total=args.hold_ms+args.walk_ms+args.end_hold_ms
    final_target=hold.copy()

    def sample(t,phase,act):
        c=bank.ob.read();rows.append(dict(elapsed_ms=t,phase=phase,qpos=d.qpos.copy(),qvel=d.qvel.copy(),
             impulse_Ns=np.array(c['vertical_impulse_Ns']),slip_um=np.array(c['slip_um']),servo_force_native=act.copy()))
    sample(0,'initial',np.zeros(48))
    with bank.ob:
        for ms in range(total):
            for j in range(1000000//args.dt_ns):
                t=(ms+j*args.dt_ns/1e6)*.001
                if t<args.hold_ms*.001:
                    phase='hold';r=np.clip(t/.050,0.,1.);r=r*r*(3-2*r);target=(1-r)*first+r*hold;target_v=None
                elif t<(args.hold_ms+args.walk_ms)*.001:
                    phase='walk';x=(t-args.hold_ms*.001)/ref_dt*args.speed
                    i=min(int(x),len(ref)-2);a=min(x-i,1.)
                    target=(1-a)*ref[i]+a*ref[i+1];target_v=(ref[i+1]-ref[i])/ref_dt*args.speed
                    final_target=target.copy()
                else:phase='end_hold';target=final_target;target_v=None
                force,act,sat=servo.force(d,target,target_v)
                # Exclusive force owner: no native muscle call, no retained old force.
                d.ctrl[:]=0;d.qfrc_applied[:]=force;d.xfrc_applied[:]=0
                assert not np.any(force[~servo.allowed])
                max_root_force=max(max_root_force,float(np.max(np.abs(force[:6]))));max_xforce=max(max_xforce,float(np.max(np.abs(d.xfrc_applied))))
                max_torque=np.maximum(max_torque,np.abs(force));saturated+=sat;physical_steps+=1
                mj.mj_step(m,d)
                if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    failure='MuJoCo warning or nonfinite state';break
            sample(ms+1,phase,act)
            if failure:break
    mg=bank.ob.read()['mgN'];n=len(rows)-1
    metrics=dict(schema='synthetic_leg_prosthesis_v1',name=args.name,parameters=vars(args),completed_ms=n,failure=failure,wall_s=time.monotonic()-start_wall,
        initialization=('independent selected reference 002/0, velocities zero' if args.initial=='selected' else 'exact CNS223 descending body at 77 ms'),
        core_body_time_s=initial_time,start_body_time_s=start_time,final_body_time_s=float(d.time),model_identity=b.identity,
        root_free=bool(m.jnt_type[0]==int(mj.mjtJoint.mjJNT_FREE)),max_root_applied_native=max_root_force,max_body_xfrc_applied=max_xforce,
        leg_servo_names=servo.names,actuator_torque_limit_Nm=args.limit*1e-7,max_joint_torque_Nm=max_torque*1e-7,
        saturation_fraction=saturated/max(physical_steps,1),muscle_force_owner='detached; muscles not advanced; old qfrc cleared before first physical step',
        CNS_changed=False,biological_validation=False,reference_trajectory='002',reference_timestep_s=ref_dt,
        hold_final20=segment_metrics(rows,mg,max(0,args.hold_ms-20),args.hold_ms),
        hold_final100=segment_metrics(rows,mg,max(0,args.hold_ms-100),args.hold_ms),
        walk=segment_metrics(rows,mg,args.hold_ms,args.hold_ms+args.walk_ms),
        final20=segment_metrics(rows,mg,max(0,n-20),n),total=segment_metrics(rows,mg,0,n))
    np.savez_compressed(out/'trace.npz',**{k:np.array([row[k] for row in rows]) for k in rows[0]})
    state=np.empty(mj.mj_stateSize(m,mj.mjtState.mjSTATE_INTEGRATION));mj.mj_getState(m,d,state,mj.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(out/'checkpoint.npz',integration=state,qpos=d.qpos,qvel=d.qvel,final_target=final_target,servo_J=servo.J,servo_k=servo.k)
    (out/'metrics.json').write_text(json.dumps(metrics,default=plain,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ('name','completed_ms','failure','wall_s','hold_final20','walk','final20')},default=plain),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--initial',choices=['selected','core'],default='selected')
    p.add_argument('--dt-ns',type=int,default=25000);p.add_argument('--k-scale',type=float,default=1.);p.add_argument('--limit',type=float,default=.15);p.add_argument('--kd',type=float,default=.002)
    p.add_argument('--hold-ms',type=int,default=200);p.add_argument('--walk-ms',type=int,default=500);p.add_argument('--end-hold-ms',type=int,default=100);p.add_argument('--speed',type=float,default=1.)
    args=p.parse_args();assert 1000000%args.dt_ns==0;run(args)
