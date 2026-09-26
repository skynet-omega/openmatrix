"""Reconstruct diagnosis from raw arrays and frozen contract, including -O."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def raw(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def geometry(q,source):
    w,x,y,z=q[:,3:7].T
    heading=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    bearing=np.unwrap(np.arctan2(source[1]-10*q[:,1],source[0]-10*q[:,0]))
    signed=np.rad2deg(np.arctan2(np.sin(bearing-heading),np.cos(bearing-heading)))
    need(np.all(np.abs(signed)<170),'Branch cut not covered by this decomposition')
    return dict(yaw_deg=np.rad2deg(heading),bearing_deg=np.rad2deg(bearing),
                signed_error_deg=signed,abs_error_deg=np.abs(signed),
                distance_mm=np.linalg.norm(q[:,:2]*10-source,axis=1))


def calculate(plan,donor,arms,source):
    need(tuple(arms)==tuple(read(HERE/'REPAIR_EXECUTION.json')['executed_arms']),'Arms/order differ from repair allocation')
    c=plan['criteria'];result={}
    for name,a in arms.items():
        need(a['step'].dtype.kind in 'iu' and np.array_equal(a['step'],np.arange(1,2001)),'Step clock')
        need(a['qpos'].shape==donor['qpos'].shape and a['qvel'].shape==donor['qvel'].shape,'State dimensions')
        for k,v in a.items():
            need(v.shape[0]==2000 and np.isfinite(v).all(),'Nonfinite/extra/short array: '+name+'/'+k)
        need(a['contact_active'].dtype==np.bool_ and a['contact_active'].shape==(2000,6),'Contact flags')
        origin=read(HERE/'inputs/physical.json')['steps']*plan['criteria']['dt_ns']*1e-9
        need(np.max(np.abs(a['time_s']-(origin+np.arange(1,2001)*.001)))<1e-9,'Physical clock drift')
        forward=donor['command_forward_mm_s'].copy();yaw=donor['command_yaw_rate_rad_s'].copy()
        wind=donor['wind_torque_native'].copy()
        if name=='no_wind':wind[:]=0.
        if name=='zero_yaw_after_wind':yaw[1020:]=0.
        if name=='zero_forward_after_wind':forward[1020:]=0.
        for k,v in [('forward_mm_s',forward),('yaw_rad_s',yaw),('wind_torque_native',wind)]:
            need(np.array_equal(a[k],v),'Intervention context differs: '+name+'/'+k)
        if name!='identity':
            prefix=1000 if name=='no_wind' else 1020
            for k in ('qpos','qvel','time_s','contact_active'):
                need(np.array_equal(a[k][:prefix],arms['identity'][k][:prefix]),'Changed preintervention prefix')
        g=geometry(a['qpos'],source);i=1019
        delta_b=g['bearing_deg'][-1]-g['bearing_deg'][i]
        delta_y=g['yaw_deg'][-1]-g['yaw_deg'][i]
        delta_e=g['signed_error_deg'][-1]-g['signed_error_deg'][i]
        need(abs(delta_e-(delta_b-delta_y))<c['numeric_report_tolerance'],'Geometric identity')
        minimum_contacts=int(np.count_nonzero(a['contact_active'],axis=1).min())
        minimum_upright=float(np.min(1-2*(a['qpos'][:,4]**2+a['qpos'][:,5]**2)))
        need(np.max(abs(a['upright']-(1-2*(a['qpos'][:,4]**2+a['qpos'][:,5]**2))))<1e-15,'Upright receipt differs')
        result[name]=dict(final_abs_error_deg=float(g['abs_error_deg'][-1]),
            postwind_start_abs_error_deg=float(g['abs_error_deg'][i]),
            postwind_abs_error_change_deg=float(g['abs_error_deg'][-1]-g['abs_error_deg'][i]),
            final_distance_mm=float(g['distance_mm'][-1]),
            delta_bearing_deg=float(delta_b),delta_yaw_deg=float(delta_y),delta_signed_error_deg=float(delta_e),
            postwind_yaw_command_integral_deg=float(np.rad2deg(np.sum(a['yaw_rad_s'][1020:])*.001)),
            minimum_contacts=minimum_contacts,minimum_upright=minimum_upright,
            support_domain=bool(minimum_contacts>=c['minimum_contacts'] and minimum_upright>=c['minimum_upright']))
    identity=arms['identity'];errs={k:float(np.max(abs(identity[k]-donor[k]))) for k in ('qpos','qvel')}
    errs['yaw_deg']=float(np.max(abs(geometry(identity['qpos'],source)['yaw_deg']-geometry(donor['qpos'],source)['yaw_deg'])))
    for k,v in errs.items():need(v<=c['identity_'+k+'_sup_max'],'Identity parity failed')
    base=result['identity']['final_abs_error_deg']
    effects={name:float(base-r['final_abs_error_deg']) for name,r in result.items() if name!='identity'}
    material={name:bool(value>=c['effect_bearing_deg_min']) for name,value in effects.items()}
    complete=set(arms)==set(plan['arms'])
    classification='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA' if complete and all(r['support_domain'] for r in result.values()) else 'PROMETEDOR_NO_CONFIRMADO'
    return dict(schema='postwind_mechanical_diagnosis_v1',classification=classification,
         scope='Mechanical replay of one exposed command train, not neural feedback or navigation',
         stage4_admission=False,stage5_admission=False,organism_runs=0,
         identity_errors=errs,arms=result,improvement_vs_identity_deg=effects,
         all_registered_conditions_complete=complete,not_run=sorted(set(plan['arms'])-set(arms)),
         material_improvement=material,all_arms_in_support_domain=all(r['support_domain'] for r in result.values()),
         inference_limit='Ablations need not add linearly. Freezing commands blocks altered sensory-neural feedback. Stopping forward motion is diagnostic, not a navigation policy.')


def verify(folder,mutate=None):
    plan=read(HERE/'PLAN.json');lock=read(HERE/'SOURCE_LOCK.json')
    for name,digest in lock.items():need(sha(HERE/name)==digest,'Frozen source/context/criterion changed: '+name)
    provenance=read(HERE/'INPUT_PROVENANCE.json')
    for name,digest in provenance['exported'].items():need(sha(HERE/name)==digest,'Input hash changed: '+name)
    queue=read(folder/'QUEUE.json')
    need(queue['status']=='COMPLETE' and queue['body_only'] is True and queue['organism_runs']==0,'Queue flags')
    need(queue['source_lock_sha256']==sha(HERE/'SOURCE_LOCK.json'),'Queue source identity')
    for k,b in [('wall_s','wall_s_max'),('CPU_s','CPU_s_max')]:
        need(np.isfinite(queue[k]) and 0<=queue[k]<=plan['budget'][b],'Budget '+k)
    need(queue['peak_RSS_bytes']<=plan['budget']['RAM_GiB_max']*1024**3,'Memory budget')
    data=raw(HERE/'inputs/donor_traces.npz');trial=np.flatnonzero(data['fase']=='ensayo')
    need(len(trial)==2000 and np.array_equal(data['paso'][trial],np.arange(1,2001)),'Donor trial clock')
    for clock in ('CNS_time_ns','PN_time_ns','body_time_ns'):
        need(np.array_equal(data[clock][trial],read(HERE/'inputs/physical.json')['steps']*plan['criteria']['dt_ns']+np.arange(1,2001)*1000000),'Donor clocks')
    donor={k:data[k][trial] for k in ('qpos','qvel','command_forward_mm_s','command_yaw_rate_rad_s','wind_torque_native')}
    source=np.asarray(read(HERE/'inputs/donor_GAUSSIAN_SPEC.json')['minus']['source_mm'])
    need(source.shape==(2,) and np.isfinite(source).all(),'Source context')
    arms={};meta={}
    for name in read(HERE/'REPAIR_EXECUTION.json')['executed_arms']:
        r=read(folder/name/'RESULT.json');meta[name]=r
        if mutate=='flag' and name=='identity':r['stage4_admission']=True
        need(r['arm']==name and r['status']=='COMPLETE' and r['body_only'] is True and
             r['stage4_admission'] is False and r['stage5_admission'] is False,'Arm flag mismatch')
        need(r['trace_sha256']==sha(folder/name/'trace.npz'),'Trace hash mismatch')
        need(r['source_lock_sha256']==sha(HERE/'SOURCE_LOCK.json'),'Arm sources changed')
        arms[name]=raw(folder/name/'trace.npz')
        need(r['wind_substeps']==40*np.count_nonzero(arms[name]['wind_torque_native']),'Wind dose receipt')
    if mutate=='context':arms['zero_yaw_after_wind']['forward_mm_s'][1050]+=.01
    if mutate=='criterion':
        edited=copy.deepcopy(plan);edited['criteria']['effect_bearing_deg_min']=0.
        need(edited==read(HERE/'PLAN.json'),'Frozen criterion corruption detected')
    result=calculate(plan,donor,arms,source)
    result['source_lock_sha256']=sha(HERE/'SOURCE_LOCK.json')
    result['trace_sha256']={n:sha(folder/n/'trace.npz') for n in arms}
    result['budget_consumed']={k:queue[k] for k in ('wall_s','CPU_s','peak_RSS_bytes')}
    result['budget_consumed']['wall_with_failed_identity_s']=queue['wall_s']+queue['previous_wall_s']
    need(result['budget_consumed']['wall_with_failed_identity_s']<=plan['budget']['wall_s_max'],'Aggregate failed+completed wall budget')
    need(len(arms)+queue['previous_attempts']<=plan['budget']['body_replays_max'],'Execution count budget')
    result['simulated_body_ms']=8000
    result['verifier_source_sha256']=sha(Path(__file__))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'run_01')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--corruption-tests',action='store_true')
    a=p.parse_args();r=verify(a.run)
    if a.corruption_tests:
        rejected=[]
        for mode in ('context','flag','criterion'):
            try:verify(a.run,mode)
            except ValueError:rejected.append(mode)
            else:raise ValueError('Corruption undetected: '+mode)
        r['corruptions_rejected']=rejected
    with a.out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r,allow_nan=False))


if __name__=='__main__':main()
