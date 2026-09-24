"""Independent raw-array readback of the bounded physical authority screen."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'campanas/etapa4_reference_budget_20260924_27/reference_plus_01/traces.npz'
FIELDS=ROOT/'campanas/etapa4_mirrored_source_20260924_26/CAMPOS.json'
ARMS=('donor','positive_ceiling','negative_ceiling')


def need(ok,reason):
    if not ok:raise ValueError(reason)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def finite(array,label):
    a=np.asarray(array)
    need(a.dtype.kind in 'biuf' and np.isfinite(a).all(), 'Invalid numeric '+label)
    return a


def yaw(q):
    w,x,y,z=np.asarray(q)[...,3:7].T
    return np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def delta(a,b):
    return (a-b+180)%360-180


def bearing(q,source):
    source_direction=np.arctan2(source[1]-10*q[:,1],source[0]-10*q[:,0])
    body_direction=np.deg2rad(yaw(q))
    return np.rad2deg(np.arctan2(np.sin(source_direction-body_direction),
                                  np.cos(source_direction-body_direction)))


def verify_arrays(data, original, specs, expected, result):
    need(result['status']=='COMPLETE' and result['name']==expected, 'Arm completion/identity')
    need(set(data)=={'qpos','qvel','yaw_deg','antennae_mm','minus_concentration',
                     'contacts','upright','normal_N'}, 'Raw trace keys')
    q=finite(data['qpos'],'qpos');v=finite(data['qvel'],'qvel')
    antenna=finite(data['antennae_mm'],'antennae')
    concentration=finite(data['minus_concentration'],'concentration')
    need(q.shape[0]==v.shape[0]==antenna.shape[0]==concentration.shape[0]==400 and
         q.ndim==v.ndim==2 and antenna.shape==(400,2,3) and concentration.shape==(400,2),
         'Raw trace layout')
    need(np.max(abs(np.sum(q[:,3:7]**2,axis=1)-1))<1e-8,'Quaternion domain')
    calc_concentration=np.exp(-np.sum((antenna[:,:,:2]-specs['source_mm'])**2,axis=-1)/
                              (2*specs['sigma_mm']**2))
    need(float(np.max(abs(concentration-calc_concentration)))<=1e-12,
         'Recorded minus source differs from actual antenna pose')
    yaw_from_q=delta(yaw(q),yaw(original['prepared']))
    need(float(np.max(abs(yaw_from_q-finite(data['yaw_deg'],'yaw'))))<=1e-10,
         'Recorded yaw differs from quaternion')
    for key in ('contacts','upright','normal_N'):
        need(finite(data[key],key).shape==(400,), 'Support layout '+key)
    need(result['ms']==400 and result['contact_count_min']==int(np.min(data['contacts'])) and
         abs(result['upright_min']-float(np.min(data['upright'])))<=1e-12 and
         abs(result['final_yaw_deg']-float(data['yaw_deg'][-1]))<=1e-12,
         'Arm result not reconstructed')
    need(result['body_only'] is True and result['stage4_admission'] is False and
         result['stage5_admission'] is False,'False stage claim')
    return {'concentration_sup':float(np.max(abs(concentration-calc_concentration))),
            'yaw_sup_deg':float(np.max(abs(yaw_from_q-data['yaw_deg'])))}


def verify():
    start=time.monotonic()
    plan=read(HERE/'PLAN.json');post=read(HERE/'POSTCLOSE_PLAN.json')
    need(plan['schema']=='stage45_body_authority_screen_v1' and
         post['schema']=='stage45_body_authority_postclose_plan_v1','Plans')
    lock=read(HERE/'SOURCE_LOCK.json')
    need(all(digest(path)==sha for path,sha in lock.items()),'Frozen source hash mismatch')
    result=read(HERE/'RESULT.json')
    need(result['schema']=='stage45_body_authority_screen_result_v1' and
         result['source_lock_sha256']==digest(HERE/'SOURCE_LOCK.json') and
         result['plan_sha256']==digest(HERE/'PLAN.json') and
         result['code_sha256']==digest(HERE/'run_screen.py'),'Result/source identity')
    fields=read(FIELDS)['minus']
    source=np.asarray(fields['source_mm'],dtype=float)
    need(source.shape==(2,) and np.isfinite(source).all() and fields['sigma_mm']>0,'Source domain')
    with np.load(SOURCE,allow_pickle=False) as z:
        trial=np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==400 and np.array_equal(z['paso'][trial],np.arange(1,401)), 'Reference trial')
        original={'prepared':z['qpos'][trial[0]-1].copy(),
                  'qpos':z['qpos'][trial].copy(),'qvel':z['qvel'][trial].copy(),
                  'yaw':z['yaw_delta_deg'][trial].copy()}
    arms={};checks={}
    for name in ARMS:
        arm=read(HERE/name/'RESULT.json')
        need(arm['trace_sha256']==digest(HERE/name/'trace.npz'), 'Arm trace hash '+name)
        with np.load(HERE/name/'trace.npz',allow_pickle=False) as z:
            trace={k:z[k].copy() for k in z.files}
        checks[name]=verify_arrays(trace,original,fields,name,arm)
        arms[name]=(arm,trace)
    need(len({arm['initial_integration_sha256'] for arm,_ in arms.values()})==1,'Physical starts differ')
    donor=arms['donor'][1];positive=arms['positive_ceiling'][1];negative=arms['negative_ceiling'][1]
    parity={'qpos':float(np.max(abs(donor['qpos']-original['qpos']))),
            'qvel':float(np.max(abs(donor['qvel']-original['qvel']))),
            'yaw':float(np.max(abs(donor['yaw_deg']-original['yaw'])))}
    bounds=plan['criteria']
    need(parity['qpos']<=bounds['baseline_qpos_sup_native_max'] and
         parity['qvel']<=bounds['baseline_qvel_sup_native_max'] and
         parity['yaw']<=bounds['baseline_yaw_sup_deg_max'], 'Body donor parity')
    need(all(np.array_equal(donor[key][:100],arms[name][1][key][:100])
             for name in ('positive_ceiling','negative_ceiling') for key in ('qpos','qvel')),
         'Branched body histories differ before command intervention')
    lr=lambda t:t['minus_concentration'][:,0]-t['minus_concentration'][:,1]
    b=lr(donor);p=lr(positive);n=lr(negative)
    effects={'final_yaw_gap_extremes_deg':float(abs(positive['yaw_deg'][-1]-negative['yaw_deg'][-1])),
             'late_mean_abs_signed_LR_gap_extremes':float(np.mean(abs(p[200:400]-n[200:400]))),
             'max_signed_LR_gap_donor_vs_either':float(max(np.max(abs(b[100:]-p[100:])),
                                                        np.max(abs(b[100:]-n[100:])))),
             'final_bearing_error_deg':{name:float(abs(bearing(trace['qpos'],source)[-1]))
                                        for name,(_,trace) in arms.items()},
             'bearing_error_at_switch_deg':float(abs(bearing(donor['qpos'],source)[99])),
             'final_minus_LR':{name:float(lr(trace)[-1]) for name,(_,trace) in arms.items()}}
    recorded=result['effects']
    for key,value in effects.items():
        if isinstance(value,dict):
            for sub,number in value.items():
                need(abs(number-recorded[key][sub])<=1e-10,'Effect mismatch '+key+'/'+sub)
        else:
            need(abs(value-recorded[key])<=1e-10,'Effect mismatch '+key)
    recomputed={'donor_parity':True,'same_initial_state':True,
       'extreme_yaw_gap':effects['final_yaw_gap_extremes_deg']>=bounds['extreme_pair_final_yaw_gap_deg_min'],
       'extreme_field_gap':effects['late_mean_abs_signed_LR_gap_extremes']>=bounds['extreme_pair_late_mean_abs_signed_LR_gap_min'],
       'baseline_field_gap':effects['max_signed_LR_gap_donor_vs_either']>=bounds['baseline_vs_extreme_max_signed_LR_gap_min'],
       'contacts':min(int(np.min(trace['contacts'])) for _,trace in arms.values())>=bounds['contact_count_each_ms_min'],
       'upright':min(float(np.min(trace['upright'])) for _,trace in arms.values())>=bounds['upright_each_ms_min'],
       'budget':result['wall_total_s']<=plan['budget']['wall_total_s_max']}
    need(recomputed==result['checks'],'Runner checks differ from raw reconstruction')
    need(result['classification']==('PHYSICAL_ENVELOPE_MATERIAL' if all(recomputed.values())
                                    else 'ENVELOPE_NOT_ESTABLISHED'),'Classification mismatch')
    need(result['body_only'] is True and result['organism_runs']==0 and
         result['stage4_admission'] is False and result['stage5_admission'] is False,
         'False scientific admission')
    # Deliberate in-memory corruption; source traces remain immutable.
    damaged={key:value.copy() for key,value in donor.items()}
    damaged['antennae_mm'][201,0,0]+=.01
    try:verify_arrays(damaged,original,fields,'donor',arms['donor'][0])
    except ValueError:antenna_rejected=True
    else:antenna_rejected=False
    damaged={key:value.copy() for key,value in original.items()}
    damaged['qpos'][201,0]+=.01
    donor_parity_rejected=float(np.max(abs(donor['qpos']-damaged['qpos'])))>bounds['baseline_qpos_sup_native_max']
    bad_result=dict(arms['donor'][0]);bad_result['status']='FAILED'
    try:verify_arrays(donor,original,fields,'donor',bad_result)
    except ValueError:status_rejected=True
    else:status_rejected=False
    need(antenna_rejected and donor_parity_rejected and status_rejected,'Corruption test failed')
    elapsed=time.monotonic()-start
    need(elapsed<=post['budget']['wall_s_max_each'],'Verifier wall budget')
    return {'schema':'stage45_body_authority_postclose_v1',
            'classification':'VERIFIED_BODY_ONLY','organism_runs':0,
            'stage4_admission':False,'stage5_admission':False,
            'arms':checks,'parity':parity,'effects':effects,'checks':recomputed,
            'corruption_rejected':{'antenna':antenna_rejected,'donor_qpos':donor_parity_rejected,
                                   'status':status_rejected},
            'source_lock_sha256':digest(HERE/'SOURCE_LOCK.json'),
            'plan_sha256':digest(HERE/'PLAN.json'),
            'postclose_plan_sha256':digest(HERE/'POSTCLOSE_PLAN.json'),
            'verifier_sha256':digest(Path(__file__)),
            'wall_s':elapsed,
            'limits':'No per-ms commanded forces recorded in body-only traces; old code and source hash remain part of evidence.'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    receipt=verify()
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'classification':receipt['classification'],
                      'effects':receipt['effects'],'corruption_rejected':receipt['corruption_rejected']},
                     allow_nan=False))
