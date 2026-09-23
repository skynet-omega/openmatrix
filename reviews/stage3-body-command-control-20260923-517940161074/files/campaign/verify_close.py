"""Reconstruct the C body-control verdict from portable saved arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES=('sham_replay','zero','plus','minus')
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(test,message):
    if not test:raise ValueError(message)
def close(a,b,tol=1e-12):return abs(a-b)<=tol
def yaw_deg(q):
    w,x,y,z=np.asarray(q)[...,3:7].T
    return np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def decode_state(root):
    descriptor=json.loads((root/'portable_body.json').read_text())
    with np.load(root/'portable_body.npz',allow_pickle=False) as arrays:
        def rec(x):
            if isinstance(x,dict):
                if set(x)=={'__array__'}:return arrays[x['__array__']].copy()
                return {k:rec(v) for k,v in x.items()}
            if isinstance(x,list):return [rec(v) for v in x]
            return x
        return rec(descriptor)

def verify(root:Path):
    plan=json.loads((root/'PLAN.json').read_text())
    reported=json.loads((root/'CLOSE.json').read_text())
    source=reported['source']
    need(sha(root/'PLAN.json')==source['plan_sha256'],'Plan source changed')
    need(sha(root/'body_replay.py')==source['body_replay_source_sha256'],'Executed replay source changed')
    need(sha(root/'reference_sham_traces.npz')==source['source_trace_sha256'],'Reference sham changed')
    state=decode_state(root)
    need(state['body']['dt_ns']==25000 and state['controller']['state']['active'],
         'Wrong portable physical model/activation')
    initial=hashlib.sha256(np.ascontiguousarray(state['body']['integration']).tobytes()).hexdigest()
    with np.load(root/'reference_sham_traces.npz',allow_pickle=False) as z:
        mask=z['fase']=='ensayo'
        ref={k:z[k][mask].copy() for k in ('qpos','qvel','yaw_delta_deg',
                                           'command_forward_mm_s','command_yaw_rate_rad_s')}
        origin=float(yaw_deg(z['qpos'][39]))
    need(len(ref['qpos'])==400,'Wrong reference horizon')
    actual={};maxpos=maxvel=maxyaw=0.
    for name in NAMES:
        folder=root/('body_'+name+'_02')
        receipt=json.loads((folder/'RESULT.json').read_text())
        with np.load(folder/'trace.npz',allow_pickle=False) as z:
            t={k:z[k].copy() for k in z.files}
        need(receipt==reported['cases'][name],'Run receipt differs from close')
        need(receipt['initial_integration_sha256']==initial,'Body start differs')
        need(receipt['ms']==400 and receipt['dt_us']==25 and receipt['status']=='COMPLETE',
             'Physical run incomplete')
        need(t['qpos'].shape==(400,109) and t['qvel'].shape[0]==400,'Body trace shape')
        need(all(np.isfinite(t[k]).all() for k in t),'Nonfinite body trace')
        yaw=np.asarray([(float(yaw_deg(q))-origin+180)%360-180 for q in t['qpos']])
        need(np.max(np.abs(yaw-t['yaw_delta_deg']))<=1e-10,'Yaw not reconstructed from pose')
        need(close(float(yaw[-1]),receipt['final_yaw_deg'],1e-10),'Reported yaw differs')
        command=np.rad2deg(np.sum(t['command_yaw_rate_rad_s'])*.001)
        need(close(float(command),receipt['command_integral_deg']),'Reported command differs')
        need(np.array_equal(t['command_forward_mm_s'],np.full(400,.2)),'Forward command differs')
        if name=='sham_replay':
            maxpos=float(np.max(np.abs(t['qpos']-ref['qpos'])))
            maxvel=float(np.max(np.abs(t['qvel']-ref['qvel'])))
            maxyaw=float(np.max(np.abs(t['yaw_delta_deg']-ref['yaw_delta_deg'])))
            need(np.array_equal(t['command_yaw_rate_rad_s'],ref['command_yaw_rate_rad_s']),
                 'Sham replay command differs')
            need(maxpos<=1e-8 and maxvel<=1e-8 and maxyaw<=1e-6,
                 'Sham physical replay fails frozen gate')
        else:
            rad={'zero':0.,'plus':np.deg2rad(.2),'minus':-np.deg2rad(.2)}[name]
            need(np.array_equal(t['command_yaw_rate_rad_s'],np.full(400,rad)),
                 'Open-loop intervention command differs')
        actual[name]=float(yaw[-1])
    sums={'zero_yaw_deg':actual['zero'],
          'plus_minus_zero_deg':actual['plus']-actual['zero'],
          'minus_minus_zero_deg':actual['minus']-actual['zero'],
          'mirror_sum_deg':actual['plus']+actual['minus']-2*actual['zero'],
          'sham_replay_minus_zero_deg':actual['sham_replay']-actual['zero']}
    for key,value in sums.items():need(close(value,reported['open_loop'][key],1e-10),key+' differs')
    need(reported['wall_all_s']<=plan['budgets']['body_replay_wall_total_s_max'],
         'Budget exceeded')
    out={'schema':'body_only_reconstructed_close_v1','stage3_admission':False,
         'classification':'C_BODY_MECHANICAL_BIAS_WEAKENED',
         'replay_qpos_max_abs':maxpos,'replay_qvel_max_abs':maxvel,
         'replay_yaw_max_abs_deg':maxyaw,'recomputed':sums,
         'interpretation':'For this frozen state and synthetic contact controller, positive sham yaw is primarily commanded; baseline neuronal/decoder source remains unvalidated. No biological handedness or stage-3 admission.',
         'budget_wall_s':reported['wall_all_s']}
    (root/'VERIFIED_CLOSE.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=HERE)
    args=parser.parse_args();print(json.dumps(verify(args.root.resolve()),allow_nan=False))
