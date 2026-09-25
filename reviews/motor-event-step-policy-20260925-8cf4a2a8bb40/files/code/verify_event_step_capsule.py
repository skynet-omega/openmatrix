"""Recompute the portable 20-ms A/B numerical and event result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def near(a,b,atol=1e-12):
    return np.isfinite(a) and np.isfinite(b) and abs(a-b)<=atol


def verify(root,corrupt=False):
    root = Path(root)
    plan_path = root/'plans/EVENT_STEP_PLAN_43.json'
    pair = json.loads((root/'results/EVENT_STEP_PAIR_LONG_RESULT.json').read_text())
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((root/'data/EVENT_STEP_CAPSULE_MANIFEST.json').read_text())
    capsule_path = root/'data/EVENT_STEP_LONG_CAPSULE.npz'
    need(pair['plan_sha256']==sha(plan_path),'Plan hash')
    need(manifest['capsule_sha256']==sha(capsule_path),'Capsule hash')
    result = {};audit={};trace={};receipt={}
    for mode in ('baseline','candidate'):
        directory=root/'results'/mode
        result[mode]=json.loads((directory/'RESULT.json').read_text())
        audit[mode]=json.loads((directory/'EVENT_AUDIT.json').read_text())
        receipt[mode]=json.loads((directory/'EVENT_STEP_RECEIPT.json').read_text())
        with np.load(directory/'traces.npz',allow_pickle=False) as z:
            trace[mode]={k:z[k].copy() for k in z.files}
        for rel in ('RESULT.json','EVENT_STEP_RECEIPT.json','EVENT_AUDIT.json','traces.npz'):
            need(manifest['source_hashes'][mode][rel]==sha(directory/rel),
                 'Published input hash: '+mode+'/'+rel)
        need(result[mode]['status']==receipt[mode]['status']=='COMPLETE', 'Organism status')
        need(result[mode]['completed_trial_ms']==20, 'Duration')
        need(receipt[mode]['plan_sha256']==sha(plan_path), 'Receipt plan')
        need(receipt[mode]['preparation_weights_sha256']==pair['preparation_weights_sha256'],
             'Preparation')
    with np.load(capsule_path,allow_pickle=False) as z:
        s=np.asarray(z['baseline_state']);t=np.asarray(z['candidate_state'])
        if corrupt:
            t=t.copy();t[0]=np.nan
        need(s.shape==t.shape==(359373,) and np.isfinite(s).all() and np.isfinite(t).all(),
             'Corrupt/nonfinite state')
        atol=float(z['baseline_atol']);rtol=float(z['baseline_rtol'])
        need(atol==float(z['candidate_atol']) and rtol==float(z['candidate_rtol']),
             'Tolerance parity')
        normalized=np.abs(s-t)/(atol+rtol*np.maximum(np.abs(s),np.abs(t)))
        for field in ('pending_excitation','pending_sensors','muscles_activation','plasticity_factors'):
            need(np.isfinite(z['baseline_'+field]).all() and np.isfinite(z['candidate_'+field]).all(),
                 'Nonfinite owner: '+field)
    need(near(float(normalized.max()),pair['normalized_state_error_max']), 'State error')
    need(near(float(np.quantile(normalized,.99)),pair['normalized_state_error_p99']), 'State quantile')
    need(near(float(np.max(np.abs(s-t))),pair['raw_state_error_max']), 'Raw state error')
    b=trace['baseline'];c=trace['candidate']
    need(set(b)==set(c) and len(b['yaw_delta_deg'])==len(c['yaw_delta_deg'])==20,
         'Trace layout')
    need(near(float(abs(b['yaw_delta_deg'][-1]-c['yaw_delta_deg'][-1])),
              pair['yaw_delta_abs_deg']), 'Yaw')
    need(near(float(np.max(np.abs(b['qpos']-c['qpos']))),pair['qpos_max_abs']), 'Body qpos')
    be=audit['baseline']['blocks'];ce=audit['candidate']['blocks']
    need(len(be)==len(ce)==320,'Epoch count')
    count=0;time_diff=jump_diff=post_diff=0.
    for x,y in zip(be,ce):
        need(x['duration_ns']==y['duration_ns'] and len(x['events'])==len(y['events']),
             'Event block mismatch')
        for u,v in zip(x['events'],y['events']):
            need(u['row']==v['row'] and u['producer']==v['producer'], 'Event identity/order')
            time_diff=max(time_diff,abs(u['time_s']-v['time_s']))
            jump_diff=max(jump_diff,abs(u['jump']-v['jump']))
            if u['post_q'] is not None and v['post_q'] is not None:
                post_diff=max(post_diff,abs(u['post_q']-v['post_q']))
            else:
                need(u['post_q']==v['post_q'],'Event post_q')
            count+=1
    for value,key in ((count,'event_count'),(time_diff,'event_time_max_abs_s'),
                      (jump_diff,'event_jump_max_abs'),(post_diff,'event_post_q_max_abs')):
        need(near(value,pair[key]),key)
    trials={k:int(result[k]['runtime']['CNS']['accepted']) for k in result}
    for k in trials:
        need(trials[k]==pair[k+'_accepted_trials'],'Trial count '+k)
    need(near(receipt['candidate']['step_wall_s']/receipt['baseline']['step_wall_s'],
              pair['step_wall_fraction']), 'Wall ratio')
    gates=plan['gates']
    safety=(float(normalized.max())<=gates['final_state_normalized_error_max'] and
            time_diff<=gates['event_time_abs_s_max'] and
            jump_diff<=gates['event_jump_abs_max'] and
            post_diff<=gates['event_post_q_abs_max'] and
            pair['yaw_delta_abs_deg']<=gates['yaw_delta_abs_deg_max'] and
            result['candidate']['runtime']['CNS']['rejected']<=gates['candidate_rejected_max'])
    speed=(trials['candidate']/trials['baseline']<=
           gates['candidate_accepted_max_fraction_of_baseline'] and
           receipt['candidate']['step_wall_s']/receipt['baseline']['step_wall_s']<=
           gates['candidate_step_wall_max_fraction_of_baseline'])
    need(safety==pair['safety_gate_pass'] and speed==pair['speed_gate_pass'],'Gate mismatch')
    return {'status':'PASS_PORTABLE_RECOMPUTATION','state_error_max':float(normalized.max()),
            'events':count,'accepted_baseline':trials['baseline'],
            'accepted_candidate':trials['candidate'],'safety':safety,'speed':speed}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--corrupt',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.root,a.corrupt)))
