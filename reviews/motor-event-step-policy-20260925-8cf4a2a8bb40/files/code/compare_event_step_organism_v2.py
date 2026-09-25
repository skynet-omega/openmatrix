"""Verify the prospective one-ms paired controller experiment from raw outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_state(stem):
    descriptor = json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'__array__'}:
                    return arrays[value['__array__']].copy()
                return {k: decode(v) for k,v in value.items()}
            if isinstance(value,list):
                return [decode(x) for x in value]
            return value
        return decode(descriptor)


def run(base, candidate, output, corrupt=False):
    plan_path = HERE/'EVENT_STEP_PLAN_42.json'
    plan = json.loads(plan_path.read_text())
    runs = []
    for mode, folder in (('baseline',base),('candidate',candidate)):
        receipt = json.loads((folder/'EVENT_STEP_RECEIPT.json').read_text())
        result_path = folder/'RESULT.json'
        result = json.loads(result_path.read_text())
        need(receipt['mode'] == mode and receipt['plan_sha256'] == sha(plan_path), 'Mode/plan')
        need(receipt['run_result_sha256'] == sha(result_path), 'Result hash')
        need(receipt['binary_sha256'] == plan['binary_sha256'][mode], 'Binary hash')
        need(receipt['status'] == result['status'] == 'COMPLETE' and receipt['budget_ok'], 'Run status/budget')
        need(result['completed_trial_ms'] == 1 and result['completed_preparation_ms'] == 0,
             'Run duration')
        need(result['runtime']['event_boundaries'] is True and result['runtime']['profile'] == 'causal_cuda',
             'Runtime')
        state = read_state(folder/'final_state/session')
        events = json.loads((folder/'EVENT_AUDIT.json').read_text())
        with np.load(folder/'traces.npz', allow_pickle=False) as z:
            trace = {k:z[k].copy() for k in z.files}
        runs.append((receipt,result,state,events,trace))
    (br,bres,bs,be,bt),(cr,cres,cs,ce,ct) = runs
    need(br['preparation_weights_sha256'] == cr['preparation_weights_sha256'], 'Preparation mismatch')
    g = plan['gates']
    by,cy = np.asarray(bs['hybrid']['state']),np.asarray(cs['hybrid']['state'])
    if corrupt:
        cy = cy.copy();cy[0] = np.nan
    need(by.shape == cy.shape == (359373,) and np.isfinite(by).all() and np.isfinite(cy).all(),
         'State shape/nonfinite')
    p = bs['hybrid']['parameters']
    need(p['rtol'] == cs['hybrid']['parameters']['rtol'] and
         p['atol'] == cs['hybrid']['parameters']['atol'], 'Tolerance changed')
    normalized = np.abs(by-cy)/(p['atol']+p['rtol']*np.maximum(np.abs(by),np.abs(cy)))
    emax = float(np.max(normalized))
    need(len(be['blocks']) == len(ce['blocks']) == 16, 'Block schedule changed')
    max_time = max_jump = max_post = 0.
    event_count = 0
    for b,c in zip(be['blocks'],ce['blocks']):
        need(b['duration_ns'] == c['duration_ns'] and len(b['events']) == len(c['events']),
             'Event count/duration')
        for x,y in zip(b['events'],c['events']):
            need(x['row'] == y['row'] and x['producer'] == y['producer'], 'Event identity/order')
            max_time = max(max_time,abs(x['time_s']-y['time_s']))
            max_jump = max(max_jump,abs(x['jump']-y['jump']))
            if x['post_q'] is not None and y['post_q'] is not None:
                max_post = max(max_post,abs(x['post_q']-y['post_q']))
            else:
                need(x['post_q'] == y['post_q'], 'Event post_q null mismatch')
            event_count += 1
    yaw_diff = float(abs(bt['yaw_delta_deg'][-1]-ct['yaw_delta_deg'][-1]))
    qpos_diff = float(np.max(np.abs(bt['qpos']-ct['qpos'])))
    accepted_base = int(bres['runtime']['CNS']['accepted'])
    accepted_candidate = int(cres['runtime']['CNS']['accepted'])
    rejected_candidate = int(cres['runtime']['CNS']['rejected'])
    report = {'schema':'event_step_pair_result_v1','plan_sha256':sha(plan_path),
              'baseline_result_sha256':sha(base/'RESULT.json'),
              'candidate_result_sha256':sha(candidate/'RESULT.json'),
              'preparation_weights_sha256':br['preparation_weights_sha256'],
              'baseline_accepted_trials':accepted_base,'candidate_accepted_trials':accepted_candidate,
              'baseline_rejected_trials':int(bres['runtime']['CNS']['rejected']),
              'candidate_rejected_trials':rejected_candidate,
              'trial_fraction':accepted_candidate/accepted_base,
              'baseline_step_wall_s':br['step_wall_s'],
              'candidate_step_wall_s':cr['step_wall_s'],
              'step_wall_fraction':cr['step_wall_s']/br['step_wall_s'],
              'normalized_state_error_max':emax,
              'normalized_state_error_p99':float(np.quantile(normalized,.99)),
              'raw_state_error_max':float(np.max(np.abs(by-cy))),
              'yaw_delta_abs_deg':yaw_diff,'qpos_max_abs':qpos_diff,
              'event_count':event_count,'event_time_max_abs_s':max_time,
              'event_jump_max_abs':max_jump,'event_post_q_max_abs':max_post,
              'baseline_next_step_ns':int(bs['hybrid']['next_step_ns']),
              'candidate_next_step_ns':int(cs['hybrid']['next_step_ns'])}
    safety = (emax <= g['final_state_normalized_error_max'] and
              yaw_diff <= g['yaw_delta_abs_deg_max'] and
              max_time <= g['event_time_abs_s_max'] and
              max_jump <= g['event_jump_abs_max'] and
              max_post <= g['event_post_q_abs_max'] and
              rejected_candidate <= g['candidate_rejected_max'])
    speed = (report['trial_fraction'] <= g['candidate_accepted_max_fraction_of_baseline'] and
             report['step_wall_fraction'] <= g['candidate_step_wall_max_fraction_of_baseline'])
    report['safety_gate_pass'] = bool(safety)
    report['speed_gate_pass'] = bool(speed)
    report['status'] = 'PASS_NECESSARY_ONE_MS_PREFLIGHT' if safety and speed else 'FAIL_PROSPECTIVE_GATE'
    report['scope'] = ('One sham1ms organism pair, same preparation and frozen equations, '
                       'does not validate long-horizon speed or odor behavior.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--corrupt',action='store_true')
    a = ap.parse_args()
    report = run(a.baseline,a.candidate,a.out,a.corrupt)
    print(json.dumps({k:report[k] for k in ('status','trial_fraction','step_wall_fraction',
                                           'normalized_state_error_max','event_count',
                                           'safety_gate_pass','speed_gate_pass')}))
