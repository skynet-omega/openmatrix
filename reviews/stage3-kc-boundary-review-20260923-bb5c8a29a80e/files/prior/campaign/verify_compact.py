"""Check the publishable selected arrays/events without importing the organism."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CASES=('native_sham_20_01','reference_sham_20_01')
FULL_NS=125000


def require(cond,msg):
    if not cond:raise ValueError(msg)


def accepted(case):
    log=json.loads((HERE/case/'EVENT_AUDIT.json').read_text())
    series=defaultdict(list);predictor=committed=0
    for block in log['blocks']:
        if block['duration_ns']==FULL_NS//2:
            predictor+=len(block['events']);continue
        require(block['duration_ns']==FULL_NS,'Unknown event block duration')
        committed+=len(block['events'])
        for e in block['events']:
            key=(e['producer'],e['row'],e['neuron_id'])
            series[key].append((block['start_elapsed_ns']+round(e['time_s']*1e9),e['post_q']))
    return {k:sorted(v) for k,v in series.items()},predictor,committed


def main():
    target=json.loads((HERE/'GATE.json').read_text())
    digest=hashlib.sha256((HERE/'COMPACT_NUMERIC.npz').read_bytes()).hexdigest()
    require(digest==target['compact_sha256'],'Compact arrays hash mismatch')
    with np.load(HERE/'COMPACT_NUMERIC.npz',allow_pickle=False) as z:
        ids=z['ids'];x=z['native_target'];y=z['reference_target']
        require(tuple(ids)==(10176,10208,10360,523769,10065,10118),'Identity mismatch')
        require(x.shape==y.shape==(20,6),'Flow shape mismatch')
        require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite flow')
        require(np.array_equal(x[:,2:4],np.zeros((20,2)))==target['dna02_both_target_zero_all_samples'],'DNa02 target mismatch')
        for i,identity in enumerate(ids):
            t=target['per_id'][str(int(identity))]
            require(float(np.max(np.abs(x[:,i]-y[:,i])))==t['target_max_abs_between_engines'],'Target difference mismatch')
            require(float(np.max(np.abs(z['native_raw'][:,i]-z['reference_raw'][:,i])))==t['raw_max_abs_between_engines'],'Raw difference mismatch')
        for path,value in target['kc_errors'].items():
            key=path.replace('/','__')
            a,b=z[key+'__native'],z[key+'__reference']
            require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid KC array')
            d=np.abs(a-b);idx=np.unravel_index(np.argmax(d),d.shape)
            require(list(map(int,idx))==value['index'] and float(d[idx])==value['max_abs'],'KC difference mismatch')
        require(float(x[-1,5]-x[-1,4])==target['dnb05_left_minus_right_target_last'],'DNb05 target mismatch')
        require(float(z['native_dn_q'][-1,2]-z['native_dn_q'][-1,3])==target['dnb05_left_minus_right_actual_q_last'],'DNb05 q mismatch')
    a,pa,na=accepted(CASES[0]);b,pb,nb=accepted(CASES[1])
    frozen=target['accepted_events']
    require((pa,pb,na,nb)==tuple(frozen[k] for k in ('predictor_events_causal','predictor_events_reference','committed_events_causal','committed_events_reference')),'Event totals mismatch')
    require(set(a)==set(b) and all(len(a[k])==len(b[k]) for k in a),'Accepted event identities/counts differ')
    times=[abs(v[0]-w[0]) for k in a for v,w in zip(a[k],b[k])]
    require(max(times)==frozen['maximum_event_time_abs_ns'],'Event time mismatch')
    print(json.dumps({'selected_arrays_verified':True,'accepted_events_each':na,
                      'max_accepted_time_ns':max(times),'numerical_gate':target['numerical_gate'],
                      'scope':'Compact focal/KC subset and logged accepted events only; no full snapshot equivalence verification.'}))


if __name__=='__main__':main()
