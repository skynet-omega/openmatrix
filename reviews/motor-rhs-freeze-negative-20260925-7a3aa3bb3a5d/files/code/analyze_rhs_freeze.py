"""Retrospective necessary screen for a zero-order full-RHS freeze, not an integrator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def analyze(out):
    start=time.monotonic();p=json.loads((HERE/'RHS_FREEZE_PLAN_44.json').read_text())
    need(p['schema']=='rhs_freeze_plan_v1' and sha(Path(__file__))==p['script_sha256'],
         'Plan/source changed')
    full=HERE/'quantized_full_01/FULL_COEFFICIENT_ARRAYS.npz'
    audit=HERE/'quantized_full_01/EVENT_AUDIT.json'
    need(sha(full)==p['full_arrays_sha256'] and sha(audit)==p['audit_sha256'],
         'Inputs changed')
    with np.load(full,allow_pickle=False) as z:
        t=z['query_s'];h=z['trial_step_s'];x=z['consumed_z']
        a=z['consumed_target'];r=z['consumed_rate']
    need(x.shape==a.shape==r.shape==(60,359373) and t.shape==h.shape==(60,),
         'Array shape')
    need(np.isfinite(x).all() and np.isfinite(a).all() and np.isfinite(r).all() and
         np.isfinite(t).all() and np.all(h>0),'Nonfinite data')
    ev=sorted(set(v['time_s'] for v in json.loads(audit.read_text())['blocks'][1]['events']))
    need(len(ev)==7 and np.max(t)<.000125,'Event schedule')
    segment=np.searchsorted(ev,t,side='right')
    anchors={int(s):int(np.flatnonzero(segment==s)[0]) for s in np.unique(segment)}
    rhs=r*(a-x)
    del a,r
    result={'schema':'rhs_freeze_result_v1','plan_sha256':sha(HERE/'RHS_FREEZE_PLAN_44.json'),
            'status':'DIAGNOSTIC_ONLY','queries':60,'segments_with_queries':len(anchors),
            'event_boundaries':len(ev),'anchor_query_by_segment':anchors,
            'scope':'Observed full RHS queries only; zero-order freeze proxy, no candidate trajectory, no speed claim.'}
    for label,indices in [('whole_block',np.zeros(60,dtype=np.int64)),
                          ('one_per_event_interval',np.array([anchors[int(s)] for s in segment]))]:
        maxima=np.empty(60,dtype=np.float64)
        worst=np.empty(60,dtype=np.int64)
        for i,j in enumerate(indices):
            scale=p['atol']+p['rtol']*np.maximum(np.abs(x[i]),np.abs(x[j]))
            score=h[i]*np.abs(rhs[i]-rhs[j])/(3.*scale)
            worst[i]=int(np.argmax(score));maxima[i]=float(score[worst[i]])
        result[label]={'max':float(np.max(maxima)),
                       'median':float(np.median(maxima)),
                       'p90':float(np.quantile(maxima,.9)),
                       'queries_over_0_1':int(np.count_nonzero(maxima>.1)),
                       'queries_over_1':int(np.count_nonzero(maxima>1.)),
                       'worst_query':int(np.argmax(maxima)),
                       'worst_state_index':int(worst[np.argmax(maxima)])}
    result['wall_s']=time.monotonic()-start
    result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    need(result['wall_s']<=p['budget']['wall_s_max'] and
         result['maxrss_kib']<=p['budget']['ram_gib_max']*1024**2,'Budget')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need(out.stat().st_size<=p['budget']['output_bytes_max'],'Output budget')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    v=analyze(args.out)
    print(json.dumps({k:v[k] for k in ('status','queries','segments_with_queries',
                                       'whole_block','one_per_event_interval','wall_s')}))
