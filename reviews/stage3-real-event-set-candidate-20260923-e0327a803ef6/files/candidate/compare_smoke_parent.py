"""Short paired numerical screen against the unchanged protected sham."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_flujo_kernel_20260923_04/smoke_off_01'
CANDIDATE=HERE/'smoke_off_01'
LIMIT=1e-4


def read_array(folder,*keys):
    root=folder/'final_state/session'
    d=json.loads(root.with_suffix('.json').read_text())
    for key in keys:d=d[key]
    with np.load(root.with_suffix('.npz'),allow_pickle=False) as z:return z[d['__array__']].copy()


def main():
    a=json.loads((PARENT/'RESULT.json').read_text())
    b=json.loads((CANDIDATE/'RESULT.json').read_text())
    if any(v['status']!='COMPLETE' or v['completed_trial_ms']!=1 for v in (a,b)):
        raise ValueError('Incomplete one-ms control')
    original=read_array(PARENT,'hybrid','state')
    new=read_array(CANDIDATE,'hybrid','state')
    if original.shape!=new.shape or not np.isfinite(new).all():raise ValueError('Invalid hybrid state')
    state_max=float(np.max(np.abs(original-new)))
    exact={str(path):np.array_equal(read_array(PARENT,*path),read_array(CANDIDATE,*path))
           for path in [('body','integration'),('pending_sensors',)]}
    traces={};changed_discrete=[]
    with np.load(PARENT/'traces.npz',allow_pickle=False) as x, np.load(CANDIDATE/'traces.npz',allow_pickle=False) as y:
        if set(x.files)!=set(y.files):raise ValueError('Trace fields changed')
        for key in x.files:
            u,v=x[key],y[key]
            if u.shape!=v.shape or u.dtype!=v.dtype:raise ValueError('Trace layout: '+key)
            if u.dtype.kind in 'fc':
                traces[key]=float(np.max(abs(u-v),initial=0.))
            elif not np.array_equal(u,v):changed_discrete.append(key)
    old_event=a['runtime']['events'];new_event=b['runtime']['events']
    event_exact={key:old_event[key]==new_event[key] for key in ('blocks','events','global_trials')}
    old_desc=json.loads((PARENT/'final_state/session.json').read_text())
    new_desc=json.loads((CANDIDATE/'final_state/session.json').read_text())
    clock_exact=all(old_desc[k]==new_desc[k] for k in ('time_ns','ticks')) and all(
        old_desc['hybrid'][k]==new_desc['hybrid'][k] for k in ('time_ns','next_step_ns'))
    passed=(state_max<=LIMIT and all(exact.values()) and all(v<=LIMIT for v in traces.values())
            and not changed_discrete and all(event_exact.values()) and clock_exact)
    result={'schema':'stage3_set_one_ms_parent_screen_v1','limit_continuous_abs':LIMIT,
            'screen_pass':bool(passed),'hybrid_state_max_abs':state_max,
            'exact_body_and_pending':exact,'event_counts_exact':event_exact,
            'trace_max_abs':traces,'changed_discrete_fields':changed_discrete,'clock_exact':clock_exact,
            'scope':'One-ms matched-sham numerical screen only; mixed trace units reported separately, no 130/400-ms guarantee.'}
    (HERE/'SMOKE_PARENT_COMPARE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'screen_pass':passed,'hybrid_state_max_abs':state_max,
                      'max_trace_abs':max(traces.values()),'changed_discrete':changed_discrete,
                      'event_counts_exact':event_exact},ensure_ascii=False))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
