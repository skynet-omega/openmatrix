"""Rebuild a 20-ms numerical comparison from saved arrays and traces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
A=HERE/'native_sham_20_01'
B=HERE/'reference_sham_20_01'
LIMIT=1e-4


def array_paths(node,prefix=()):
    if isinstance(node,dict):
        if set(node)=={'__array__'}:
            yield '/'.join(map(str,prefix)),node['__array__']
        else:
            for key,value in node.items():yield from array_paths(value,prefix+(key,))
    elif isinstance(node,list):
        for index,value in enumerate(node):yield from array_paths(value,prefix+(index,))


def main():
    ra=json.loads((A/'RESULT.json').read_text());rb=json.loads((B/'RESULT.json').read_text())
    result={'schema':'stage3_dnb05_reference_20ms_comparison_v1',
            'causal_status':ra['status'],'reference_status':rb['status'],
            'causal_completed_ms':ra['completed_trial_ms'],
            'reference_completed_ms':rb['completed_trial_ms'],
            'continuous_abs_limit':LIMIT,'reference_failure':rb['error']}
    if ra['status']!='COMPLETE' or rb['status']!='COMPLETE' or ra['completed_trial_ms']!=20 or rb['completed_trial_ms']!=20:
        result['screen_pass']=False;result['reason']='A numerical comparator arm did not finish 20 ms.'
        (HERE/'REFERENCE_20_COMPARE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('screen_pass','reason','causal_completed_ms','reference_completed_ms')}))
        return 2
    ja=json.loads((A/'final_state/session.json').read_text())
    jb=json.loads((B/'final_state/session.json').read_text())
    pa=dict(array_paths(ja));pb=dict(array_paths(jb))
    if set(pa)!=set(pb):raise ValueError('Scientific snapshot array paths differ')
    maxima={};changed_exact=[];nonfinite=[]
    with np.load(A/'final_state/session.npz',allow_pickle=False) as za, np.load(B/'final_state/session.npz',allow_pickle=False) as zb:
        for path in sorted(pa):
            x,y=za[pa[path]],zb[pb[path]]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Snapshot array layout changed: '+path)
            if x.dtype.kind in 'fc':
                if not np.isfinite(x).all() or not np.isfinite(y).all():nonfinite.append(path)
                else:maxima[path]=float(np.max(np.abs(x-y),initial=0.))
            elif not np.array_equal(x,y):changed_exact.append(path)
    traces={};changed_trace=[]
    with np.load(A/'traces.npz',allow_pickle=False) as za, np.load(B/'traces.npz',allow_pickle=False) as zb:
        if set(za.files)!=set(zb.files):raise ValueError('Trace schemas differ')
        for key in za.files:
            x,y=za[key],zb[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Trace layout changed: '+key)
            if x.dtype.kind in 'fc':
                if not np.isfinite(x).all() or not np.isfinite(y).all():nonfinite.append('trace/'+key)
                else:traces[key]=float(np.max(np.abs(x-y),initial=0.))
            elif not np.array_equal(x,y):changed_trace.append(key)
    flow={};flow_discrete=[]
    with np.load(A/'flow/FLOW.npz',allow_pickle=False) as za, np.load(B/'flow/FLOW.npz',allow_pickle=False) as zb:
        if set(za.files)!=set(zb.files):raise ValueError('Native-flow schemas differ')
        for key in za.files:
            x,y=za[key],zb[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Native-flow layout changed: '+key)
            if x.dtype.kind in 'fc':
                if not np.isfinite(x).all() or not np.isfinite(y).all():nonfinite.append('flow/'+key)
                else:flow[key]=float(np.max(np.abs(x-y),initial=0.))
            elif not np.array_equal(x,y):flow_discrete.append(key)
    clock_exact=all(ja[k]==jb[k] for k in ('time_ns','ticks')) and ja['hybrid']['time_ns']==jb['hybrid']['time_ns']
    events_exact={k:ra['runtime']['events'][k]==rb['runtime']['events'][k] for k in ('blocks','events')}
    # Flow raw inputs have different units and are reported, not screened with
    # the saved-state/trace absolute limit. Do not add a threshold post hoc.
    passed=(clock_exact and all(events_exact.values()) and not nonfinite and not changed_exact and not changed_trace
            and not flow_discrete and all(x<=LIMIT for x in maxima.values())
            and all(x<=LIMIT for x in traces.values()))
    result.update({'screen_pass':bool(passed),'snapshot_arrays_compared':len(pa),
                   'snapshot_max_abs':max(maxima.values(),default=0.),
                   'snapshot_worst_path':max(maxima,key=maxima.get) if maxima else None,
                   'snapshot_top_abs':dict(sorted(maxima.items(),key=lambda kv:kv[1],reverse=True)[:30]),
                   'trace_max_abs':traces,'flow_max_abs':flow,'changed_discrete_flow_fields':flow_discrete,
                   'changed_exact_snapshot_paths':changed_exact,
                   'changed_discrete_traces':changed_trace,'nonfinite_paths':nonfinite,
                   'clock_exact':clock_exact,'event_counts_exact':events_exact,
                   'scope':'20-ms sham engine comparison only. No extrapolation of the error to 400 ms.'})
    (HERE/'REFERENCE_20_COMPARE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('screen_pass','snapshot_arrays_compared','snapshot_max_abs','snapshot_worst_path','clock_exact','event_counts_exact')}))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
