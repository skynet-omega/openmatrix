"""Compare committed full-block events; predictor half-block events are discarded."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


H=Path(__file__).resolve().parent
FULL_NS=125000


def collect(folder):
    data=json.loads((H/folder/'EVENT_AUDIT.json').read_text())
    events=defaultdict(list);predictor=0;committed=0
    for block in data['blocks']:
        duration=block['duration_ns']
        if duration not in (FULL_NS//2,FULL_NS):raise ValueError('Unknown midpoint block duration')
        if duration!=FULL_NS:
            predictor+=len(block['events']);continue
        committed+=len(block['events'])
        for event in block['events']:
            key=(event['producer'],event['row'],event['neuron_id'])
            when=block['start_elapsed_ns']+round(event['time_s']*1e9)
            events[key].append((when,event['post_q']))
    for rows in events.values():rows.sort()
    return events,predictor,committed


def main():
    a,preda,na=collect('audit_causal_20_01')
    b,predb,nb=collect('audit_native_20_01')
    mismatched=[];timing=[];posts=[];worst=None
    for key in sorted(set(a)|set(b)):
        aa=a.get(key,[]);bb=b.get(key,[])
        if len(aa)!=len(bb):
            mismatched.append({'producer':key[0],'row':key[1],'neuron_id':key[2],
                               'causal':len(aa),'reference':len(bb)})
            continue
        for (ta,qa),(tb,qb) in zip(aa,bb):
            dt=abs(ta-tb);timing.append(dt)
            if qa is not None and qb is not None:posts.append(abs(qa-qb))
            if worst is None or dt>worst['abs_ns']:
                worst={'producer':key[0],'row':key[1],'neuron_id':key[2],
                       'causal_time_ns':ta,'reference_time_ns':tb,'abs_ns':dt}
    result={'schema':'stage3_committed_event_comparison_v1',
            'accepted_block_ns':FULL_NS,'predictor_events_causal':preda,
            'predictor_events_reference':predb,'committed_events_causal':na,
            'committed_events_reference':nb,'per_neuron_count_differences':mismatched,
            'paired_events':len(timing),'maximum_event_time_abs_ns':max(timing,default=0),
            'event_time_p99_abs_ns':float(np.quantile(timing,.99)) if timing else 0.,
            'maximum_physical_post_q_abs':max(posts,default=0.),
            'worst_timing_event':worst,
            'scope':'Midpoint code confirms half-block predictor is restored; this filters by the frozen 125000-ns full-block clock. Event timing error is measured, not accepted by a new threshold.'}
    (H/'ACCEPTED_EVENT_COMPARE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('predictor_events_causal','predictor_events_reference','committed_events_causal','committed_events_reference','per_neuron_count_differences','maximum_event_time_abs_ns','worst_timing_event')}))


if __name__=='__main__':main()
