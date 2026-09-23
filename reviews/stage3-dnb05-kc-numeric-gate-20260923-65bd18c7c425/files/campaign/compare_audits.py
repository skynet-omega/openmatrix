"""Find first source/row/block mismatch without trusting runner verdicts."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path


H=Path(__file__).resolve().parent
A=H/'native_sham_20_01'
B=H/'reference_sham_20_01'


def read(folder):
    r=json.loads((folder/'RESULT.json').read_text())
    x=json.loads((folder/'EVENT_AUDIT.json').read_text())
    if r['status']!='COMPLETE' or r['completed_trial_ms']!=20 or len(x['blocks'])!=320:
        raise ValueError('Incomplete event audit')
    if sum(len(y['events']) for y in x['blocks'])!=r['runtime']['events']['events']:
        raise ValueError('Logged events do not reconstruct the runtime count')
    return r,x


def counts(block):return Counter((e['producer'],e['row'],e['neuron_id']) for e in block['events'])


def main():
    ra,a=read(A);rb,b=read(B)
    first=None;differing=[]
    for aa,bb in zip(a['blocks'],b['blocks']):
        if aa['block']!=bb['block'] or aa['start_elapsed_ns']!=bb['start_elapsed_ns'] or aa['duration_ns']!=bb['duration_ns']:
            raise ValueError('Physical block clocks changed')
        ca,cb=counts(aa),counts(bb)
        if ca!=cb:
            keys=sorted(set(ca)|set(cb))
            diff=[{'producer':k[0],'row':k[1],'neuron_id':k[2],
                   'causal_events':ca[k],'reference_events':cb[k]}
                  for k in keys if ca[k]!=cb[k]]
            item={'block':aa['block'],'start_elapsed_ns':aa['start_elapsed_ns'],
                  'duration_ns':aa['duration_ns'],'differences':diff}
            differing.append(item)
            if first is None:first=item
    total_a=Counter(k for block in a['blocks'] for k in counts(block).elements())
    total_b=Counter(k for block in b['blocks'] for k in counts(block).elements())
    totals=[{'producer':k[0],'row':k[1],'neuron_id':k[2],
             'causal_events':total_a[k],'reference_events':total_b[k]}
            for k in sorted(set(total_a)|set(total_b)) if total_a[k]!=total_b[k]]
    # Inspect temporal differences only for events with the same source/row
    # cardinality in a block. Do not pair two traces after a count mismatch.
    first_time=None;maximum_time_abs=0.
    for aa,bb in zip(a['blocks'],b['blocks']):
        if counts(aa)!=counts(bb):continue
        da=defaultdict(list);db=defaultdict(list)
        for e in aa['events']:da[(e['producer'],e['row'])].append(e['time_s'])
        for e in bb['events']:db[(e['producer'],e['row'])].append(e['time_s'])
        for key in da:
            for ta,tb in zip(sorted(da[key]),sorted(db[key])):
                gap=abs(ta-tb);maximum_time_abs=max(maximum_time_abs,gap)
                if gap and first_time is None:
                    first_time={'block':aa['block'],'producer':key[0],
                                'row':key[1],'causal_time_s':ta,'reference_time_s':tb,
                                'abs_s':gap}
    result={'schema':'stage3_first_event_discrepancy_v1',
            'causal_events':ra['runtime']['events']['events'],
            'reference_events':rb['runtime']['events']['events'],
            'differing_blocks':len(differing),'first_count_difference':first,
            'all_count_differences':differing,'per_neuron_total_differences':totals,
            'first_time_difference_when_counts_equal':first_time,
            'maximum_matched_event_time_abs_s':maximum_time_abs,
            'scope':'Exact event identities/counts at block and row level; temporal pairing only when local counts match. No causal attribution from this report alone.'}
    (H/'EVENT_COMPARE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('causal_events','reference_events','differing_blocks','first_count_difference','per_neuron_total_differences','maximum_matched_event_time_abs_s')}))


if __name__=='__main__':main()
