"""Reconstruct where every observed CNS trial ended relative to event cuts."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(arrays_path,audit_path,plan_path,out):
    start=time.monotonic();p=json.loads(plan_path.read_text())
    need(p['schema']=='event_clock_analysis_plan_v1','Plan schema')
    need(sha(Path(__file__))==p['script_sha256'],'Script changed')
    need(sha(arrays_path)==p['arrays_sha256'] and sha(audit_path)==p['audit_sha256'],
         'Inputs changed')
    audit=json.loads(audit_path.read_text())
    with np.load(arrays_path,allow_pickle=False) as z:
        times=z['times'];errors=z['reconstructed_error'];winners=z['winner_indices']
        reference=z['reference_error'];block_values=z['block_values']
    need(times.shape==(191,2) and len(audit['blocks'])==16,'Cohort changed')
    need(np.max(np.abs(np.max(block_values,axis=1)-reference))<=p['reconstruction_max_abs'],
         'Error reconstruction')
    starts=[0]+[j for j in range(1,len(times))
                if times[j,0]<times[j-1,0]-p['clock_tolerance_s']]
    starts.append(len(times))
    need(len(starts)-1==len(audit['blocks']),'Epoch partition')
    by_epoch=[];by_trial=[]
    for block,b in enumerate(audit['blocks']):
        first,last=starts[block:block+2]
        event_times=np.unique([x['time_s'] for x in b['events']])
        end=b['duration_ns']*1e-9
        need(abs(times[first,0])<=p['clock_tolerance_s'],'Epoch must start at zero')
        h=times[first:last,1];t=times[first:last,0];endpoint=t+h
        need(abs(endpoint[-1]-end)<=p['clock_tolerance_s'] and
             np.all(h>0) and np.all(np.diff(t)>=0),'Epoch clock coverage')
        cuts=0;endcuts=0
        for local,j in enumerate(range(first,last)):
            e=float(endpoint[local])
            event_cut=bool(len(event_times) and
                           np.min(np.abs(event_times-e))<=p['clock_tolerance_s'])
            epoch_end=abs(e-end)<=p['clock_tolerance_s']
            cuts+=int(event_cut);endcuts+=int(epoch_end)
            by_trial.append({'global_trial':j,'block':block,'phase':
                             'predictor' if block%2==0 else 'accepted',
                             'h_s':float(h[local]),'error':float(errors[j]),
                             'limiter_row':int(winners[j]),
                             'event_cut':event_cut,'epoch_end':epoch_end})
        by_epoch.append({'block':block,'phase':'predictor' if block%2==0 else 'accepted',
                         'trials':last-first,'event_times':int(len(event_times)),
                         'event_cut_trials':cuts,'epoch_end_trials':endcuts,
                         'max_error':float(np.max(errors[first:last])),
                         'median_error':float(np.median(errors[first:last])),
                         'duration_s':end,'sum_h_s':float(np.sum(h))})
    accepted=[x for x in by_trial if x['phase']=='accepted']
    predictor=[x for x in by_trial if x['phase']=='predictor']
    noncut=[x for x in by_trial if not x['event_cut'] and not x['epoch_end']]
    result={'schema':'event_clock_analysis_result_v1','status':'PASS_CLOCK_PARTITION',
            'plan_sha256':sha(plan_path),
            'all_trials':len(by_trial),'accepted_trials':len(accepted),
            'predictor_trials':len(predictor),
            'event_cut_trials':sum(x['event_cut'] for x in by_trial),
            'epoch_end_trials':sum(x['epoch_end'] for x in by_trial),
            'noncut_trials':len(noncut),
            'noncut_median_error':float(np.median([x['error'] for x in noncut])),
            'accepted_event_cut_trials':sum(x['event_cut'] for x in accepted),
            'accepted_noncut_trials':sum(not x['event_cut'] and not x['epoch_end'] for x in accepted),
            'limiter_52584_accepted':sum(x['limiter_row']==52584 for x in accepted),
            'limiter_52584_predictor':sum(x['limiter_row']==52584 for x in predictor),
            'by_epoch':by_epoch,'by_trial':by_trial,
            'wall_s':time.monotonic()-start,
            'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'Offline classification of observed accepted/predictor CNS trial endpoints. Does not rerun integrator or prove event-cut policy can be changed safely.'}
    b=p['budget']
    need(result['wall_s']<=b['wall_seconds_max'] and
         result['maxrss_kib']<=b['ram_gib_max']*1024**2,'Budget')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need(out.stat().st_size<=b['output_bytes_max'],'Output budget')
    return result


if __name__=='__main__':
    x=argparse.ArgumentParser()
    x.add_argument('--arrays',type=Path,required=True)
    x.add_argument('--audit',type=Path,required=True)
    x.add_argument('--plan',type=Path,required=True)
    x.add_argument('--out',type=Path,required=True)
    a=x.parse_args();r=run(a.arrays,a.audit,a.plan,a.out)
    print(json.dumps({k:r[k] for k in ('status','all_trials','accepted_trials',
                                      'predictor_trials','event_cut_trials',
                                      'epoch_end_trials','noncut_trials',
                                      'accepted_noncut_trials','noncut_median_error')}))
