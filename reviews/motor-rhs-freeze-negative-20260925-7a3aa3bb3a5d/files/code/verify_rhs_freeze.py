"""Independent recomputation of the frozen-RHS retrospective diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(corrupt=False):
    p=json.loads((HERE/'RHS_FREEZE_PLAN_44.json').read_text())
    result=json.loads((HERE/'RHS_FREEZE_RESULT.json').read_text())
    if corrupt:result['one_per_event_interval']['queries_over_1']+=1
    need(result['plan_sha256']==sha(HERE/'RHS_FREEZE_PLAN_44.json'),'Plan/result mismatch')
    source=HERE/'quantized_full_01/FULL_COEFFICIENT_ARRAYS.npz'
    audit=HERE/'quantized_full_01/EVENT_AUDIT.json'
    need(sha(source)==p['full_arrays_sha256'] and sha(audit)==p['audit_sha256'],
         'Input hash changed')
    with np.load(source,allow_pickle=False) as z:
        x=z['consumed_z'];a=z['consumed_target'];r=z['consumed_rate']
        h=z['trial_step_s'];t=z['query_s']
    events=sorted({e['time_s'] for e in json.loads(audit.read_text())['blocks'][1]['events']})
    segment=np.searchsorted(events,t,side='right')
    groups={int(s):np.where(segment==s)[0][0] for s in np.unique(segment)}
    f=np.multiply(r,np.subtract(a,x))
    need(result['queries']==len(t)==60 and result['segments_with_queries']==len(groups)==8,
         'Query/segment count')
    for label in ('whole_block','one_per_event_interval'):
        maxima=[];maxidx=[]
        for i in range(60):
            j=0 if label=='whole_block' else groups[int(segment[i])]
            zmax=np.maximum(np.abs(x[i]),np.abs(x[j]))
            v=np.divide(h[i]*np.abs(f[i]-f[j]),3*(p['atol']+p['rtol']*zmax))
            k=int(np.argmax(v));maxidx.append(k);maxima.append(float(v[k]))
        v=np.array(maxima);row=result[label];worst=int(np.argmax(v))
        for key,actual in (('max',v.max()),('median',np.median(v)),('p90',np.quantile(v,.9))):
            need(abs(row[key]-actual)<1e-11,label+'/'+key)
        for key,actual in (('queries_over_0_1',int((v>.1).sum())),
                           ('queries_over_1',int((v>1).sum())),
                           ('worst_query',worst),
                           ('worst_state_index',maxidx[worst])):
            need(row[key]==actual,label+'/'+key)
    return {'status':'PASS_RECOMPUTED','whole_over_1':result['whole_block']['queries_over_1'],
            'interval_over_1':result['one_per_event_interval']['queries_over_1']}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrupt',action='store_true')
    args=ap.parse_args();print(json.dumps(run(args.corrupt)))
