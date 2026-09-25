"""Independent position-based reachability check of the recorded event cone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'motor_nuevo/multirate_real_20260924_01/capture_01'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify():
    report=json.loads((HERE/'EVENT_CONE_RESULT.json').read_text())
    need(report['status']=='STRUCTURAL_ONLY' and
         report['plan_sha256']==sha(HERE/'EVENT_CONE_PLAN_53.json'),
         'report/plan')
    mapping=json.loads((CAP/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(CAP/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']]
        indices=z[mapping['cuda/indices']]
        weights=z[mapping['cuda/weights']]
    n=len(ptr)-1
    need(n==report['n_neurons'] and len(indices)==report['csr_stored_edges'],
         'graph layout')
    active=weights!=0
    need(int(active.sum())==report['csr_active_edges'],'active edges')
    sources=np.asarray(report['sources'],dtype=np.int32)
    need(len(sources)==7 and np.all((sources>=0)&(sources<n)), 'source domain')
    outdegree=np.bincount(indices[active],minlength=n)
    need(int(outdegree[sources].sum())==report['event_source_outgoing_active_edges'],
         'source outgoing')
    seen=np.zeros(n,dtype=bool);seen[sources]=True
    frontier=seen.copy()
    computed=[]
    for depth in range(1,5):
        # Independent from the CSR @ vector implementation: select edge slots
        # by source ID, map their positions back to receptor rows.
        slots=np.flatnonzero(active & frontier[indices])
        posts=np.unique(np.searchsorted(ptr,slots,side='right')-1)
        reached=np.zeros(n,dtype=bool);reached[posts]=True
        new=reached & ~seen
        seen|=reached
        wanted=report['layers'][depth-1]
        need(int(new.sum())==wanted['new_neurons'] and
             int(seen.sum())==wanted['reachable_neurons_cumulative'] and
             int(outdegree[seen].sum())==wanted['outgoing_active_edges_from_cumulative'],
             'layer '+str(depth))
        computed.append({'depth':depth,'new':int(new.sum()),
                         'cumulative':int(seen.sum())})
        frontier=new
    need(report['target_first_reachability_depth']==2 and
         not report['target_direct_event_sources'], 'target path')
    bad=sources.copy();bad[0]=report['target_row']
    need(int(outdegree[bad].sum())!=report['event_source_outgoing_active_edges'],
         'source corruption undetected')
    return {'schema':'event_cone_verify_v1','status':'PASS_STRUCTURAL_RECOMPUTE',
            'layers':computed,'source_corruption_rejected':True,
            'graph_sha256':sha(CAP/'effective_gpu_arrays.npz'),
            'scope':'Base CSR nonzero reachability; no claims about effective specialized edges or dynamics.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();need(not a.out.exists(),'output exists')
    result=verify();a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
