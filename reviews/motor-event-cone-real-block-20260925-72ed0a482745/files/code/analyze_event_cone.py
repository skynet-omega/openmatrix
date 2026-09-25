"""Structural CNS reachability from seven real events in a frozen accepted block."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'motor_nuevo/multirate_real_20260924_01/capture_01'
PLAN=HERE/'EVENT_CONE_PLAN_53.json'
TARGET=29460


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def decode(stem):
    j=json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:
        return {k:(z[v['__array__']].copy() if isinstance(v,dict) and
                   set(v)=={'__array__'} else v) for k,v in j.items()}


def run(out):
    t0=time.monotonic()
    need(not out.exists(),'output exists')
    p=json.loads(PLAN.read_text());need(p['schema']=='event_cone_plan_v1','plan')
    for rel,expected in p['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source/data changed: '+rel)
    audit0=json.loads((CAP/'EVENT_AUDIT.json').read_text())['blocks'][1]
    audit1=json.loads((HERE/'mri_real_frozen_fast_03/EVENT_AUDIT.json').read_text())['blocks'][1]
    need(audit0==audit1 and audit0['duration_ns']==125000 and len(audit0['events'])==7,
         'wrong accepted block')
    ports=decode(CAP/'block_events')
    sources=np.unique(ports['rows'][ports['event_rows']]).astype(np.int32)
    need(len(sources)==7 and len(ports['times'])==7,'source event cohort')
    mapping=json.loads((CAP/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(CAP/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']]
        indices=z[mapping['cuda/indices']]
        weights=z[mapping['cuda/weights']]
    n=len(ptr)-1;m=len(indices)
    need(n==166700 and m==25582938 and ptr[-1]==m and weights.shape==(m,),
         'CSR layout')
    active=weights!=0
    graph=sparse.csr_matrix((active,indices,ptr),shape=(n,n),dtype=np.bool_)
    outdegree=np.bincount(indices[active],minlength=n).astype(np.int64)
    n_active=int(active.sum())
    seen=np.zeros(n,dtype=bool)
    frontier=np.zeros(n,dtype=bool);frontier[sources]=True
    seen[sources]=True
    layers=[];target_first=None
    for depth in range(1,5):
        reached=np.asarray(graph@frontier,dtype=bool)
        new=reached & ~seen
        if reached[TARGET] and target_first is None:target_first=depth
        seen |= reached
        layers.append({'depth':depth,'new_neurons':int(new.sum()),
            'reachable_neurons_cumulative':int(seen.sum()),
            'reachable_fraction':float(seen.mean()),
            'outgoing_active_edges_from_new':int(outdegree[new].sum()),
            'outgoing_active_edges_from_cumulative':int(outdegree[seen].sum()),
            'cumulative_outgoing_fraction':float(outdegree[seen].sum()/n_active),
            'target_reachable':bool(reached[TARGET])})
        frontier=new
    target_sources=indices[ptr[TARGET]:ptr[TARGET+1]]
    target_active=active[ptr[TARGET]:ptr[TARGET+1]]
    direct=np.intersect1d(target_sources[target_active],sources)
    # Recompute first-hop to identify the direct predecessors that reach target.
    flag=np.zeros(n,dtype=bool);flag[sources]=True
    first=np.asarray(graph@flag,dtype=bool)
    via_first=np.unique(target_sources[target_active & first[target_sources]])
    result={'schema':'event_cone_result_v1','plan_sha256':sha(PLAN),
        'status':'STRUCTURAL_ONLY','sources':sources.tolist(),
        'n_neurons':n,'csr_stored_edges':m,'csr_active_edges':n_active,
        'event_source_outgoing_active_edges':int(outdegree[sources].sum()),
        'target_row':TARGET,'target_indegree_active':int(target_active.sum()),
        'target_direct_event_sources':direct.tolist(),
        'target_first_reachability_depth':target_first,
        'target_first_hop_predecessors':via_first.tolist(),
        'layers':layers,'wall_s':time.monotonic()-t0,
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Base nonzero CSR structural reachability; not dynamics, current, specialized PN/KC/visual edges, or proof of causal source for MRI error.'}
    result['budget_ok']=(result['wall_s']<=p['budget']['wall_s_max'] and
        result['maxrss_kib']<=p['budget']['ram_gib_max']*1024**2)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need(result['budget_ok'],'budget exceeded')
    print(json.dumps({k:result[k] for k in ('status','target_first_reachability_depth',
                                            'event_source_outgoing_active_edges','layers','wall_s')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.out)
