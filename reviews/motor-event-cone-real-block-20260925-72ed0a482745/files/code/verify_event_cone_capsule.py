"""Portable integrity and first/two-hop checks on the weighted cone subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root=Path(root)
    m=json.loads((root/'EVENT_CONE_CAPSULE_MANIFEST.json').read_text())
    result=json.loads((root/'EVENT_CONE_RESULT.json').read_text())
    need(m['capsule_sha256']==sha(root/'EVENT_CONE_CAPSULE.npz') and
         m['result_sha256']==sha(root/'EVENT_CONE_RESULT.json'),
         'payload/result hash')
    with np.load(root/'EVENT_CONE_CAPSULE.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    need(set(data)=={'source_rows','first_hop_rows','edge1_pre','edge1_post',
                     'edge1_weight','edge2_pre','edge2_post','edge2_weight',
                     'target_row'},'layout')
    s,first=data['source_rows'],data['first_hop_rows']
    need(np.array_equal(s,np.asarray(result['sources'],dtype=np.int32)) and
         len(s)==m['n_source_rows']==7 and
         len(first)==m['n_first_hop_rows']==1263,'sources/first')
    for depth in (1,2):
        pre,post,w=[data[f'edge{depth}_{name}'] for name in ('pre','post','weight')]
        need(pre.shape==post.shape==w.shape and len(pre)==m[f'n_edge{depth}'] and
             np.isfinite(w).all() and np.all(w!=0),'edge '+str(depth))
        need(np.isin(pre,s if depth==1 else first).all(),'edge source '+str(depth))
    recomputed_first=np.setdiff1d(np.unique(data['edge1_post']),s)
    need(np.array_equal(recomputed_first,first),'first-hop IDs')
    second=np.setdiff1d(np.unique(data['edge2_post']),np.union1d(s,first))
    need(len(second)==result['layers'][1]['new_neurons']==13328,'second-hop IDs')
    target=int(data['target_row'][0]);need(target==result['target_row']==29460,'target')
    need(not np.any(data['edge1_post']==target),'direct target edge')
    via=np.unique(data['edge2_pre'][data['edge2_post']==target])
    need(np.array_equal(via,np.asarray(result['target_first_hop_predecessors'])),
         'second-hop witnesses')
    broken=data['edge2_weight'].copy();broken[0]=np.nan
    try:need(np.isfinite(broken).all(),'corrupted weight')
    except ValueError:pass
    else:raise ValueError('corruption accepted')
    return {'schema':'event_cone_capsule_verify_v1',
            'status':'PASS_PORTABLE_TWO_HOP_STRUCTURE',
            'sources':len(s),'first_hop':len(first),'second_new':len(second),
            'edge1':len(data['edge1_pre']),'edge2':len(data['edge2_pre']),
            'target_two_hop_predecessors':via.tolist(),
            'corruption_rejected':True,
            'scope':'Weighted base CSR subset only; cannot certify completeness without parent full-graph hash/source.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();need(not a.out.exists(),'output exists')
    r=verify(a.root);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
