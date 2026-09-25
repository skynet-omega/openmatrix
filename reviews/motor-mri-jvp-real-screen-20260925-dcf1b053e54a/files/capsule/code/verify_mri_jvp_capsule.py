"""Cold, portable recomputation of MRI seed cost and one JVP screen.

This verifies packaged numeric arrays and receipts. It cannot replay the live
organism or certify that the packaged base-CSR edge counts are complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def same(a,b,label):need(np.array_equal(a,b),label+' mismatch')


def verify(root):
    root=Path(root)
    m=json.loads((root/'CAPSULE_MANIFEST.json').read_text())
    need(m['schema']=='mri_jvp_capsule_manifest_v1','manifest schema')
    for rel,expected in m['files'].items():
        need(sha(root/rel)==expected,'hash '+rel)
    slow=json.loads((root/'receipts/MRI_SLOW_RESULT.json').read_text())['mri']
    eff=json.loads((root/'receipts/EFFECTIVE_EVENT_READER_RESULT.json').read_text())['probe']
    row=json.loads((root/'receipts/MRI_ROW_ZONE_COST_RESULT.json').read_text())
    jvp=json.loads((root/'receipts/JVP_REAL_RESULT.json').read_text())['probe']
    with np.load(root/'data/MRI_SLOW_INPUTS.npz',allow_pickle=False) as z:
        s=z['slow_f'];y1=z['flow_y1'];y2=z['flow_y2'];high=z['high']
        need(s.shape==(5,359373) and y1.shape==y2.shape==high.shape==(359373,),
             'slow shape')
        d1=s[3]-(-s[0]+2*s[1])
        d2=s[4]-((-2*s[1]+3*s[2])+1.5*(s[0]-s[2]))
        score=np.maximum(125e-6*np.abs(d1)/(1e-7+1e-5*np.maximum(np.abs(y1),np.abs(y2))),
                         125e-6*np.abs(d2)/(1e-7+1e-5*np.maximum(np.abs(y2),np.abs(high))))
        same(d1,z['defect1'],'defect1');same(d2,z['defect2'],'defect2')
        same(score,z['score'],'sampled score')
        need(math.isclose(float(score.max()),slow['sampled_defect_normalized'],rel_tol=1e-12) and
             int(np.count_nonzero(score>1))==slow['sampled_defect_coordinates_gt1'],
             'sampled score receipt')
        prescribed=z['mask'].copy()
    with np.load(root/'data/EFFECTIVE_EVENT_READERS.npz',allow_pickle=False) as z:
        masks=z['changed_masks'];readers=np.any(masks,axis=0)
        need(masks.shape==(7,359373) and masks.dtype==np.bool_,'event masks')
        need(int(readers.sum())==eff['all_state_changed'] and
             int(readers[:166700].sum())==eff['cns_rows_changed'],
             'reader receipt')
    with np.load(root/'data/MRI_ROW_ZONE_ROWS_61.npz',allow_pickle=False) as z:
        q_def=z['q_defect'];q_ev=z['q_event_readers'];q_union=z['q_union']
        incoming=z['union_incoming_active'];marked=z['union_owner_marked']
        same(q_def,np.flatnonzero((score[:166700]>1)&~prescribed[:166700]),'q defect')
        same(q_ev,np.flatnonzero(readers[:166700]&~prescribed[:166700]),'event q')
        same(q_union,np.union1d(q_def,q_ev),'union q')
        need(incoming.shape==marked.shape==q_union.shape and incoming.dtype.kind in 'iu',
             'zone cost shape')
        E=25582938
        union_cost=int(incoming.sum());generic_cost=int(incoming[~marked].sum())
        need(union_cost==row['scenarios']['union_q']['incoming_active_base_edges'] and
             generic_cost==row['scenarios']['union_q_without_marked_owners']['incoming_active_base_edges'] and
             math.isclose(5+229*union_cost/E,
                 row['scenarios']['union_q']['equivalent_passes_no_setup_owners'],rel_tol=1e-12),
             'zone cost receipt')
        need(int(np.count_nonzero(score[:166700]>1))==row['above_one_in_q'] and
             int(np.count_nonzero(score[166700:]>1))==row['above_one_in_non_q'],
             'defect state partition')
    with np.load(root/'data/JVP_REAL_ARRAYS.npz',allow_pickle=False) as z:
        x=z['z'];v=z['v'];f=z['full_f'];mask=z['mask']
        need(x.shape==v.shape==mask.shape==(359373,) and f.shape==(6,359373,),
             'JVP layout')
        same(f[0],f[5],'repeat F(z)')
        j1=(f[1]-f[2])/2;jhalf=f[3]-f[4]
        residual=125e-6*np.abs(j1-jhalf)/(1e-7+1e-5*np.abs(x))
        same(j1,z['j1'],'JVP full');same(jhalf,z['jhalf'],'JVP half')
        same(residual,z['normalized_discrepancy'],'JVP residual')
        js=float(residual[~mask].max())
        need(math.isclose(js,jvp['normalized_linearity_discrepancy'],rel_tol=1e-12) and
             (js<=.1)==jvp['gate_le_0_1'],'JVP receipt')
    bad=score.copy();bad[0]+=1
    try:same(bad,score,'corrupt score')
    except ValueError:pass
    else:raise ValueError('corruption accepted')
    return {'schema':'mri_jvp_capsule_verify_v1','status':'PASS_PORTABLE_NUMERIC_RECOMPUTE',
        'sampled_defect_gt1':int(np.count_nonzero(score>1)),
        'effective_reader_q':int(len(q_ev)),
        'union_q':int(len(q_union)),
        'union_q_base_equivalent_passes':5+229*union_cost/E,
        'generic_q_base_equivalent_passes':5+229*generic_cost/E,
        'jvp_linearity_discrepancy':js,'corruption_rejected':True,
        'scope':'Archived arrays/hashes only; no live operator, organism, owner closure or full-graph recomputation.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();need(not a.out.exists(),'output exists')
    result=verify(a.root);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
