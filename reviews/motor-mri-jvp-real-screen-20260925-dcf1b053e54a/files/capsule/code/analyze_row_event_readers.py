"""Cost screen of incoming base CSR rows reached directly by real event ports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'motor_nuevo/multirate_real_20260924_01/capture_01'
PLAN=HERE/'ROW_EVENT_READER_PLAN_57.json'
SPECIAL_KEYS=(
    '_mi9_cuda/rows','_retinal_port_cuda_rows','_pvlp_cuda_rows',
    '_orn_pn_cuda/rows','_regional_cuda/rows','_gaba_cuda/rows',
    '_retinal_cuda/rows','_cvn7_cuda_rows','_boundary_cuda/rows',
    '_pnkc_cuda/rows','_output_cuda/rows','_apl_gpu_rows',
    '_dynamic_gpu_rows','_orn_terminal_cuda/rows')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(out,rows_out):
    start=time.monotonic()
    need(not out.exists() and not rows_out.exists(),'output exists')
    plan=json.loads(PLAN.read_text())
    need(plan['schema']=='row_event_reader_plan_v1','plan schema')
    for rel,expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
    cone=json.loads((HERE/'EVENT_CONE_RESULT.json').read_text())
    mri=json.loads((HERE/'mri_real_frozen_fast_04/MRI_REAL_FROZEN_FAST_RESULT.json').read_text())
    need(mri['mri']['full_calls']==5 and mri['mri']['fast_calls']==229,'MRI call count')
    mapping=json.loads((CAP/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(CAP/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']].copy()
        idx=z[mapping['cuda/indices']].copy()
        weights=z[mapping['cuda/weights']].copy()
        visual=z[mapping['cuda/visual']].copy()
        extra={key:z[mapping[key]].copy() for key in SPECIAL_KEYS}
    n=len(ptr)-1;E=len(idx)
    need(n==166700 and E==cone['csr_stored_edges'] and ptr[-1]==E,
         'CSR identity')
    need(visual.shape==(n,) and visual.dtype==np.bool_,'visual mask')
    sources=np.asarray(cone['sources'],dtype=np.int64)
    source_mask=np.zeros(n,dtype=bool);source_mask[sources]=True
    active=weights!=0
    edge_slot=np.flatnonzero(active & source_mask[idx])
    need(len(edge_slot)==cone['event_source_outgoing_active_edges'],'source edge count')
    first=np.setdiff1d(np.unique(np.searchsorted(ptr,edge_slot,side='right')-1),sources)
    need(len(first)==cone['layers'][0]['new_neurons'],'first-hop identity')
    incoming_active=np.asarray([int(np.count_nonzero(active[ptr[i]:ptr[i+1]]))
                                for i in first],dtype=np.int32)
    incoming_stored=(ptr[first+1]-ptr[first]).astype(np.int32)
    special=np.zeros(n,dtype=bool)
    detail={}
    for key,rows in extra.items():
        need(rows.ndim==1 and rows.dtype.kind in 'iu' and
             np.all((rows>=0)&(rows<n)),'special layout: '+key)
        special[rows]=True
        detail[key]={'declared_rows':int(len(rows)),
                     'direct_event_readers':int(np.isin(first,rows).sum())}
    labels={
        'all_direct_base_readers':np.ones(len(first),dtype=bool),
        'readers_without_listed_special_owner':~special[first],
        'readers_without_listed_special_owner_or_visual':~special[first]&~visual[first],
    }
    need(list(labels)==plan['scenarios'],'scenario mismatch')
    scenarios={}
    for label,mask in labels.items():
        edges=int(incoming_active[mask].sum())
        scenarios[label]={'rows':int(mask.sum()),
            'incoming_active_base_edges':edges,
            'incoming_stored_base_edges':int(incoming_stored[mask].sum()),
            'five_full_plus_229_fast_equivalent_passes_no_setup_owners':5+229*edges/E,
            'six_pass_screen':5+229*edges/E<=6}
    need(scenarios['all_direct_base_readers']['rows']==len(first),'all readers')
    need(scenarios['all_direct_base_readers']['incoming_active_base_edges']>=
         scenarios['readers_without_listed_special_owner']['incoming_active_base_edges']>=
         scenarios['readers_without_listed_special_owner_or_visual']['incoming_active_base_edges'],
         'cost monotonicity')
    np.savez_compressed(rows_out,first_hop_rows=first.astype(np.int32),
        incoming_active=incoming_active,incoming_stored=incoming_stored,
        listed_special=special[first],visual=visual[first],sources=sources.astype(np.int32))
    result={'schema':'row_event_reader_result_v1','status':'STRUCTURAL_COST_SCREEN_ONLY',
        'plan_sha256':sha(PLAN),'rows_sha256':sha(rows_out),
        'n_event_sources':len(sources),'n_first_hop_rows':len(first),
        'base_stored_edges':E,'base_active_edges':int(active.sum()),
        'maximum_edges_per_fast_for_six_passes_no_setup_owners':E//229,
        'scenarios':scenarios,'special_metadata':detail,
        'hdeltaa_target_direct_reader':bool(np.any(first==cone['target_row'])),
        'wall_s':time.monotonic()-start,
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Direct readers in nonzero base CSR. Owner/visual exclusion is an exploratory mask, not effective final read dependency; specialized costs and time omitted.'}
    result['budget_ok']=(result['wall_s']<=plan['budget']['wall_s_max'] and
        result['maxrss_kib']<=plan['budget']['ram_gib_max']*1024**2 and
        rows_out.stat().st_size+len(json.dumps(result).encode())<=plan['budget']['disk_bytes_max'])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need(result['budget_ok'],'budget exceeded')
    print(json.dumps({'status':result['status'],'scenarios':scenarios,
                      'wall_s':result['wall_s']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--rows-out',type=Path,required=True)
    a=ap.parse_args();run(a.out,a.rows_out)
