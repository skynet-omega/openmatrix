"""Offline optimistic base-CSR cost of real MRI row seed, before owner closure."""
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
PLAN=HERE/'MRI_ROW_ZONE_COST_PLAN_61.json'
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
    t0=time.monotonic()
    need(not out.exists() and not rows_out.exists(),'output exists')
    plan=json.loads(PLAN.read_text());need(plan['schema']=='mri_row_zone_cost_plan_v1','plan')
    for rel,expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
    with np.load(HERE/'mri_slow_inputs_01/MRI_SLOW_INPUTS.npz',allow_pickle=False) as z:
        score=z['score'].copy();prescribed=z['mask'].copy()
    with np.load(HERE/'effective_event_readers_01/EFFECTIVE_EVENT_READERS.npz',allow_pickle=False) as z:
        readers=np.any(z['changed_masks'],axis=0)
    n=166700
    need(score.shape==prescribed.shape==readers.shape==(359373,) and
         np.isfinite(score).all(),'state layout')
    need(not np.any((score>1)&prescribed),'high defect on prescribed')
    q_defect=np.flatnonzero((score[:n]>1)&~prescribed[:n])
    q_readers=np.flatnonzero(readers[:n]&~prescribed[:n])
    union=np.union1d(q_defect,q_readers)
    need(len(union)>0 and len(q_defect)>0 and len(q_readers)>0,'empty seed')
    mapping=json.loads((CAP/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(CAP/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']].copy()
        weights=z[mapping['cuda/weights']].copy()
        visual=z[mapping['cuda/visual']].copy()
        owners={key:z[mapping[key]].copy() for key in SPECIAL_KEYS}
    E=len(weights);need(len(ptr)==n+1 and ptr[-1]==E==25582938,'base CSR')
    active=weights!=0
    special=np.zeros(n,dtype=bool)
    for rows in owners.values():special[rows]=True
    def cost(rows):
        incoming=np.asarray([int(np.count_nonzero(active[ptr[i]:ptr[i+1]]))
                             for i in rows],dtype=np.int32)
        return incoming
    counts=cost(union)
    scenarios={}
    for name,rows in (('event_readers_q',q_readers),('sampled_defect_q',q_defect),
                      ('union_q',union),('union_q_without_marked_owners',union[~special[union]]),
                      ('union_q_without_marked_owners_or_visual',
                       union[~special[union]&~visual[union]])):
        edges=int(cost(rows).sum())
        total=5+229*edges/E
        scenarios[name]={'rows':int(len(rows)),'incoming_active_base_edges':edges,
            'equivalent_passes_no_setup_owners':total,'six_pass_screen':total<=6}
    np.savez_compressed(rows_out,q_defect=q_defect.astype(np.int32),
        q_event_readers=q_readers.astype(np.int32),q_union=union.astype(np.int32),
        union_incoming_active=counts,union_owner_marked=special[union],
        union_visual=visual[union],all_defect_indices=np.flatnonzero(score>1).astype(np.int32),
        all_defect_scores=score[score>1])
    result={'schema':'mri_row_zone_cost_result_v1','status':'OPTIMISTIC_BASE_CSR_SCREEN_ONLY',
        'plan_sha256':sha(PLAN),'rows_sha256':sha(rows_out),
        'full_state_coordinates_above_sampled_defect_one':int(np.count_nonzero(score>1)),
        'above_one_in_q':int(np.count_nonzero(score[:n]>1)),
        'above_one_in_non_q':int(np.count_nonzero(score[n:]>1)),
        'event_reader_q':int(len(q_readers)),
        'q_union_owner_marked':int(special[union].sum()),
        'q_union_visual':int(visual[union].sum()),
        'max_base_edges_per_fast_with_five_full':E//229,
        'scenarios':scenarios,'wall_s':time.monotonic()-t0,
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'State-dependent observed q seed; full s/tail mappings and true owner job costs absent. Direct-row base cost is conditional, not universal MRI lower bound.'}
    result['budget_ok']=(result['wall_s']<=plan['budget']['wall_s_max'] and
        result['maxrss_kib']<=plan['budget']['ram_gib_max']*1024**2 and
        rows_out.stat().st_size+len(json.dumps(result).encode())<=plan['budget']['disk_bytes_max'])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need(result['budget_ok'],'budget exceeded')
    print(json.dumps({'status':result['status'],'scenarios':scenarios,
                      'non_q_defect':result['above_one_in_non_q'],'wall_s':result['wall_s']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--rows-out',type=Path,required=True)
    a=ap.parse_args();run(a.out,a.rows_out)
