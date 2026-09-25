"""Verify Nsight-marker run and state exactly; report missing GPU activity honestly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from verify_cell_kernel_timing import compare_tree, need, sha

HERE=Path(__file__).resolve().parent
BASE=HERE/'event_step_baseline_v2_01'
RUN=HERE/'cell_nsys_01'
DB=HERE/'cell_nsys_profile_01.sqlite'


def verify():
    report=json.loads((RUN/'CELL_NSYS_RUN_RESULT.json').read_text())
    need(report['status']=='COMPLETE_DIAGNOSTIC_ONLY' and report['calls']==16 and
         report['profile_started'] and report['profile_stopped'], 'run/marker coverage')
    need(report['plan_sha256']==sha(HERE/'CELL_NSYS_PLAN_48.json') and
         report['run_result_sha256']==sha(RUN/'RESULT.json'),'frozen provenance')
    need(json.loads((RUN/'EVENT_AUDIT.json').read_text()) ==
         json.loads((BASE/'EVENT_AUDIT.json').read_text()),'event log changed')
    for rel in ('traces.npz','final_state/published.json','final_state/published.npz',
                'final_state/prosthesis.json','final_state/prosthesis.npz',
                'preparation_inputs/intervenciones_W.npz'):
        need(sha(BASE/rel)==sha(RUN/rel),'output changed: '+rel)
    with np.load(BASE/'final_state/session.npz') as x, np.load(RUN/'final_state/session.npz') as y:
        compare_tree(json.loads((BASE/'final_state/session.json').read_text()),
                     json.loads((RUN/'final_state/session.json').read_text()),x,y)
    with sqlite3.connect(DB) as db:
        tables={row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        runtime_calls=db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0]
    need(runtime_calls>0,'No CUDA API trace')
    return {'schema':'cell_nsys_verify_v1','status':'SCIENTIFIC_OUTPUT_EXACT__GPU_KERNEL_TRACE_UNAVAILABLE',
            'run_result_sha256':sha(RUN/'RESULT.json'),
            'sqlite_sha256':sha(DB),'runtime_api_calls':runtime_calls,
            'kernel_table_present':'CUPTI_ACTIVITY_KIND_KERNEL' in tables,
            'gpu_memcpy_table_present':'CUPTI_ACTIVITY_KIND_MEMCPY' in tables,
            'owner_wall_s':report['native_owner_wall_s'],
            'scope':'Nsight 2022.4 captured CUDA API but no GPU kernel/memcpy activity table on this WSL host. No kernel fraction can be inferred.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();need(not a.out.exists(),'output exists')
    r=verify();a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r))
