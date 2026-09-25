"""Localize base-weight differences between captured graph and current loader."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def run(out):
    started=time.monotonic();here=Path(__file__).resolve().parent;root=here.parents[1]
    p=json.loads((here/'GRAPH_WEIGHT_DIFF_PLAN_34.json').read_text())
    need(not out.exists() and p['schema']=='graph_weight_diff_plan_v1','Unique run/plan')
    need(sha(Path(__file__))==p['script_sha256'],'Script changed')
    need({x:sha(root/x) for x in p['frozen_inputs_sha256']}==p['frozen_inputs_sha256'],'Input changed')
    mapping=json.loads((root/p['mapping']).read_text())
    with np.load(root/p['graph_archive'],allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']]
        old=z[mapping['cuda/weights']]
    sys.path.insert(0,'/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922')
    from motor_runtime import load
    import cupy as cp
    obj,*_=load(out/'preparation_inputs')
    try:
        current=cp.asnumpy(obj.core.hybrid.cuda['weights'])
        need(current.shape==old.shape and current.dtype==old.dtype,'Weight layout')
        diff=np.flatnonzero(current!=old)
        rows=np.searchsorted(ptr,diff,side='right')-1
        result={'schema':'graph_weight_diff_result_v1',
                'status':'LOCALIZED_WEIGHT_DIFFERENCE',
                'count':int(len(diff)),
                'fraction':float(len(diff)/len(old)),
                'affected_rows':int(len(np.unique(rows))),
                'maximum_abs_difference':float(np.max(np.abs(current[diff]-old[diff]))) if len(diff) else 0.,
                'sample_positions':diff[:16].tolist(),
                'sample_rows':rows[:16].tolist(),
                'old_sha256':hashlib.sha256(old.tobytes()).hexdigest(),
                'current_sha256':hashlib.sha256(current.tobytes()).hexdigest(),
                'plan_sha256':sha(here/'GRAPH_WEIGHT_DIFF_PLAN_34.json'),
                'wall_s':time.monotonic()-started,
                'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scope':'Fresh current loader versus older captured W; no simulation. Full current weights saved for an isolated matched-base rerun, not for online ownership.'}
        np.savez_compressed(out/'FRESH_WEIGHTS.npz',weights=current)
        result['fresh_weights_sha256']=sha(out/'FRESH_WEIGHTS.npz')
    finally:
        obj.close()
    b=p['budget'];result['budget_ok']=result['wall_s']<=b['wall_seconds_max'] and result['maxrss_kib']<=b['ram_gib_max']*1024**2 and (out/'FRESH_WEIGHTS.npz').stat().st_size<=b['weights_bytes_max']
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--out',type=Path,required=True);a=x.parse_args()
    r=run(a.out)
    print(json.dumps({k:r[k] for k in ('status','count','fraction','affected_rows',
                                       'maximum_abs_difference','wall_s','budget_ok')}))
