"""Compare frozen CSR operator with a fresh load of the current preparation."""
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
    start=time.monotonic()
    here=Path(__file__).resolve().parent
    root=here.parents[1]
    p=json.loads((here/'GRAPH_PREPARATION_PLAN_33.json').read_text())
    need(not out.exists() and p['schema']=='graph_preparation_comparison_v1',
         'Unique run/plan')
    need(sha(Path(__file__))==p['script_sha256'],'Script changed')
    need({x:sha(root/x) for x in p['frozen_inputs_sha256']}==p['frozen_inputs_sha256'],
         'Input changed')
    mapping=json.loads((root/p['mapping']).read_text())
    with np.load(root/p['graph_archive'],allow_pickle=False) as z:
        frozen={key:z[mapping['cuda/'+key]] for key in
                ('indptr','indices','weights','caps','visual')}
    sys.path.insert(0,str(Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922')))
    from motor_runtime import load
    import cupy as cp
    obj,*_=load(out/'preparation_inputs')
    try:
        h=obj.core.hybrid
        live={key:cp.asnumpy(h.cuda[key]) for key in frozen}
        same={key:bool(np.array_equal(frozen[key],live[key])) for key in frozen}
        result={'schema':'graph_preparation_comparison_result_v1',
                'status':'IDENTICAL_BASE_OPERATOR' if all(same.values()) else 'DIFFERENT_BASE_OPERATOR',
                'plan_sha256':sha(here/'GRAPH_PREPARATION_PLAN_33.json'),
                'same':same,'frozen_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest()
                                               for key,value in frozen.items()},
                'live_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest()
                                             for key,value in live.items()},
                'wall_s':time.monotonic()-start,
                'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scope':'Fresh load from the current loader versus archived graph arrays; no CNS/body step. Matching base arrays do not prove all owners/state matched between runs.'}
    finally:
        obj.close()
    b=p['budget']
    result['budget_ok']=result['wall_s']<=b['wall_seconds_max'] and result['maxrss_kib']<=b['ram_gib_max']*1024**2
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--out',type=Path,required=True);a=x.parse_args()
    r=run(a.out)
    print(json.dumps({k:r[k] for k in ('status','same','wall_s','budget_ok')}))
