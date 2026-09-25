"""Partition source-scan/event/continuous costs and full CSR base cost on GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(out):
    import cupy as cp
    import scipy.sparse as sp
    from csc_quantized_gpu_probe import CUDA
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    plan_path = here / 'CSC_COST_CURRENT_PLAN_36.json'
    p = json.loads(plan_path.read_text())
    need(not out.exists() and p['schema'] == 'csc_cost_split_plan_v1', 'Unique plan/run')
    need(sha(Path(__file__)) == p['script_sha256'], 'Script changed')
    need({x: sha(root/x) for x in p['frozen_inputs_sha256']} ==
         p['frozen_inputs_sha256'], 'Inputs changed')
    started = time.monotonic()
    paths = json.loads((root/p['mapping']).read_text())
    with np.load(root/p['graph_archive'], allow_pickle=False) as z:
        ptr, src, w, caps, visual = [z[paths[k]] for k in
                                      ('cuda/indptr','cuda/indices','cuda/weights',
                                       'cuda/caps','cuda/visual')]
    with np.load(root/p['current_weights_archive'], allow_pickle=False) as z:
        current_w = z['weights']
    need(current_w.shape == w.shape and current_w.dtype == w.dtype,
         'Current weight layout')
    w = current_w
    with np.load(root/p['release_capsule'], allow_pickle=False) as z:
        release, event_rows = z['release'], z['event_rows']
    n = len(caps)
    csc = sp.csr_matrix((w,src,ptr),shape=(n,n)).tocsc()
    event_only = np.broadcast_to(release[0],release.shape).copy()
    event_only[:,event_rows] = release[:,event_rows]
    continuous_only = release.copy()
    continuous_only[:,event_rows] = release[0,event_rows]
    null = np.broadcast_to(release[0],release.shape).copy()
    module = cp.RawModule(code=CUDA,options=('--std=c++11','--fmad=false',
                                            '--prec-div=true','--prec-sqrt=true'))
    seed,delta = [module.get_function(name) for name in ('csr_seed','csc_delta')]
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        dptr,dsrc,dw,dcp,ddst,dcw = [cp.asarray(x) for x in
                                      (ptr,src,w,csc.indptr.astype(np.int64),
                                       csc.indices.astype(np.int32),csc.data)]
        dcaps,dvisual = cp.asarray(caps),cp.asarray(visual)
        devent = cp.zeros(n,dtype=cp.bool_)
        devent[event_rows] = True
        variants = {'all':cp.asarray(release), 'event':cp.asarray(event_only),
                    'continuous':cp.asarray(continuous_only),'null':cp.asarray(null)}
        cache = cp.empty(n,dtype=cp.float64)
        a,b = cp.empty(n,dtype=cp.float64),cp.empty(n,dtype=cp.float64)
        grid = ((n*32+255)//256,)
        def reset():
            cp.copyto(cache,variants['all'][0])
            seed(grid,(256,),(np.int32(n),dptr,dsrc,dw,variants['all'][0],
                               dcaps,dvisual,np.float64(p['scale']),a,b))
        def capture(mode):
            states = variants[mode]
            reset();stream.synchronize();stream.begin_capture()
            for j in range(1,60):
                delta(grid,(256,),(np.int32(n),dcp,ddst,dcw,states[j],cache,
                                      dcaps,dvisual,devent,np.float64(p['scale']),
                                      np.float64(p['absolute_release_threshold']),a,b))
            return stream.end_capture()
        graphs={mode:capture(mode) for mode in ('null','event','continuous','all')}
        reset();stream.synchronize();stream.begin_capture()
        for j in range(1,60):
            seed(grid,(256,),(np.int32(n),dptr,dsrc,dw,variants['all'][j],
                               dcaps,dvisual,np.float64(p['scale']),a,b))
        graphs['full_csr']=stream.end_capture()
        times={}
        for mode in ('null','event','continuous','all','full_csr'):
            readings=[]
            for _ in range(p['timing_repetitions']):
                reset();stream.synchronize()
                first,last=cp.cuda.Event(),cp.cuda.Event()
                first.record(stream)
                graphs[mode].launch(stream)
                last.record(stream);last.synchronize()
                readings.append(float(cp.cuda.get_elapsed_time(first,last)))
            times[mode]={'ms_59':readings,'median_ms_per_query':float(np.median(readings)/59)}
    result={'schema':'csc_cost_split_result_v1','status':'COMPLETE_ISOLATED_TIMING',
            'plan_sha256':sha(plan_path),'times':times,
            'sparse_over_full_ratio':times['all']['median_ms_per_query']/times['full_csr']['median_ms_per_query'],
            'wall_s':time.monotonic()-started,
            'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'One real CNS graph/release cohort; CUDA Graph base E/I primitive only. No owner layers, integrator, MuJoCo or organism timing.'}
    b=p['budget']
    result['budget_ok']=result['wall_s']<=b['wall_seconds_max'] and result['maxrss_kib']<=b['ram_gib_max']*1024**2
    out.mkdir()
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    need((out/'RESULT.json').stat().st_size<=b['output_bytes_max'],'Output budget')
    return result


if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--out',type=Path,required=True);a=x.parse_args()
    r=run(a.out)
    print(json.dumps({'status':r['status'],'times':{k:v['median_ms_per_query']
          for k,v in r['times'].items()},'ratio':r['sparse_over_full_ratio'],
          'wall_s':r['wall_s'],'budget_ok':r['budget_ok']}))
