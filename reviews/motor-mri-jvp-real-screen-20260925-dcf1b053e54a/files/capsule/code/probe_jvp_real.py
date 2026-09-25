"""Six-call local linearization screen on one complete effective real-block RHS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'JVP_REAL_PLAN_62.json'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def run(out):
    start=time.monotonic();out=Path(out).resolve()
    need(__debug__ and not out.exists(),'normal Python and unique output required')
    plan=json.loads(PLAN.read_text());need(plan['schema']=='jvp_real_plan_v1','plan')
    lock=json.loads((HERE/'JVP_EXEC_LOCK_63.json').read_text())
    need(lock['schema']=='jvp_exec_lock_v1' and
         lock['plan_sha256']==sha(PLAN) and
         lock['source_sha256']==sha(Path(__file__)),'execution source lock')
    for rel,expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
    preflight=json.loads((HERE/'JVP_PREFLIGHT_62.json').read_text())
    need(preflight['status']=='READY_FOR_SIX_QUERIES' and
         preflight['plan_sha256']==sha(PLAN) and
         preflight['direction_sha256']==sha(HERE/'JVP_DIRECTION_62.npz'),
         'preflight incomplete')
    with np.load(HERE/'JVP_DIRECTION_62.npz',allow_pickle=False) as z:
        candidate=z['z'].copy();direction=z['v'].copy()
        archived=z['archived_full_f'].copy();mask=z['mask'].copy()
        t=float(z['time'][0]);side=str(z['side'][0])
    need(side=='right' and candidate.shape==direction.shape==archived.shape==mask.shape==(359373,),
         'direction layout')
    import cupy as cp
    sys.path[:0]=[str(ROOT/'motor_nuevo/epoch_cost_20260923'),
                  str(ROOT/'motor_nuevo/pipeline_review_20260922'),
                  str(ROOT/'motor_nuevo/multirate_real_20260924_01')]
    import run_set
    from runtime_session import RuntimeSession
    from effective_oracle import EffectiveOracle
    context={'physical_epochs':0,'tested_blocks':0,'diagnostic_error':None}
    old_init=RuntimeSession.__init__

    def installed(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        adapter=session.adapter;old_build=adapter.build
        def build(drive,light):
            old_build(drive,light)
            oracle=EffectiveOracle(adapter);g=adapter.core
            context['g']=g;context['old_advance']=g.advance
            def advance(*aa,**kk):
                if context['active_epoch']!=1:return context['old_advance'](*aa,**kk)
                need(context['tested_blocks']==0 and aa and aa[0]==125000,'wrong block')
                parent=context['old_advance'](*aa,**kk)
                context['tested_blocks']=1
                try:
                    points=(candidate,candidate+direction,candidate-direction,
                            candidate+.5*direction,candidate-.5*direction,candidate)
                    fields=[]
                    for point in points:
                        with oracle.stream:x=cp.asarray(point,dtype=cp.float64)
                        z,a,r=oracle.query(x,t)
                        with oracle.stream:f=cp.asnumpy(r*(a-z),stream=oracle.stream)
                        oracle.stream.synchronize()
                        need(np.isfinite(f).all(),'nonfinite full RHS')
                        f[mask]=0.
                        fields.append(f)
                    need(oracle.calls==plan['budget']['full_oracle_queries_max']==6,
                         'six full queries')
                    full=np.stack(fields)
                    need(np.array_equal(full[0],archived),
                         'same-run F(z) differs from archived stage')
                    need(np.array_equal(full[0],full[5]),
                         'F(z) repeat changed')
                    j1=(full[1]-full[2])/2
                    jhalf=full[3]-full[4]
                    scale=1e-7+1e-5*np.abs(candidate)
                    residual=125e-6*np.abs(j1-jhalf)/scale
                    need(np.isfinite(residual).all(),'nonfinite JVP discrepancy')
                    np.savez_compressed(out/'JVP_REAL_ARRAYS.npz',
                        z=candidate,v=direction,full_f=full,j1=j1,jhalf=jhalf,
                        normalized_discrepancy=residual,mask=mask,
                        time=np.asarray([t]))
                    score=float(np.max(residual[~mask]))
                    context['probe']={'time_s':t,'stage_index':plan['stage_index'],
                        'direction_sha256':sha(HERE/'JVP_DIRECTION_62.npz'),
                        'arrays_sha256':sha(out/'JVP_REAL_ARRAYS.npz'),
                        'full_queries':oracle.calls,'baseline_matches_archived':True,
                        'repeat_exact':True,'normalized_linearity_discrepancy':score,
                        'gate_le_0_1':score<=.1,
                        'coordinates_above_0_1':int(np.count_nonzero(residual[~mask]>.1)),
                        'scope':'One local finite-difference direction; does not prove Krylov efficiency or nonlinear long-horizon error.'}
                except BaseException:
                    context['diagnostic_error']=traceback.format_exc()
                return parent
            g.advance=advance
        adapter.build=build
        old_step=session.events.step
        def step(brain,ns,drive,light):
            context['active_epoch']=context['physical_epochs']
            context['physical_epochs']+=1
            return old_step(brain,ns,drive,light)
        session.events.step=step

    RuntimeSession.__init__=installed
    old_argv=sys.argv
    sys.argv=['run_set.py','--out',str(out),'--odor','sham','--engine','causal_cuda',
              '--ms','1','--observe','off','--cuda-profile','off','--profile','off',
              '--kc-capture','off']
    try:code=run_set.main()
    finally:
        sys.argv=old_argv;RuntimeSession.__init__=old_init
        if 'g' in context:context['g'].advance=context['old_advance']
    run_result=json.loads((out/'RESULT.json').read_text())
    result={'schema':'jvp_real_result_v1','plan_sha256':sha(PLAN),
        'source_sha256':sha(Path(__file__)),'run_result_sha256':sha(out/'RESULT.json'),
        'runner_exit_code':code,'runner_status':run_result['status'],
        'completed_trial_ms':run_result['completed_trial_ms'],
        'physical_epochs':context['physical_epochs'],'tested_blocks':context['tested_blocks'],
        'diagnostic_error':context['diagnostic_error'],'probe':context.get('probe'),
        'wall_s':time.monotonic()-start,
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'disk_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    result['budget_ok']=(result['wall_s']<=plan['budget']['wall_s_max'] and
        result['maxrss_kib']<=plan['budget']['ram_gib_max']*1024**2 and
        result['disk_bytes']<=plan['budget']['disk_bytes_max'])
    result['status']='COMPLETE_DIAGNOSTIC_ONLY' if (
        code==0 and result['budget_ok'] and result['probe'] is not None and
        result['diagnostic_error'] is None) else 'INCOMPLETE_RETAINED'
    (out/'JVP_REAL_RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','tested_blocks','wall_s','budget_ok')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.out)
