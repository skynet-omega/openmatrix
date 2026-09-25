"""One same-run A/B/A probe of final coefficient readers of seven real events.

The organism executes the unchanged parent step. After that accepted block,
one event-port value at a time is perturbed in the private diagnostic stream,
queried through the complete effective coefficient stack, then restored.
"""
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
PLAN=HERE/'EFFECTIVE_EVENT_READER_PLAN_59.json'


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
    plan=json.loads(PLAN.read_text());need(plan['schema']=='effective_event_reader_plan_v1','plan')
    for rel,expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
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
        adapter=session.adapter
        old_build=adapter.build
        def build(drive,light):
            old_build(drive,light)
            oracle=EffectiveOracle(adapter)
            g=adapter.core
            context['g']=g;context['old_advance']=g.advance
            p=adapter.ports
            def advance(*aa,**kk):
                if context['active_epoch']!=1:return context['old_advance'](*aa,**kk)
                need(context['tested_blocks']==0 and aa and aa[0]==125000,
                     'wrong accepted block')
                parent=context['old_advance'](*aa,**kk)
                context['tested_blocks']=1
                try:
                    with oracle.stream:
                        state=cp.array(g.x)
                    oracle.stream.synchronize()
                    count=cp.asnumpy(p.counts,stream=oracle.stream)
                    qr=cp.asnumpy(p.qr,stream=oracle.stream)
                    times=cp.asnumpy(p.times,stream=oracle.stream)
                    jumps=cp.asnumpy(p.jumps,stream=oracle.stream)
                    posts=cp.asnumpy(p.posts,stream=oracle.stream)
                    sets=cp.asnumpy(p.sets,stream=oracle.stream)
                    slots=[(i,k) for i in range(p.n) for k in range(int(count[i]))]
                    need(len(slots)==len(session.events.active.times)==7,'seven event slots')
                    cone=json.loads((HERE/'EVENT_CONE_RESULT.json').read_text())
                    need(sorted(int(qr[i]) for i,k in slots)==cone['sources'],
                         'source rows mismatch')
                    need(np.allclose(sorted(float(times[i,k]) for i,k in slots),
                         sorted(session.events.active.times),atol=1e-15,rtol=0),
                         'event times mismatch')
                    with np.load(HERE/'ROW_EVENT_READER_ROWS_57.npz',allow_pickle=False) as z:
                        structural_first=z['first_hop_rows'].copy()
                        generic_first=z['first_hop_rows'][~z['listed_special']].copy()
                    t=float(plan['time_s'])
                    z0,a0,r0=oracle.query(state,t)
                    with oracle.stream:
                        z0,a0,r0=z0.copy(),a0.copy(),r0.copy()
                    oracle.stream.synchronize()
                    need(bool(cp.all(cp.isfinite(a0))) and bool(cp.all(cp.isfinite(r0))),
                         'nonfinite baseline')
                    max_a=np.zeros(len(a0),dtype=np.float64)
                    max_r=np.zeros(len(r0),dtype=np.float64)
                    changed_masks=[];records=[]
                    for i,k in slots:
                        key='post' if bool(sets[i,k]) else 'jump'
                        field=p.posts if key=='post' else p.jumps
                        before=float(posts[i,k] if key=='post' else jumps[i,k])
                        try:
                            with oracle.stream:field[i,k]=before+plan['delta']
                            oracle.stream.synchronize()
                            z1,a1,r1=oracle.query(state,t)
                            with oracle.stream:
                                z1,a1,r1=z1.copy(),a1.copy(),r1.copy()
                            oracle.stream.synchronize()
                            da=cp.asnumpy(a1-a0,stream=oracle.stream)
                            dr=cp.asnumpy(r1-r0,stream=oracle.stream)
                            need(np.isfinite(da).all() and np.isfinite(dr).all(),
                                 'nonfinite perturbation response')
                            changed=(da!=0)|(dr!=0)
                            changed_masks.append(changed)
                            np.maximum(max_a,np.abs(da),out=max_a)
                            np.maximum(max_r,np.abs(dr),out=max_r)
                            changed_cns=np.flatnonzero(changed[:166700])
                            records.append({'source_row':int(qr[i]),'port':int(i),
                                'slot':int(k),'time_s':float(times[i,k]),'kind':key,
                                'delta':float(plan['delta']),
                                'changed_state_coordinates':int(changed.sum()),
                                'changed_cns_rows':int(len(changed_cns)),
                                'changed_structural_first':int(np.isin(changed_cns,structural_first).sum()),
                                'changed_generic_first':int(np.isin(changed_cns,generic_first).sum()),
                                'max_abs_target':float(np.max(np.abs(da))),
                                'max_abs_rate':float(np.max(np.abs(dr)))})
                        finally:
                            with oracle.stream:field[i,k]=before
                            oracle.stream.synchronize()
                        z2,a2,r2=oracle.query(state,t)
                        oracle.stream.synchronize()
                        need(bool(cp.array_equal(z2,z0)) and bool(cp.array_equal(a2,a0))
                             and bool(cp.array_equal(r2,r0)),'A/B/A restoration failed')
                    masks=np.asarray(changed_masks,dtype=bool)
                    need(oracle.calls<=plan['budget']['oracle_queries_max'],
                         'diagnostic oracle query budget')
                    union=np.any(masks,axis=0)
                    union_cns=np.flatnonzero(union[:166700])
                    np.savez_compressed(out/'EFFECTIVE_EVENT_READERS.npz',
                        source_rows=np.asarray([x['source_row'] for x in records],dtype=np.int32),
                        changed_masks=masks,max_abs_target=max_a,max_abs_rate=max_r,
                        structural_first=structural_first,generic_first=generic_first)
                    context['probe']={'event_records':records,
                        'all_state_changed':int(union.sum()),
                        'cns_rows_changed':int(len(union_cns)),
                        'structural_first_changed':int(np.isin(union_cns,structural_first).sum()),
                        'generic_first_changed':int(np.isin(union_cns,generic_first).sum()),
                        'changed_outside_structural_first':int((~np.isin(union_cns,structural_first)).sum()),
                        'arrays_sha256':sha(out/'EFFECTIVE_EVENT_READERS.npz'),
                        'oracle_queries':oracle.calls,'all_seven_A_B_A_restored':True,
                        'scope':'Single endpoint-state coefficient perturbation, not a temporal integration or exact biological event intervention.'}
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
    result={'schema':'effective_event_reader_result_v1','plan_sha256':sha(PLAN),
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
    (out/'EFFECTIVE_EVENT_READER_RESULT.json').write_text(
        json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','tested_blocks','wall_s','budget_ok')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.out)
