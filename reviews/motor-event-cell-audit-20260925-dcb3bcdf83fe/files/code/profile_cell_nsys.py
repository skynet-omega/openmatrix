"""Bound one real sham ms with CUDA Profiler API markers for Nsight Systems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'CELL_NSYS_PLAN_48.json'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out):
    start=time.monotonic()
    p=json.loads(PLAN.read_text())
    need(p['schema']=='cell_nsys_plan_v1' and not out.exists(),'Plan/output')
    for rel,expected in p['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'Source changed: '+rel)
    import cupy as cp
    sys.path[:0]=[str(ROOT/'motor_nuevo/epoch_cost_20260923'),
                  str(ROOT/'motor_nuevo/pipeline_review_20260922')]
    import run_set
    from runtime_session import RuntimeSession
    original_init=RuntimeSession.__init__
    context={'calls':0,'profile_started':False,'profile_stopped':False,
             'host_advance_ms':[],'owner_wall_delta_ms':[]}
    def installed(session,*args,**kwargs):
        original_init(session,*args,**kwargs)
        import device_cell
        cls=device_cell.DeviceCell
        old=cls.advance
        context['cls']=cls;context['old_advance']=old
        def timed(self,*a,**kw):
            if not context['profile_started']:
                cp.cuda.runtime.profilerStart()
                context['profile_started']=True
            owner_before=self.report['native_wall_s']
            t=time.perf_counter()
            result=old(self,*a,**kw)
            context['host_advance_ms'].append((time.perf_counter()-t)*1000)
            context['owner_wall_delta_ms'].append(
                (self.report['native_wall_s']-owner_before)*1000)
            context['calls']+=1
            if context['calls']==16:
                cp.cuda.runtime.profilerStop()
                context['profile_stopped']=True
            return result
        cls.advance=timed
    RuntimeSession.__init__=installed
    old_argv=sys.argv
    sys.argv=['run_set.py','--out',str(out),'--odor','sham','--engine','causal_cuda',
              '--ms','1','--observe','off','--cuda-profile','off','--profile','off',
              '--kc-capture','off']
    try:code=run_set.main()
    finally:
        sys.argv=old_argv;RuntimeSession.__init__=original_init
        if 'cls' in context:context['cls'].advance=context['old_advance']
        if context['profile_started'] and not context['profile_stopped']:
            cp.cuda.runtime.profilerStop()
    result=json.loads((out/'RESULT.json').read_text())
    report={'schema':'cell_nsys_run_result_v1','plan_sha256':sha(PLAN),
            'run_result_sha256':sha(out/'RESULT.json'),
            'status':'COMPLETE_DIAGNOSTIC_ONLY' if code==0 and result['status']=='COMPLETE' else 'INCOMPLETE',
            'runner_exit_code':code,'calls':context['calls'],
            'profile_started':context['profile_started'],
            'profile_stopped':context['profile_stopped'],
            'host_advance_ms':context['host_advance_ms'],
            'owner_wall_delta_ms':context['owner_wall_delta_ms'],
            'native_owner_wall_s':result['runtime']['cell']['native_wall_s'] if result['runtime'] else None,
            'wall_s':time.monotonic()-start,
            'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'disk_bytes':sum(x.stat().st_size for x in out.rglob('*') if x.is_file())}
    report['budget_ok']=(report['wall_s']<=p['budget']['wall_s_max'] and
                         report['maxrss_kib']<=p['budget']['ram_gib_max']*1024**2 and
                         report['disk_bytes']<=p['budget']['disk_bytes_max'])
    if not report['budget_ok']:report['status']='OVER_BUDGET'
    (out/'CELL_NSYS_RUN_RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    need(report['status']=='COMPLETE_DIAGNOSTIC_ONLY' and context['calls']==16 and
         context['profile_stopped'],'Run/profile coverage')
    print(json.dumps({k:report[k] for k in ('status','calls','native_owner_wall_s','wall_s')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.out)
