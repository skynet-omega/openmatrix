"""Read-only CUDA event timing around the real per-cell kernel in one sham ms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'CELL_KERNEL_PLAN_46.json'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(out):
    start=time.monotonic();p=json.loads(PLAN.read_text())
    need(p['schema']=='cell_kernel_timing_plan_v1' and not out.exists(), 'Plan/output')
    for rel,expected in p['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'Source changed: '+rel)
    import cupy as cp
    sys.path[:0]=[str(ROOT/'motor_nuevo/epoch_cost_20260923'),
                  str(ROOT/'motor_nuevo/pipeline_review_20260922')]
    import run_set
    from runtime_session import RuntimeSession
    old_init=RuntimeSession.__init__
    context={'timings_ms':[],'wrapped':0,'calls':0}
    def installed(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        import device_cell
        cls=device_cell.DeviceCell
        old_advance=cls.advance
        context['cls']=cls;context['old_advance']=old_advance
        def timed_advance(self,*a,**kw):
            if not getattr(self,'_axioma_timing_wrapped',False):
                kernel=self.kernel
                def timed_kernel(grid,block,arguments):
                    before=cp.cuda.Event();after=cp.cuda.Event()
                    before.record(self.stream)
                    kernel(grid,block,arguments)
                    after.record(self.stream)
                    self._axioma_last_events=(before,after)
                self.kernel=timed_kernel
                self._axioma_timing_wrapped=True
                context['wrapped']+=1
            result=old_advance(self,*a,**kw)
            before,after=self._axioma_last_events
            context['timings_ms'].append(float(cp.cuda.get_elapsed_time(before,after)))
            context['calls']+=1
            return result
        cls.advance=timed_advance
    RuntimeSession.__init__=installed
    old_argv=sys.argv
    sys.argv=['run_set.py','--out',str(out),'--odor','sham','--engine','causal_cuda',
              '--ms','1','--observe','off','--cuda-profile','off','--profile','off',
              '--kc-capture','off']
    try:code=run_set.main()
    finally:
        sys.argv=old_argv;RuntimeSession.__init__=old_init
        if 'cls' in context:context['cls'].advance=context['old_advance']
    result=json.loads((out/'RESULT.json').read_text())
    report={'schema':'cell_kernel_timing_result_v1','plan_sha256':sha(PLAN),
            'run_result_sha256':sha(out/'RESULT.json'),
            'status':'COMPLETE_DIAGNOSTIC_ONLY' if code==0 and result['status']=='COMPLETE' else 'INCOMPLETE',
            'runner_exit_code':code,'wrapped_owners':context['wrapped'],
            'calls':context['calls'],'per_call_kernel_ms':context['timings_ms'],
            'sum_kernel_s':sum(context['timings_ms'])/1000,
            'native_owner_wall_s':result['runtime']['cell']['native_wall_s'] if result['runtime'] else None,
            'step_wall_s':sum(json.loads(line)['step_wall_s'] for line in
                              (out/'PROGRESS.jsonl').read_text().splitlines())
                          if (out/'PROGRESS.jsonl').exists() else 0.,
            'wall_s':time.monotonic()-start,
            'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'disk_bytes':sum(x.stat().st_size for x in out.rglob('*') if x.is_file())}
    report['budget_ok']=(report['wall_s']<=p['budget']['wall_s_max'] and
                         report['maxrss_kib']<=p['budget']['ram_gib_max']*1024**2 and
                         report['disk_bytes']<=p['budget']['disk_bytes_max'])
    if not report['budget_ok']:report['status']='OVER_BUDGET'
    need(context['calls']==16 and context['wrapped']==1,'Kernel timing coverage')
    need(np.isfinite(context['timings_ms']).all() and min(context['timings_ms'])>0,
         'Nonfinite timing')
    (out/'CELL_KERNEL_TIMING_RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    need(report['status']=='COMPLETE_DIAGNOSTIC_ONLY','Run status/budget')
    print(json.dumps({k:report[k] for k in ('status','calls','sum_kernel_s',
                                           'native_owner_wall_s','step_wall_s','wall_s')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.out)
