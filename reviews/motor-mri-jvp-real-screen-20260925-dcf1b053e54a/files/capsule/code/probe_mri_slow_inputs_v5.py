"""Capture the five full MRI calls and sampled defect vectors on one real block.

The organism executes its original CNS. An independent post-step replay asks
whether five full effective-operator calls plus exact port projection suffice
when the cheap RHS holds the initial target/rate. This deliberately weak fast
provider tests the simplest possible MRI split; it is not a candidate engine.
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
PLAN=HERE/'MRI_SLOW_INPUTS_PLAN_60.json'
GPU_EDGES_LOWER=25_582_938


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


class FrozenFast:
    setup_edges=0  # No graph partition; oracle compilation is timed separately.
    def __init__(self,adapter,oracle,mask,version):
        import cupy as cp
        import event_ports as ep
        self.adapter=adapter;self.oracle=oracle;self.mask=mask;self.token=version
        self.anchor_a=None;self.anchor_r=None;self.full_calls=0
        self.fast_calls=0;self.project_calls=0
        self.full_log=[]
        code=ep.CODE
        old='const double*clock,double fraction)'
        need(code.count(old)==1 and code.count('et[p]<=t')==2 and
             code.count('mark>t')==1,'unsupported port source')
        code=code.replace(old,'const double*clock,double fraction,int right)')
        code=code.replace('et[p]<=t','(et[p]<t||(right&&et[p]==t))')
        code=code.replace('mark>t','(mark>t||(!right&&mark==t))')
        self.port_kernel=cp.RawKernel(code,'port',options=('--fmad=false',))
        self.port_kernel.compile()
    def version(self):return self.token
    def project(self,t,z,side):
        import cupy as cp
        need(side in ('left','right'),'projection side')
        self.project_calls+=1
        with self.oracle.stream:
            x=cp.asarray(z).copy()
            clock=cp.asarray([float(t),0.],dtype=cp.float64)
            p=self.adapter.ports
            self.port_kernel(((p.n+255)//256,),(256,),
                (x,p.qr,p.sr,p.q,p.s,p.tq,p.ts,p.times,p.jumps,p.sets,p.posts,
                 p.counts,np.int32(p.n),np.int32(p.width),clock,np.float64(0),
                 np.int32(side=='right')),stream=self.oracle.stream)
            out=cp.asnumpy(x,stream=self.oracle.stream)
        self.oracle.stream.synchronize()
        return out
    def full(self,t,z,side):
        import cupy as cp
        self.full_calls+=1
        with self.oracle.stream:
            x=cp.asarray(z)
            zz,aa,rr=self.oracle.query(x,float(t))
            d=rr*(aa-zz)
            derivative=cp.asnumpy(d,stream=self.oracle.stream)
            if self.anchor_a is None:
                self.anchor_a=cp.asnumpy(aa,stream=self.oracle.stream)
                self.anchor_r=cp.asnumpy(rr,stream=self.oracle.stream)
        self.oracle.stream.synchronize()
        need(np.isfinite(derivative).all(),'nonfinite full RHS')
        derivative[self.mask]=0.
        self.full_log.append((float(t),side,np.asarray(z,dtype=np.float64).copy(),
                              derivative.copy()))
        # A lower bound: specialized PN/KC/visual and setup are NOT counted.
        return derivative,GPU_EDGES_LOWER
    def fast(self,t,z,side):
        self.fast_calls+=1
        need(self.anchor_a is not None,'anchor missing')
        out=self.anchor_r*(self.anchor_a-z)
        out[self.mask]=0.
        need(np.isfinite(out).all(),'nonfinite fast RHS')
        return out,0  # No graph edges; local vector work is separately counted.


def run(out):
    started=time.monotonic()
    out=Path(out).resolve()
    need(__debug__ and not out.exists(),'normal Python/unique output required')
    plan=json.loads(PLAN.read_text())
    need(plan['schema']=='mri_slow_inputs_plan_v1','plan schema')
    for rel,expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
    import cupy as cp
    sys.path[:0]=[str(ROOT/'motor_nuevo/epoch_cost_20260923'),
                  str(ROOT/'motor_nuevo/pipeline_review_20260922'),
                  str(ROOT/'motor_nuevo/multirate_real_20260924_01'),
                  str(HERE/'chatgpt_original')]
    import run_set
    from runtime_session import RuntimeSession
    from effective_oracle import EffectiveOracle
    import mri_event_probe
    class AuditEnsayo(mri_event_probe.Ensayo):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.slow_log=[];self.flow_log=[]
        def slow(self,t,z,side='right'):
            result=super().slow(t,z,side)
            self.slow_log.append((float(t),side,result.copy()))
            return result
        def flow(self,*args,**kwargs):
            result=super().flow(*args,**kwargs)
            self.flow_log.append((result[0].copy(),
                                  None if result[1] is None else result[1].copy()))
            return result
    context={'physical_epochs':0,'replayed':0,'oracle_build_s':None,'diagnostic_error':None}
    old_init=RuntimeSession.__init__

    def installed(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        adapter=session.adapter
        old_build=adapter.build
        def build(drive,light):
            old_build(drive,light)
            t=time.perf_counter()
            oracle=EffectiveOracle(adapter)
            context['oracle_build_s']=time.perf_counter()-t
            g=adapter.core
            old_advance=g.advance
            context['g']=g;context['old_advance']=old_advance
            def advance(*aa,**kk):
                if context['active_epoch']!=1:return old_advance(*aa,**kk)
                need(context['replayed']==0 and aa and aa[0]==125000,
                     'unexpected accepted block')
                initial=cp.asnumpy(g.x)
                events=np.asarray(session.events.active.times,dtype=np.float64).copy()
                need(np.isfinite(events).all() and np.all((events>=0.)&(events<=125e-6)),
                     'event schedule')
                mask=np.zeros(len(initial),dtype=bool)
                mask[cp.asnumpy(adapter.ports.qr)]=True
                mask[cp.asnumpy(adapter.ports.sr)]=True
                event_identity=hashlib.sha256(events.tobytes()).hexdigest()
                parent=old_advance(*aa,**kk)
                reference=cp.asnumpy(g.x)
                context['replayed']=1
                reference_path=out/'MRI_REAL_BLOCK_STATES.npz'
                try:
                    model=FrozenFast(adapter,oracle,mask,event_identity)
                    trial=AuditEnsayo(
                        model,np.asarray(initial,dtype=np.float64),mask,
                        (-np.inf,np.inf),0.,125e-6,sorted(events.tolist()),
                        60*GPU_EDGES_LOWER,6*GPU_EDGES_LOWER)
                    probe_start=time.perf_counter()
                    verdict,high,low=trial.run(np.asarray(reference,dtype=np.float64))
                    probe_wall=time.perf_counter()-probe_start
                    need(len(model.full_log)==len(trial.slow_log)==5 and
                         len(trial.flow_log)==4,'MRI capture cardinality')
                    slow=np.stack([x[2] for x in trial.slow_log])
                    y1,y2=trial.flow_log[0][0],trial.flow_log[1][0]
                    h=125e-6
                    d1=slow[3]-(-slow[0]+2*slow[1])
                    d2=slow[4]-((-2*slow[1]+3*slow[2])+1.5*(slow[0]-slow[2]))
                    scale1=mri_event_probe.ATOL+mri_event_probe.RTOL*np.maximum(np.abs(y1),np.abs(y2))
                    scale2=mri_event_probe.ATOL+mri_event_probe.RTOL*np.maximum(np.abs(y2),np.abs(high))
                    score=np.maximum(h*np.abs(d1)/scale1,h*np.abs(d2)/scale2)
                    need(np.isfinite(score).all() and
                         abs(float(score.max())-verdict['sampled_defect_normalized'])<1e-10,
                         'defect score recomputation mismatch')
                    slow_path=out/'MRI_SLOW_INPUTS.npz'
                    np.savez_compressed(slow_path,
                        full_time=np.asarray([x[0] for x in model.full_log]),
                        full_side=np.asarray([x[1] for x in model.full_log],dtype='U5'),
                        full_z=np.stack([x[2] for x in model.full_log]),
                        full_f=np.stack([x[3] for x in model.full_log]),
                        slow_time=np.asarray([x[0] for x in trial.slow_log]),
                        slow_f=slow,flow_y1=y1,flow_y2=y2,high=high,
                        defect1=d1,defect2=d2,score=score,mask=mask)
                    np.savez_compressed(reference_path,initial=initial,reference=reference,
                                        high=high,low=low)
                    if hashlib.sha256(np.asarray(session.events.active.times,
                                                 dtype=np.float64).tobytes()).hexdigest()!=event_identity:
                        raise ValueError('event schedule mutated during replay')
                    context['mri']={'status':verdict['status'],'gates':verdict['gates'],
                        'endpoint_normalized':verdict['endpoint_normalized'],
                        'embedded_3_2':verdict['embedded_3_2'],
                        'sampled_defect_normalized':verdict['sampled_defect_normalized'],
                        'full_calls':verdict['full_calls'],'fast_calls':verdict['fast_calls'],
                        'edge_visits_lower_bound':verdict['edge_visits'],
                        'parent_edge_visits_lower_bound':verdict['parent_edge_visits'],
                        'project_calls':model.project_calls,'replay_wall_s':probe_wall,
                        'events':len(events),'mask_coordinates':int(mask.sum()),
                        'state_dimension':len(initial),'states_sha256':sha(reference_path),
                        'slow_inputs_sha256':sha(slow_path),
                        'sampled_defect_coordinates_gt1':int(np.count_nonzero(score>1)),
                        'sampled_defect_free_coordinates_gt1':int(np.count_nonzero((score>1)&~mask)),
                        'work_gate_valid':False,
                        'work_gate_note':'Graph-edge visits are lower bounds and ignore local-vector, specialized owner and setup work; wall is measured.'}
                except BaseException:
                    context['diagnostic_error']=traceback.format_exc()
                return parent
            g.advance=advance
        adapter.build=build
        physical_step=session.events.step
        def step(brain,ns,drive,light):
            context['active_epoch']=context['physical_epochs']
            context['physical_epochs']+=1
            return physical_step(brain,ns,drive,light)
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
    report={'schema':'mri_slow_inputs_result_v1','plan_sha256':sha(PLAN),
        'source_sha256':sha(Path(__file__)),'chatgpt_mri_sha256':sha(Path(mri_event_probe.__file__)),
        'run_result_sha256':sha(out/'RESULT.json'),'runner_exit_code':code,
        'runner_status':run_result['status'],'completed_trial_ms':run_result['completed_trial_ms'],
        'physical_epochs':context['physical_epochs'],'replayed_blocks':context['replayed'],
        'oracle_build_s':context['oracle_build_s'],'mri':context.get('mri'),
        'diagnostic_error':context['diagnostic_error'],
        'wall_s':time.monotonic()-started,
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'disk_bytes':sum(x.stat().st_size for x in out.rglob('*') if x.is_file()),
        'scope':'One sham125us block; five full MRI inputs and two sampled defect vectors archived for offline zone selection. Frozen diagonal fast remains negative; no state published.'}
    report['budget_ok']=(report['wall_s']<=plan['budget']['wall_s_max'] and
        report['maxrss_kib']<=plan['budget']['ram_gib_max']*1024**2 and
        report['disk_bytes']<=plan['budget']['disk_bytes_max'])
    report['status']=('COMPLETE_DIAGNOSTIC_ONLY' if (
        code==0 and report['budget_ok'] and report['mri'] is not None and
        report['diagnostic_error'] is None) else 'INCOMPLETE_RETAINED')
    (out/'MRI_REAL_FROZEN_FAST_RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','runner_status','physical_epochs',
                                            'replayed_blocks','wall_s','budget_ok')},allow_nan=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.out)
