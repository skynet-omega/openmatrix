"""One neutral full-organism step; effective 125-us operator trace and oracle.

Uses the historical runner unchanged. Separate CUDA buffers observe the first
accepted CNS block, including the actual complete target/rate after all model
overrides. No captured value feeds back into the organism.
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT/'motor_nuevo/architecture_round_20260924_01'
EPOCH = ROOT/'motor_nuevo/epoch_cost_20260923'
OUT = HERE/'capture_01'
BASE = PREV/'fusion_base_1ms_01'
SLOTS = 256

TRACE = r'''
extern "C" __global__ void mark(const long long*epoch,
 unsigned long long*count,const double*clock,double frac,double*times,
 double*starts,double*steps,double*fractions,int slots){
 if(blockIdx.x==0&&threadIdx.x==0&&epoch[0]==1){
  unsigned long long k=atomicAdd(count,1ULL);
  if(k<(unsigned long long)slots){times[k]=clock[0]+frac*clock[1];
   starts[k]=clock[0];steps[k]=clock[1];fractions[k]=frac;}
 }
}
extern "C" __global__ void capture(const long long*epoch,
 const unsigned long long*count,const double*x,const double*a,const double*r,
 double*xs,double*as,double*rs,int n,int slots){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 long long k=(long long)count[0]-1;
 if(epoch[0]==1&&i<n&&k>=0&&k<slots){
  long long p=k*(long long)n+i;xs[p]=x[i];as[p]=a[i];rs[p]=r[i];
 }
}
'''


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(p,v):
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def main():
    need(__debug__,'Historical runner requires normal Python')
    need(not OUT.exists(),'Unique capture output required')
    import cupy as cp
    sys.path[:0]=[str(EPOCH),str(HERE)]
    import run_set
    from runtime_session import RuntimeSession
    context={}
    old_load=run_set.load
    old_init=RuntimeSession.__init__
    old_coeff=None
    graph_class=None
    record={'schema':'effective_real_block_capture_v1','status':'STARTED',
            'gate_sha256':sha(PREV/'MULTIRATE_NEXT_GATE_01.json'),
            'clarification_sha256':sha(HERE/'GATE_CLARIFICATION_01.md'),
            'capture_source_sha256':sha(Path(__file__)),
            'oracle_source_sha256':sha(HERE/'effective_oracle.py'),
            'stage_admission':False,'solver_candidate_executed':False}
    started=time.perf_counter()

    def loaded(*args,**kwargs):
        v=old_load(*args,**kwargs);context['obj']=v[0]
        return v

    def installed(session,*args,**kwargs):
        nonlocal old_coeff,graph_class
        old_init(session,*args,**kwargs)
        from effective_oracle import EffectiveOracle
        from execution_parent_snapshot import evolving_state
        from session_io import write_state
        import graph_core,event_ports
        need(Path(event_ports.__file__).resolve()==EPOCH/'event_ports.py','Wrong event owner')
        record['event_ports_source_sha256']=sha(event_ports.__file__)
        adapter=session.adapter;b=adapter.brain;n=len(b.state)
        need(n==359373,'Unexpected full state dimension')
        context['session']=session
        module=cp.RawModule(code=TRACE,options=('--std=c++11','--fmad=false'))
        marker=module.get_function('mark');capture=module.get_function('capture')
        tag=cp.asarray([-1],dtype=cp.int64);count=cp.zeros(1,dtype=cp.uint64)
        ts,starts,steps,fractions=[cp.full(SLOTS,np.nan) for _ in range(4)]
        xs,aa,rr=[cp.empty((SLOTS,n),dtype=cp.float64) for _ in range(3)]
        context['buffers']=(tag,count,ts,starts,steps,fractions,xs,aa,rr)
        graph_class=graph_core.NativeGraph;old_coeff=graph_class.coeff
        def traced(self,y,frac):
            z=self.project(y,self.clock,frac) if self.project else y
            a,r=self.coefficient(z)
            marker((1,),(1,),(tag,count,self.clock,np.float64(frac),ts,starts,steps,fractions,np.int32(SLOTS)))
            capture(((n+255)//256,),(256,),(tag,count,z,a,r,xs,aa,rr,np.int32(n),np.int32(SLOTS)))
            self.check(self.grid,(256,),(z,a,r,np.int32(self.n),self.flag))
            return z,a,r
        graph_class.coeff=traced
        build=adapter.build
        def build_with_oracle(drive,light):
            build(drive,light)
            context['oracle']=EffectiveOracle(adapter)
        adapter.build=build_with_oracle
        physical_step=session.events.step
        calls=[]

        def owner_snapshot(folder):
            folder.mkdir(parents=True,exist_ok=False)
            write_state(folder/'evolving',evolving_state(b))
            write_state(folder/'pn',b._online_source.state_dict())
            write_state(folder/'published',{'rates':b.brain.rates,'time_ns':b.brain.time_ns,
                         'rng':b.brain.rng.bit_generator.state})

        def step(brain,ns,drive,light):
            epoch=len(calls)
            frame=inspect.currentframe()
            phase=None
            try:
                while frame is not None:
                    if frame.f_code.co_name=='cns' and 'accepted' in frame.f_locals:
                        phase='accepted' if frame.f_locals['accepted'] else 'predictor'
                        break
                    frame=frame.f_back
            finally:del frame
            need(phase is not None,'Unidentified coupling phase')
            entry={'epoch':epoch,'phase':phase,'start_clock_ns':int(b.time_ns),
                   'duration_ns':int(ns),'pn_clock_ns':int(b._online_source.time_ns)}
            calls.append(entry)
            tag.set(np.asarray([epoch],dtype=np.int64));cp.cuda.get_current_stream().synchronize()
            target=epoch==1
            if target:
                need(phase=='accepted' and ns==125000,'Wrong real replay block')
                owner_snapshot(OUT/'block_start')
                wave=session.events.active
                write_state(OUT/'block_events',{'rows':session.events.rows,
                    'q':wave.q,'s':wave.s,'tau':wave.tau,'ts':wave.ts,
                    'times':np.asarray(wave.times),'event_rows':np.asarray(wave.rows),
                    'jumps':np.asarray(wave.jumps),
                    'setop':np.isfinite(wave.posts),'posts':np.nan_to_num(wave.posts,nan=0.),
                    'duration_ns':ns})
                write_state(OUT/'block_inputs',{'drive':np.asarray(drive),'light':np.asarray(light),
                    'held_boundary_light':b.held_boundary_light,
                    'held_afferent_rate_hz':b.held_afferent_rate_hz,
                    'general_transmission':adapter.host_read(),'parameters':dict(b.parameters),
                    'norm_size':b._norm_size(),'next_step_ns':b.next_step_ns})
            result=physical_step(brain,ns,drive,light)
            if target:
                adapter.core.stream.synchronize()
                m=int(count.get()[0]);need(0<m<=SLOTS,'Incomplete/overflowed effective trace')
                entry['coefficient_queries']=m
                owner_snapshot(OUT/'block_end')
                checks=[];oracle=context['oracle']
                for j in sorted(set((0,m//2,m-1))):
                    t=float(ts[j].get())
                    z,a,r=oracle.query(xs[j],t)
                    oracle.stream.synchronize()
                    checks.append({'query':j,'time_s':t,
                        'projected_state_exact':bool(cp.array_equal(z,xs[j])),
                        'target_exact':bool(cp.array_equal(a,aa[j])),
                        'rate_exact':bool(cp.array_equal(r,rr[j])),
                        'target_max_abs':float(cp.max(cp.abs(a-aa[j]))),
                        'rate_max_abs':float(cp.max(cp.abs(r-rr[j])))})
                save(OUT/'ORACLE_CHECK.json',checks)
                need(all(x['projected_state_exact'] and x['target_exact'] and x['rate_exact'] for x in checks),
                     'Effective query graph differs from consumed operator')
                # The source and held buffers remain inside the accepted stage here.
                # Preserve all top-level GPU arrays and array-valued dictionaries;
                # this is an operator inventory, not a portable model serializer.
                arrays={};mapping={};seen={}
                def collect(path,value):
                    if isinstance(value,cp.ndarray):
                        key=(value.data.ptr,value.shape,value.dtype.str,value.strides)
                        if key not in seen:
                            dest='gpu_'+str(len(arrays));seen[key]=dest
                            arrays[dest]=cp.asnumpy(value)
                        mapping[path]=seen[key]
                    elif isinstance(value,dict):
                        for key,item in value.items():
                            if isinstance(key,str):collect(path+'/'+key,item)
                for name,value in b.__dict__.items():collect(name,value)
                for name in ('drive','light','pn','boundary','held'):collect('adapter/'+name,getattr(adapter,name))
                np.savez_compressed(OUT/'effective_gpu_arrays.npz',**arrays)
                save(OUT/'EFFECTIVE_ARRAY_PATHS.json',mapping)
                del arrays
                for name,value in (('state',xs),('target',aa),('rate',rr)):
                    data=cp.asnumpy(value[:m]);need(np.isfinite(data).all(),'Nonfinite coefficient trace')
                    np.savez_compressed(OUT/('trace_'+name+'.npz'),values=data)
                    del data
                np.savez_compressed(OUT/'trace_clock.npz',query_s=cp.asnumpy(ts[:m]),
                    trial_start_s=cp.asnumpy(starts[:m]),trial_step_s=cp.asnumpy(steps[:m]),
                    fraction=cp.asnumpy(fractions[:m]),epoch=np.full(m,epoch,dtype=np.int64),
                    phase=np.array(phase),global_start_ns=np.array(entry['start_clock_ns']))
                record.update(captured_epoch=entry,oracle_bitwise_queries=len(checks),
                              trace_queries=m,owner_snapshot_scope='evolving CNS/KC/APL/axon, PN source and published RNG/rates at block entry/exit',
                              effective_operator_scope='actual full target/rate plus effective GPU arrays; oracle graph validated at three real queried states',
                              portable_off_trajectory_runtime=False)
            save(OUT/'EPOCHS.json',calls)
            return result
        session.events.step=step

    run_set.load=loaded;RuntimeSession.__init__=installed
    sys.argv=['run_set.py','--out',str(OUT),'--odor','sham','--engine','causal_cuda',
              '--ms','1','--observe','off','--cuda-profile','off','--profile','off','--kc-capture','off']
    code=2
    try:
        code=run_set.main()
    finally:
        run_set.load=old_load;RuntimeSession.__init__=old_init
        if old_coeff is not None:graph_class.coeff=old_coeff
        record['wall_s']=time.perf_counter()-started
        record['runner_exit_code']=code
        record['status']='CAPTURED_PENDING_NEUTRALITY' if code==0 else 'FAILED_RETAINED'
        if OUT.exists():
            record['output_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if record['output_bytes']>2*1024**3:
                record['status']='FAILED_DISK_BUDGET';code=2
            save(OUT/'CAPTURE_RESULT.json',record)
            print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
