"""Focused GPU correctness checks for the event-only proposal recovery."""
from pathlib import Path
import sys,json,math
import numpy as np
import cupy as cp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'engine'))
from graph_runtime import GraphRK23

def need(ok,message):
    if not ok:raise ValueError(message)

def run(lib,events):
    def rhs(y,clock,fraction):return 100.*(.4-y)
    g=GraphRK23(np.array([.2]),rhs,rtol=1e-5,atol=1e-7,library=lib)
    try:
        nxt,counts,error=g.advance(100000,100000,100,100000,boundaries=events)
        value=float(g.x.get(stream=g.stream)[0])
        exact=.4+(.2-.4)*math.exp(-.01)
        need(abs(value-exact)<1e-9,'Analytic solution changed beyond fixture bound')
        return {'next_ns':nxt,'counts':counts,'max_error':error,'value':value}
    finally:g.close()

def main():
    parent=HERE.parent/'persistent_fp32_20260925_07/libresident_controller.so'
    candidate=HERE/'engine/libresident_controller.so'
    rows={}
    for name,events in [('none',[]),('terminal',[100000*1e-9]),('interior',[.5e-6,50e-6])]:
        a=run(parent,events);b=run(candidate,events);rows[name]={'control':a,'candidate':b}
        if name!='interior':need(a==b,'Non-interior-event behavior changed')
        else:need(b['counts'][0]<a['counts'][0] and a['counts'][1]==b['counts'][1]==0,'No legal reduction')
    # Reuse the model-independent event/rejection/partial-failure checks.
    import importlib.util
    path=HERE.parent/'neurocore_real_20260925_01/check_runtime.py'
    spec=importlib.util.spec_from_file_location('prior_runtime_guards',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows['rejection_rollback']=m.check(candidate)
    result={'status':'PASS_POLICY_GUARDS','python_optimized':not __debug__,
            'checks':rows,'scope':'Synthetic correctness only, no organism or speed claim'}
    (HERE/'FIXTURE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
