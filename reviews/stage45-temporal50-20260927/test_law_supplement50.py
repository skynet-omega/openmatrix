"""Repeat the reviewer's synthetic counterexamples; inspect no neural arms."""
import argparse,copy,json,sys,tempfile,time
from pathlib import Path
import numpy as np
from analyze50 import HERE,need,compute,read_json,save
from verify_dng_law50 import check_law
sys.path.insert(0,str(HERE/'aporte_motor50'))
from probe_analysis import fixture

def test():
    start=time.process_time();d,_=fixture();check_law(d);cases={}
    for name in ('negative_tau','illegal_positive_target','RHS_clock_shift','RHS_fraction','wrong_rate'):
        z={k:v.copy() for k,v in d.items()};f=z['records'][:,:128].reshape(-1,4,2,16)
        if name=='negative_tau':f[0,0,0,9]=-1
        elif name=='illegal_positive_target':f[0,0,0,[7,11]]=.5;f[0,0,0,13]=.5
        elif name=='RHS_clock_shift':f[0,0,0,14]+=1
        elif name=='RHS_fraction':f[0,1,0,15]=.4
        else:f[0,0,0,[8,12]]=2
        try:check_law(z)
        except ValueError as exc:cases[name]=dict(rejected=True,error=str(exc))
        else:raise ValueError('Synthetic corruption escaped '+name)
    with tempfile.TemporaryDirectory(prefix='criterion_test_',dir=HERE) as tmp:
        changed=read_json(HERE/'PLAN.json');changed['primary']['minimum_abs_neural_J']=0
        save(Path(tmp)/'PLAN.json',changed)
        try:compute(Path(tmp))
        except ValueError as exc:
            need(str(exc)=='Changed frozen criteria','Criterion failed for unrelated reason')
            cases['criterion']=dict(rejected=True,error=str(exc))
        else:raise ValueError('Changed criterion escaped')
    return dict(cases=cases,CPU_s=time.process_time()-start,new_neural_ms=0,partial_neural_results_inspected=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();v=test()
    if a.output:need(not a.output.exists(),'Preserve prior test');save(a.output,v)
    print(json.dumps(v))
