"""The cold-state guard must reject value, dtype, schema and nonfinite corruption."""
from pathlib import Path
import json,tempfile
import numpy as np
from checkpoint_compare import compare
def write(p,*,value=1.,dtype='float64',flag=True):
    p.mkdir(exist_ok=True)
    for name in ('session','prosthesis','published','effective_operator'):
        (p/(name+'.json')).write_text(json.dumps({'array':{'__array__':'a'},'pending':flag,'rng':{'counter':3}}))
        np.savez_compressed(p/(name+'.npz'),a=np.asarray([value],dtype=dtype))
    (p/'boundary.json').write_text('{"clock":100}')
    (p/'MANIFEST.json').write_text('{}')
def main():
    tests=[]
    with tempfile.TemporaryDirectory(prefix='state_guard_') as directory:
        a=Path(directory)/'a';b=Path(directory)/'b';write(a);write(b)
        if not compare(a,b)['exact']:raise RuntimeError('Unchanged control rejected')
        for label,params in [('value',{'value':2.}),('dtype',{'dtype':'float32'}),('flag',{'flag':False}),('typed_flag',{'flag':1}),('nonfinite',{'value':float('nan')})]:
            write(b,**params)
            try:rejected=not compare(a,b)['exact']
            except ValueError:rejected=True
            if not rejected:raise RuntimeError('Corruption escaped '+label)
            tests.append(label)
    print(json.dumps({'unchanged_exact':True,'corruptions_rejected':tests,'GPU_loaded':False}))
if __name__=='__main__':main()
