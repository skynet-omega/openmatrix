"""Compare scientific checkpoint fields for the fixed one-ms sham on/off gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ('time_ns','ticks','pending_sensors','pending_light','pending_excitation',
          'pending_proprioception','pending_cyborg_command','used_light','world',
          'body','hybrid','muscles','rotor','plasticity','proprioception')


def read(name):
    root=HERE/name/'final_state'
    return (json.loads((root/'session.json').read_text()),
            np.load(root/'session.npz',allow_pickle=False))


def compare(a,b,za,zb,path,differences):
    if isinstance(a,dict) and set(a)=={'__array__'}:
        if not isinstance(b,dict) or set(b)!={'__array__'}:
            differences.append(path);return
        x,y=za[a['__array__']],zb[b['__array__']]
        if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(
                x,y,equal_nan=x.dtype.kind in 'fc'):
            differences.append(path)
        return
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):
            differences.append(path);return
        for k in a:compare(a[k],b[k],za,zb,path+'/'+k,differences)
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):
            differences.append(path);return
        for i,(x,y) in enumerate(zip(a,b)):
            compare(x,y,za,zb,path+'/'+str(i),differences)
    elif a!=b:
        differences.append(path)


def main():
    off,on=read('smoke_off_01'),read('smoke_on_01')
    a,za=off;b,zb=on
    results={}
    for field in FIELDS:
        differences=[]
        compare(a[field],b[field],za,zb,field,differences)
        results[field]={'equal':not differences,'different_paths':differences[:20],
                        'different_path_count':len(differences)}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    same_files={name:sha(HERE/'smoke_off_01/final_state'/name)==sha(HERE/'smoke_on_01/final_state'/name)
                for name in ('published.npz','prosthesis.npz','boundary.json')}
    verdict=all(v['equal'] for v in results.values()) and all(same_files.values())
    result={'schema':'native_sideband_one_ms_on_off_v1','exact_scientific_state':verdict,
            'fields':results,'exact_auxiliary_files':same_files,
            'scope':'Selected full scientific state trees and three auxiliary checkpoint files; excludes run wall time and diagnostic buffers.'}
    (HERE/'SMOKE_COMPARE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'exact_scientific_state':verdict,
                      'different_fields':[k for k,v in results.items() if not v['equal']],
                      'auxiliary':same_files},ensure_ascii=False))
    if not verdict:raise SystemExit(2)


if __name__=='__main__':main()
