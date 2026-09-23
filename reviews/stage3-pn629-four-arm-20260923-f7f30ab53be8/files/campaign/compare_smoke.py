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
    verdicts={}
    for engine in ('smoke',):
        off,on=read(engine+'_off_01'),read(engine+'_on_01')
        a,za=off;b,zb=on
        results={}
        for field in FIELDS:
            differences=[]
            compare(a[field],b[field],za,zb,field,differences)
            results[field]={'equal':not differences,'different_paths':differences[:20],
                            'different_path_count':len(differences)}
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        same_files={name:sha(HERE/(engine+'_off_01')/'final_state'/name)==sha(HERE/(engine+'_on_01')/'final_state'/name)
                    for name in ('published.npz','prosthesis.npz','boundary.json')}
        verdict=all(v['equal'] for v in results.values()) and all(same_files.values())
        verdicts[engine]={'exact_scientific_state':verdict,'fields':results,'exact_auxiliary_files':same_files}
        za.close();zb.close()
    result={'schema':'kc_post_commit_capture_two_ms_neutrality_v1','engines':verdicts,
            'reference_engine_neutrality_tested':False,
            'scope':'Full child PN629-disabled state and auxiliary checkpoints at1ms; observer on/off, same biological intervention.'}
    (HERE/'CAPTURE_NEUTRALITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['exact_scientific_state'] for k,v in verdicts.items()}))
    if not all(v['exact_scientific_state'] for v in verdicts.values()):raise SystemExit(2)


if __name__=='__main__':main()
