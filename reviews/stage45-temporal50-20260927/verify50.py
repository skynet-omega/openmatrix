"""Portable recorded-evidence verification; does not run or restore the CNS."""
import argparse
from pathlib import Path
import json
import time
from analyze50 import HERE,compute,render,need,sha,read_json

def verify(folder=HERE,manifest=True):
    start=time.process_time();folder=Path(folder)
    if manifest:
        m=read_json(folder/'MANIFEST.json')
        for name,item in m['files'].items():
            path=folder/name
            need(path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Changed capsule member '+name)
    value=compute(folder);saved=read_json(folder/'RESULTADOS.json')
    a=dict(value);b=dict(saved)
    a.pop('analysis_CPU_s');b.pop('analysis_CPU_s')
    need(a==b,'Recomputed results or verdict differ')
    need(render(value)==(folder/'RESULTADOS.md').read_text(),'Generated report differs')
    return dict(scope='Reconstruction of all eight recorded arms, not full CNS replay',
                decision=value['decision'],arms=8,neural_ms_replayed=0,CPU_s=time.process_time()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=HERE);p.add_argument('--without-manifest',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.folder,not a.without_manifest)))
