"""Exact semantic comparison of serialized states, including RNG and pending data."""
from pathlib import Path
import hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(a,b,names=('session','prosthesis','published','effective_operator')):
    a,b=Path(a),Path(b);differences=[];count=0
    def different(path):
        nonlocal count
        count+=1
        if len(differences)<50:differences.append(path)
    def walk(x,y,zx,zy,path):
        if isinstance(x,dict) and set(x)=={'__array__'}:
            if not isinstance(y,dict) or set(y)!={'__array__'}:different(path);return
            u,v=zx[x['__array__']],zy[y['__array__']]
            if u.dtype.kind in 'fc' and not np.isfinite(u).all():raise ValueError('Nonfinite saved state '+path)
            if v.dtype.kind in 'fc' and not np.isfinite(v).all():raise ValueError('Nonfinite reconstructed state '+path)
            if u.dtype!=v.dtype or u.shape!=v.shape or not np.array_equal(u,v):different(path)
        elif isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y):different(path);return
            for k in x:walk(x[k],y[k],zx,zy,path+'/'+k)
        elif isinstance(x,list):
            if not isinstance(y,list) or len(x)!=len(y):different(path);return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,zx,zy,path+'/'+str(i))
        elif type(x)!=type(y) or x!=y:different(path)
    for name in names:
        x=json.loads((a/(name+'.json')).read_text(encoding='utf-8'));y=json.loads((b/(name+'.json')).read_text(encoding='utf-8'))
        with np.load(a/(name+'.npz'),allow_pickle=False) as zx,np.load(b/(name+'.npz'),allow_pickle=False) as zy:walk(x,y,zx,zy,name)
    if json.loads((a/'boundary.json').read_text(encoding='utf-8'))!=json.loads((b/'boundary.json').read_text(encoding='utf-8')):different('boundary')
    return {'exact':count==0,'different_path_count':count,'different_paths':differences,
            'compared_complete_trees':list(names)+['boundary'],
            'manifest_sha256':[sha(a/'MANIFEST.json'),sha(b/'MANIFEST.json')]}
