"""Array and saved-state error metrics; no engine or historical imports."""

import json

import numpy as np

def need(ok,message):
    if not ok:raise ValueError(message)

def array_error(a,b):
    need(a.shape==b.shape and (a.dtype==b.dtype or a.dtype.kind==b.dtype.kind=='U'),
         'Array schema differs')
    exact=np.array_equal(a,b)
    if a.dtype.kind not in 'fciub':return {'exact':exact}
    need(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite comparison')
    # Integer differences must be formed before float conversion; otherwise
    # adjacent integers above2**53 can spuriously appear to differ by zero.
    d=np.asarray(a.astype(object)-b.astype(object),dtype=float) if a.dtype.kind in 'iu' else np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return {'exact':exact,'max_abs':float(np.max(np.abs(d))) if d.size else 0.,
            'rms':float(np.sqrt(np.mean(d*d))) if d.size else 0.,
            'changed_values':int(np.count_nonzero(a!=b)),'values':int(d.size)}

def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def saved_tree_errors(a,b):
    x=json.loads(a.with_suffix('.json').read_text());y=json.loads(b.with_suffix('.json').read_text())
    ax=load_arrays(a.with_suffix('.npz'));ay=load_arrays(b.with_suffix('.npz'));out={}
    def walk(u,v,path):
        if isinstance(u,dict) and set(u)=={'__array__'}:
            need(isinstance(v,dict) and set(v)=={'__array__'},'State tree schema differs')
            out[path]=array_error(ax[u['__array__']],ay[v['__array__']])
        elif isinstance(u,dict):
            need(isinstance(v,dict) and set(u)==set(v),'State dictionary differs')
            for k in u:walk(u[k],v[k],path+'/'+k)
        elif isinstance(u,list):
            need(isinstance(v,list) and len(u)==len(v),'State list differs')
            for i,(s,t) in enumerate(zip(u,v)):walk(s,t,path+'/'+str(i))
        else:
            for value in (u,v):
                if isinstance(value,float):need(np.isfinite(value),'Nonfinite scalar '+path)
            out[path]={'exact':type(u)==type(v) and u==v}
    walk(x,y,'')
    return out
