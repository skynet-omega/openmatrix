"""Direct state comparison, respecting the historical codec's tuple/list rule."""
from pathlib import Path
import hashlib,json,math
import numpy as np

def compare_trees(a,b):
    counts={'arrays':0,'array_values':0,'scalar_leaves':0,'containers':0}
    differences=[]
    def bad(path):
        if len(differences)<50:differences.append(path)
    def walk(x,y,path):
        if isinstance(x,np.generic):x=x.item()
        if isinstance(y,np.generic):y=y.item()
        if isinstance(x,np.ndarray):
            counts['arrays']+=1;counts['array_values']+=x.size
            if not isinstance(y,np.ndarray) or x.shape!=y.shape or x.dtype!=y.dtype:
                bad(path);return
            if x.dtype.kind in 'fc' and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError('Nonfinite state '+path)
            if not np.array_equal(x,y):bad(path)
        elif isinstance(x,dict):
            counts['containers']+=1
            if not isinstance(y,dict) or set(x)!=set(y):bad(path);return
            for k in x:walk(x[k],y[k],path+'/'+k)
        elif isinstance(x,(list,tuple)):
            counts['containers']+=1
            if not isinstance(y,(list,tuple)) or len(x)!=len(y):bad(path);return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        else:
            counts['scalar_leaves']+=1
            for value in (x,y):
                if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite scalar '+path)
            if type(x)!=type(y) or x!=y:bad(path)
    walk(a,b,'')
    return {'exact':not differences,'different_paths':differences,**counts}

def qualify_initial(obj,source):
    from session_io import read_state
    from operator_state import OperatorState,LEGACY_BINDINGS
    source=Path(source)
    getters={'session':lambda:obj.core.state_dict(),'prosthesis':obj.state,
             'published':lambda:{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,
                                'rng':obj.core.brain.rng.bit_generator.state},
             'effective_operator':lambda:OperatorState(obj.core.hybrid,LEGACY_BINDINGS).state_dict()}
    trees={k:compare_trees(read_state(source/k),fn()) for k,fn in getters.items()}
    trees['boundary']=compare_trees(json.loads((source/'boundary.json').read_text()),obj.core.world.boundary.metadata())
    return {'exact':all(t['exact'] for t in trees.values()),'trees':trees,
            'method':'Direct recursive key/length/type/shape/dtype/value comparison; no structural hash',
            'checkpoint_manifest_sha256':hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()}
