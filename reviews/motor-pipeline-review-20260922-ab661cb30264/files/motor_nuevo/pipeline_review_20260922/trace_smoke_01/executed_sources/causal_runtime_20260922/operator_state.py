"""Portable effective operator with adapter-bound identity and guarded restore.

Use at a quiescent boundary before graph capture. Values are not interventions:
restore them verbatim. A failed rollback invalidates the owner explicitly.
"""
import hashlib,json
import numpy as np
SCHEMA='effective_operator_v2'

def resolve(owner,path):
    for name in path:owner=owner[name] if isinstance(owner,dict) else getattr(owner,name)
    return owner

def host(value):return value.get() if hasattr(value,'get') else np.asarray(value)
def fingerprint(value):
    a=np.ascontiguousarray(host(value))
    return {'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def bindings_checked(bindings):
    if not isinstance(bindings,dict) or not bindings:raise ValueError('Empty operator bindings')
    out={}
    for k,p in bindings.items():
        if not isinstance(k,str) or not isinstance(p,(list,tuple)) or not p or not all(isinstance(x,str) and x and not x.startswith('_') for x in p):raise ValueError('Invalid operator binding')
        out[k]=tuple(p)
    if len(set(out.values()))!=len(out):raise ValueError('Aliased operator bindings')
    return out

def identity(bindings,manifest):
    data={'schema':SCHEMA,'bindings':bindings,'manifest':manifest}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _copy(dst,src):
    if hasattr(dst,'set'):dst.set(src)
    else:np.copyto(dst,src)
def _invalidate(owner):
    if isinstance(owner,dict):owner['_operator_restore_invalid']=True
    else:setattr(owner,'_operator_restore_invalid',True)

class OperatorState:
    @classmethod
    def from_state(cls,state,*,expected_bindings):
        if set(state)!={'schema','bindings','values','manifest','identity_sha256'} or state['schema']!=SCHEMA:raise ValueError('Incomplete/obsolete operator snapshot')
        expected=bindings_checked(expected_bindings);bindings=bindings_checked(state['bindings'])
        if bindings!=expected:raise ValueError('Operator routes differ from destination adapter')
        if set(bindings)!=set(state['values']) or set(bindings)!=set(state['manifest']):raise ValueError('Operator identity/layout mismatch')
        obj=cls.__new__(cls);obj.bindings=bindings;obj.values={k:np.array(v,copy=True) for k,v in state['values'].items()};obj.manifest={k:dict(v) for k,v in state['manifest'].items()}
        if identity(bindings,obj.manifest)!=state['identity_sha256']:raise ValueError('Operator schema/routes/values identity changed')
        for k,v in obj.values.items():
            if v.dtype.kind not in 'biuf' or not np.isfinite(v).all() or fingerprint(v)!=obj.manifest[k]:raise ValueError('Corrupt operator snapshot: '+k)
            v.flags.writeable=False
        return obj
    def state_dict(self):
        return {'schema':SCHEMA,'bindings':{k:list(v) for k,v in self.bindings.items()},'values':{k:v.copy() for k,v in self.values.items()},'manifest':{k:dict(v) for k,v in self.manifest.items()},'identity_sha256':identity(self.bindings,self.manifest)}
    def __init__(self,owner,bindings):
        self.bindings=bindings_checked(bindings);self.values={k:np.array(host(resolve(owner,p)),copy=True) for k,p in self.bindings.items()}
        for k,v in self.values.items():
            if v.dtype.kind not in 'biuf' or not np.isfinite(v).all():raise ValueError('Unsupported/nonfinite operator field: '+k)
            v.flags.writeable=False
        self.manifest={k:fingerprint(v) for k,v in self.values.items()}
    def differences(self,owner):
        changed={}
        for k,p in self.bindings.items():
            value=fingerprint(resolve(owner,p))
            if value!=self.manifest[k]:changed[k]={'saved':self.manifest[k],'current':value}
        return changed
    def restore(self,owner):
        destinations={k:resolve(owner,p) for k,p in self.bindings.items()}
        for k,dst in destinations.items():
            src=self.values[k]
            if dst.shape!=src.shape or dst.dtype!=src.dtype:raise ValueError('Operator layout changed: '+k)
            if isinstance(dst,np.ndarray) and not dst.flags.writeable:raise ValueError('Read-only operator destination: '+k)
            if fingerprint(src)!=self.manifest[k]:raise ValueError('Operator snapshot corrupted: '+k)
        # Cold reconstruction only: this backup is never in the numerical loop.
        previous={k:np.array(host(v),copy=True) for k,v in destinations.items()}
        try:
            for k,dst in destinations.items():_copy(dst,self.values[k])
            if self.differences(owner):raise RuntimeError('Operator transfer verification failed')
        except BaseException:
            restored=True
            for k,dst in destinations.items():
                try:
                    _copy(dst,previous[k])
                    if fingerprint(dst)!=fingerprint(previous[k]):restored=False
                except BaseException:restored=False
            if not restored:_invalidate(owner)
            raise

LEGACY_BINDINGS={'tau':('tau',),'theta':('rate_theta',),'gain':('rate_gain',),'weights':('weights64',),'cuda_tau':('cuda','tau'),'cuda_theta':('cuda','theta'),'cuda_gain':('cuda','gain'),'cuda_weights':('cuda','weights')}
