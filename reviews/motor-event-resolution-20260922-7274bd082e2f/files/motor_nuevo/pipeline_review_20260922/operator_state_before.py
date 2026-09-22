"""Explicit portable effective-operator state, separate from physical state.

Bindings are provided by adapters; the registry knows neither neurons nor anatomy.
Restore values exactly, never re-run preparation or apply an intervention twice.
Call only at a quiescent epoch boundary, before capturing an execution graph.
"""
import hashlib
import numpy as np

def resolve(owner, path):
    for name in path:
        owner = owner[name] if isinstance(owner, dict) else getattr(owner, name)
    return owner

def host(value):
    return value.get() if hasattr(value, 'get') else np.asarray(value)

def fingerprint(value):
    a = np.ascontiguousarray(host(value))
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': hashlib.sha256(a.tobytes()).hexdigest()}

class OperatorState:
    @classmethod
    def from_state(cls, state):
        if set(state) != {'bindings','values','manifest'}:
            raise ValueError('Incomplete operator snapshot')
        obj = cls.__new__(cls)
        obj.bindings = {str(k):tuple(v) for k,v in state['bindings'].items()}
        if set(obj.bindings) != set(state['values']) or set(obj.bindings) != set(state['manifest']):
            raise ValueError('Operator identity/layout mismatch')
        obj.values = {k:np.array(v,copy=True) for k,v in state['values'].items()}
        obj.manifest = {k:dict(v) for k,v in state['manifest'].items()}
        for k,v in obj.values.items():
            if v.dtype.kind not in 'biuf' or not np.isfinite(v).all() or fingerprint(v)!=obj.manifest[k]:
                raise ValueError('Corrupt operator snapshot: '+k)
            v.flags.writeable=False
        return obj

    def state_dict(self):
        return {'bindings': {k:list(v) for k,v in self.bindings.items()},
                'values': {k:v.copy() for k,v in self.values.items()},
                'manifest': {k:dict(v) for k,v in self.manifest.items()}}

    def __init__(self, owner, bindings):
        self.bindings = {str(k): tuple(v) for k,v in bindings.items()}
        self.values = {k: np.array(host(resolve(owner,p)), copy=True)
                       for k,p in self.bindings.items()}
        for k,v in self.values.items():
            if v.dtype.kind not in 'biuf' or not np.isfinite(v).all():
                raise ValueError('Unsupported/nonfinite operator field: '+k)
            v.flags.writeable = False
        self.manifest = {k: fingerprint(v) for k,v in self.values.items()}

    def differences(self, owner):
        return {k: {'saved': self.manifest[k], 'current': fingerprint(resolve(owner,p))}
                for k,p in self.bindings.items()
                if fingerprint(resolve(owner,p)) != self.manifest[k]}

    def restore(self, owner):
        # Validate every destination before writing any field.
        for k,p in self.bindings.items():
            dst = resolve(owner,p); src = self.values[k]
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise ValueError('Operator layout changed: '+k)
            if fingerprint(src) != self.manifest[k]:
                raise ValueError('Operator snapshot corrupted: '+k)
        for k,p in self.bindings.items():
            dst = resolve(owner,p); src = self.values[k]
            if hasattr(dst,'set'): dst.set(src)
            else: np.copyto(dst,src)
        changed = self.differences(owner)
        if changed: raise RuntimeError('Operator restore mismatch: '+str(changed))

# Declared legacy adapter. Host and CUDA weights may intentionally differ;
# preserve both, rather than guessing which one contains an intervention.
LEGACY_BINDINGS = {
    'tau': ('tau',), 'theta': ('rate_theta',), 'gain': ('rate_gain',),
    'weights': ('weights64',), 'cuda_tau': ('cuda','tau'),
    'cuda_theta': ('cuda','theta'), 'cuda_gain': ('cuda','gain'),
    'cuda_weights': ('cuda','weights')}
