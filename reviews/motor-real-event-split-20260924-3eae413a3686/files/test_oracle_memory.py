"""GPU canary and A->B->A test for captured temporary ownership."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import cupy as cp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'native_hybrid_20260922/legacy_sources'))


def load(path):
    spec=importlib.util.spec_from_file_location('oracle_test_version',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.EffectiveOracle


def check(version):
    n=65536
    stream=cp.cuda.Stream(non_blocking=True)
    source=SimpleNamespace(general_transmission=lambda:None)
    brain=SimpleNamespace(_online_source=source,statistics={})
    def coefficient(z):
        temporary=z*z
        other=cp.roll(z,1)+temporary
        return .5+.2*cp.tanh(other), 70.+.01*temporary
    with stream:
        x=cp.linspace(0.,1.,n)
    stream.synchronize()
    core=SimpleNamespace(n=n,stream=stream,x=x,
                         project=lambda y,c,f:y.copy(),coefficient=coefficient)
    oracle=load(version)(SimpleNamespace(core=core,brain=brain))
    with stream:
        # These live allocations compete for any released captured temporary.
        sentinels=[cp.full(n,1234.+k) for k in range(24)]
        inputs=[x.copy(),x[::-1].copy(),x.copy()]
        expectations=[tuple(v.copy() for v in coefficient(y)) for y in inputs]
    stream.synchronize()
    equality=[]
    for y,expected in zip(inputs,expectations):
        z,a,r=oracle.query(y,0.)
        stream.synchronize()
        equality.append(bool(cp.array_equal(a,expected[0]) and cp.array_equal(r,expected[1]) and cp.array_equal(z,y)))
    intact=all(bool(cp.all(v==1234.+k)) for k,v in enumerate(sentinels))
    return {'canaries_intact':intact,'A_B_A_direct_equal':equality,
            'private_pool':hasattr(oracle,'pool')}


def main():
    out=HERE/'ORACLE_MEMORY_TEST_01.json'
    if out.exists():raise FileExistsError(out)
    old=check(HERE/'capture_01/executed_sources/effective_oracle.py')
    new=check(HERE/'effective_oracle.py')
    result={'schema':'oracle_memory_regression_v1','old':old,'fixed':new,
            'fixed_pass':new['canaries_intact'] and all(new['A_B_A_direct_equal']),
            'scope':'Synthetic allocation/interference regression, not organism validation'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['fixed_pass']:raise RuntimeError('Oracle lifetime regression')


if __name__=='__main__':main()
