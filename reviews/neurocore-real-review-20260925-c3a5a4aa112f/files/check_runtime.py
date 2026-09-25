"""Supplemental ABI/event/rejection checks, not an organism performance test."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import cupy as cp
from graph_runtime import GraphRK23


def need(ok,message):
    if not ok:raise ValueError(message)


def check(library):
    # A new RHS needs no organism code or edits to the generic engine. A hard
    # input edge exercises the left-sided endpoint derivative. Large initial
    # steps deliberately exercise rejection, absent in the first real pair.
    edge=0.00073123456789
    def rhs(y,clock,fraction):
        t=clock[0]+clock[1]*fraction
        return -2000*y+cp.where(t>=edge,800.,0.)
    core=GraphRK23(np.array([1.]),rhs,rtol=1e-7,atol=1e-10,library=library)
    try:
        _,counts,_=core.advance(2000000,2000000,1,2000000,boundaries=[edge])
        observed=float(core.x.get(stream=core.stream)[0])
        expected=math.exp(-4)+.4*(1-math.exp(-2000*(.002-edge)))
        error=abs(observed-expected)
        need(error<=3e-8,'event/rejection accuracy')
        need(counts[1]>0,'rejection branch was not exercised')
    finally:core.close()
    # Rejection at the accuracy floor must restore the whole epoch, including
    # accepted earlier substeps. This is a local engine transaction check.
    def bad_rhs(y,clock,fraction):
        t=clock[0]+clock[1]*fraction
        return cp.where(t<1e-6,-y,cp.nan)
    rollback=GraphRK23(np.array([1.,2.]),bad_rhs,rtol=1e-7,atol=1e-10,library=library)
    try:
        original=rollback.x.get(stream=rollback.stream)
        try:rollback.advance(4000,500,100,500,boundaries=[1e-6])
        except RuntimeError as exc:
            need('failure code 4' in str(exc),'wrong rejection-floor diagnostic')
        else:raise ValueError('invalid RHS was accepted')
        need(np.array_equal(rollback.x.get(stream=rollback.stream),original),'failed epoch changed state')
        need(rollback.last_failure['clock'][0]>=.5e-6,'failure occurred before partial progress')
        need(rollback.calls==rollback.steps==0,'failed transaction changed committed counters')
    finally:rollback.close()
    return {'status':'PASS_RUNTIME_GUARDS','event_abs_error':error,
            'accepted':counts[0],'rejected':counts[1],'epoch_rollback_exact':True,
            'library':str(library),'python_optimized':not __debug__,
            'scope':'Supplemental synthetic correctness checks; actual coupled performance is in PAIR*.json.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.library.resolve()),allow_nan=False))
