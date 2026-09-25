"""Analytic accuracy, aliased coefficient buffers, event sides and rollback."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import numpy as np
import cupy as cp
from graph_runtime import GraphMidpoint, KERNEL, LEFT, RIGHT
from event_projection import FilterPorts


def need(ok,message):
    if not ok:raise ValueError(message)


def check(library):
    # Exact logistic solution exercises an actually state-dependent model and
    # rejected steps. The model deliberately aliases its returned buffers.
    target=cp.ones(1);rate=cp.empty(1)
    def logistic(y,t,side):
        cp.multiply(y,1500.,out=rate)
        return target,rate
    core=GraphMidpoint([.1],logistic,rtol=1e-7,atol=1e-10,library=library)
    try:
        _,counts,_=core.advance(2000000,2000000,1,2000000)
        observed=float(core.x.get(stream=core.stream)[0])
        expected=1/(1+9*math.exp(-3))
        error=abs(observed-expected)
        need(error<=2e-5,'logistic analytic accuracy / shared-buffer ownership')
        need(counts[1]>0,'rejection branch was not exercised')
        core._gate.acquire()
        try:
            for action in (lambda:core.advance(1000,1000,1,1000),core.close):
                try:action()
                except RuntimeError as exc:need('already in use' in str(exc),'wrong ownership error')
                else:raise ValueError('overlapping operation accepted')
        finally:core._gate.release()
    finally:core.close()
    backing=cp.arange(4,dtype=cp.float64)
    try:GraphMidpoint([.1,.2],lambda y,t,s:(backing[::2],cp.ones_like(y)),rtol=1e-7,atol=1e-10,library=library)
    except ValueError as exc:need('contiguous' in str(exc),'wrong memory-layout error')
    else:raise ValueError('strided coefficients accepted by flat CUDA kernel')

    # An exact piecewise-constant relaxation crosses the awkward binary64
    # boundary from the external counterexample, inside a 62.5 us epoch.
    event=.5*float.fromhex('0x1.e4191e2c29091p-14')
    start=.5*float.fromhex('0x1.5d7c76b7dc13bp-15')
    need(math.nextafter(start+(event-start),start)==event,'counterexample changed')
    def piecewise(y,t,side):
        reached=(t>event)|((t==event)&(side==RIGHT))
        return cp.broadcast_to(cp.where(reached,.4,0.),y.shape),cp.full_like(y,2000.)
    core=GraphMidpoint([1.],piecewise,rtol=1e-7,atol=1e-10,library=library)
    try:
        core.advance(62500,20831,1,100000,boundaries=[event])
        v=float(core.x.get(stream=core.stream)[0])
        expected=math.exp(-2000*62.5e-6)+.4*(1-math.exp(-2000*(62.5e-6-event)))
        event_error=abs(v-expected)
        need(event_error<=3e-14,'piecewise analytic endpoint accuracy')
    finally:core.close()

    # The GPU uses the supplied endpoint, and distinguishes LEFT/RIGHT at the
    # very same floating point time. Coincident SET then ADD retains ordering.
    ports=FilterPorts(np.array([0]),np.array([1]))
    w=SimpleNamespace(q=np.array([.1]),s=np.array([.2]),tau=np.array([.01]),ts=.005,
        rows=np.array([0,0]),times=np.array([event,event]),jumps=np.array([.8,.2]),posts=np.array([.8,np.nan]))
    ports.update(w)
    times=cp.empty(5);flags=cp.zeros(2,dtype=cp.int32)
    module=cp.RawModule(code=KERNEL,options=('--std=c++17','--fmad=false'))
    module.get_function('stage_times')((1,),(1,),(cp.asarray([start,event-start,event]),times,flags))
    need(float(times[4].get())==event,'GPU reconstructed the authorized endpoint')
    left=ports.project(cp.zeros(2),times[4:5],LEFT).get()
    right=ports.project(cp.zeros(2),times[4:5],RIGHT).get()
    need(abs(left[0]-.1*math.exp(-event/.01))<1e-15,'LEFT consumed boundary event')
    need(right[0]==1.,'RIGHT lost SET/ADD ordering')
    need(abs(left[1]-right[1])<1e-15,'synaptic filter jumped at zero elapsed time')

    def invalid(y,t,side):
        return cp.where(t<1e-6,cp.zeros_like(y),cp.full_like(y,cp.nan)),cp.ones_like(y)
    core=GraphMidpoint([1.,2.],invalid,rtol=1e-7,atol=1e-10,library=library)
    try:
        initial=core.x.get(stream=core.stream)
        try:core.advance(4000,500,100,500,boundaries=[1e-6])
        except RuntimeError as exc:
            need('failure code 2' in str(exc),'wrong nonfinite diagnostic')
        else:raise ValueError('nonfinite model accepted')
        need(core.last_failure['clock'][0]>=1e-6,'rollback did not exercise partial progress')
        need(np.array_equal(core.x.get(stream=core.stream),initial),'epoch rollback not exact')
        need(core.calls==core.steps==0,'failed epoch changed committed counters')
    finally:core.close()
    return dict(status='PASS_RUNTIME_GUARDS',logistic_error=error,accepted=counts[0],rejected=counts[1],
        piecewise_error=event_error,gpu_authoritative_endpoint=True,explicit_sides=True,
        simultaneous_set_add=True,aliased_coefficients=True,rollback_exact=True,
        noncontiguous_rejected=True,overlapping_calls_rejected=True,
        library=str(library),python_optimized=not __debug__,scope='Analytic/transaction checks, not organism validation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.library.resolve()),allow_nan=False))
