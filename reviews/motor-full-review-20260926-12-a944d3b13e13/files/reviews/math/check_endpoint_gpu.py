"""Prepared GPU fixture for the coordinator; not executed by the math reviewer.

Two small projected systems and exact event-side checks. No organism loader,
neuronal simulation, historical source modification or external application.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
ROOT = HERE.parents[3]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing source literal")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wall-limit", type=int, default=40)
    args = p.parse_args()
    need(args.out.resolve().is_relative_to(HERE), "Output must remain in the math review directory")
    need(not args.out.exists(), "Fresh output required")
    need(1 <= args.wall_limit <= 120, "Bounded fixture wall budget required")
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("GPU fixture budget")))
    signal.alarm(args.wall_limit)
    import numpy as np
    import cupy as cp
    sys.path.insert(0, str(REVIEW/"engine"))
    from graph_runtime import GraphRK23
    port_path = ROOT/"campanas/etapa3_pn629_intervention_20260923_15/event_ports.py"
    kernel = cp.RawKernel(literal(port_path, "CODE"), "port", options=("--fmad=false",))

    class Port:
        def __init__(self, stop):
            self.qr=cp.asarray([1],dtype=cp.int64);self.sr=cp.asarray([2],dtype=cp.int64)
            self.q=cp.zeros(1);self.s=cp.zeros(1);self.tq=cp.asarray([.02]);self.ts=cp.asarray([.005])
            self.et=cp.asarray([stop]);self.ej=cp.ones(1);self.post=cp.ones(1)
            self.sets=cp.asarray([True]);self.counts=cp.asarray([1],dtype=cp.int32)
        def project(self,y,clock,fraction):
            out=y.copy()
            kernel((1,),(1,),(out,self.qr,self.sr,self.q,self.s,self.tq,self.ts,self.et,
                   self.ej,self.sets,self.post,self.counts,np.int32(1),np.int32(1),clock,np.float64(fraction)))
            return out

    results=[]
    cases=[("recorded_above",1.1461103097022343e-05,2.8348089914503676e-05,125000,1.),
           ("recorded_below",5.581565907498342e-06,5.825387811154045e-05,125000,1.),
           ("manufactured_accuracy_limit",1.1461103097022343e-05/128.,2.8348089914503676e-05/128.,1000,1000.)]
    for name,start,stop,ns,rate in cases:
        owner=Port(stop)
        def rhs(y,clock,fraction):
            return cp.stack((rate*(y[1]-y[0]),0.*y[1],0.*y[2]))
        g=GraphRK23(np.zeros(3),rhs,rtol=1e-5,atol=1e-7,project=owner.project,state_bounds=(0.,1.))
        try:
            # Directly exercise the actual endpoint kernel and unchanged port.
            with g.stream:
                g.clock.set(np.asarray([start,stop-start,stop]))
                g.endpoint_kernel((1,),(1,),(g.clock,g.left,g.right))
                l=owner.project(g.x,g.left,0.);r=owner.project(g.x,g.right,0.)
                left=l.get();right=r.get()
            g.stream.synchronize()
            need(left[1]==0. and right[1]==1.,"Wrong GPU event side: "+name)
            nxt,counts,error=g.advance(ns,ns,100,ns,boundaries=np.asarray([start,stop]))
            state=g.x.get(stream=g.stream)
            elapsed=ns*1e-9-stop
            exact=rate*(math.exp(-50.*elapsed)-math.exp(-rate*elapsed))/(rate-50.)
            need(counts[:2]==[3,0],"Unexpected rejection/missing boundary: "+name)
            need(abs(state[0]-exact)<1e-9,"Manufactured analytic endpoint differs: "+name)
            need(np.isfinite(state).all() and np.all((state>=0)&(state<=1)),"Invalid committed state")
            results.append(dict(case=name,counts=counts,max_error=error,next_ns=nxt,
                                exact_free_state=exact,free_state=float(state[0]),
                                free_abs_error=abs(float(state[0])-exact),
                                left_q=float(left[1]),right_q=float(right[1])))
        finally:
            g.close()
    signal.alarm(0)
    output=dict(status="PASS_ENDPOINT_GPU",wall_s=time.perf_counter()-started,cases=results,
                scope="Synthetic projected systems only; no organism integration",
                source_sha256={str(x):sha(x) for x in (Path(__file__),REVIEW/"engine/graph_runtime.py",
                    REVIEW/"engine/resident_controller.cu",REVIEW/"engine/libresident_controller.so",port_path)})
    args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
