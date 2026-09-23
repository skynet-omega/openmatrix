"""Failure-only readback of one KC event port; original failure is re-raised."""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import numpy as np

ROW=28296


def install_failure_capture(session, output: Path):
    import cupy as cp
    from graph_core import NativeGraph

    adapter=session.adapter
    rows=np.asarray(session.events.rows,dtype=np.int64)
    slots=np.flatnonzero(rows==ROW)
    if len(slots)!=1:
        raise ValueError('Failed KC row is not one unique event port')
    slot=int(slots[0])
    original=NativeGraph.advance
    if getattr(NativeGraph,'_axioma_failure_capture',False):
        raise RuntimeError('Failure capture already installed')

    def advance(g,*args,**kwargs):
        try:
            return original(g,*args,**kwargs)
        except RuntimeError as exc:
            if g is adapter.core and str(exc)=='accepted event state outside domain':
                try:
                    ports=adapter.ports
                    active=session.events.active
                    if active is None:
                        raise RuntimeError('Physical event owner already released')
                    def scalar(a):return float(cp.asnumpy(a).item())
                    count=int(cp.asnumpy(ports.counts[slot]).item())
                    q0=scalar(ports.q[slot]);s0=scalar(ports.s[slot])
                    tau=scalar(ports.tq[slot]);tau_s=scalar(ports.ts[0])
                    clock=cp.asnumpy(g.clock).tolist()
                    times=cp.asnumpy(ports.times[slot,:count]).tolist()
                    jumps=cp.asnumpy(ports.jumps[slot,:count]).tolist()
                    t=clock[0]+clock[1]
                    reconstructed=q0*math.exp(-t/tau)
                    for te,j in zip(times,jumps):
                        if te<=t:reconstructed+=j*math.exp(-(t-te)/tau)
                    wq,ws=active.at(t)
                    owner_ix=[i for i,r in enumerate(active.rows) if r==slot]
                    report={
                        'schema':'failed_kc_event_port_pre_rollback_v1',
                        'scope':'Failure-only diagnostic. No accepted endpoint or resumable checkpoint.',
                        'error':str(exc),'native_diagnostic':getattr(exc,'numerical_diagnostic',None),
                        'row':ROW,'event_slot':slot,'port_q_row':int(cp.asnumpy(ports.qr[slot]).item()),
                        'port_s_row':int(cp.asnumpy(ports.sr[slot]).item()),
                        'clock_s':clock,'query_time_s':t,
                        'port':{'q0':q0,'s0':s0,'tau_q_s':tau,'tau_s_s':tau_s,
                                'count':count,'times_s':times,'jumps':jumps,
                                'projected_q_cpu_formula':reconstructed},
                        'owner':{'q0':float(active.q[slot]),'s0':float(active.s[slot]),
                                 'tau_q_s':float(active.tau[slot]),'tau_s_s':float(active.ts),
                                 'duration_s':float(active.duration),
                                 'times_s':[float(active.times[i]) for i in owner_ix],
                                 'jumps':[float(active.jumps[i]) for i in owner_ix],
                                 'at_q_cpu':float(wq[slot]),'at_s_cpu':float(ws[slot])},
                        'attempted_fine_q':scalar(g.fine[ROW]),
                        'attempted_full_q':scalar(g.full[ROW]),
                    }
                    (output/'DOMAIN_CAPTURE.json').write_text(
                        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                except BaseException:
                    (output/'DOMAIN_CAPTURE_ERROR.json').write_text(
                        json.dumps({'traceback':traceback.format_exc()},ensure_ascii=False,indent=2)+'\n')
            raise

    NativeGraph.advance=advance
    NativeGraph._axioma_failure_capture=True
    def undo():
        NativeGraph.advance=original
        del NativeGraph._axioma_failure_capture
    return undo
