"""Experimental isolation at copied sensory buffers and physical boundaries."""
import hashlib
import json
from pathlib import Path
import numpy as np


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    a=np.ascontiguousarray(value)
    need(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'Invalid observed array')
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


class PairedInputs:
    def __init__(self,obj,session,donor,out,reference=None):
        self.obj,self.session,self.donor=obj,session,donor
        self.out=Path(out);self.rows=[];self.reference=None
        if reference is not None:
            self.reference=[json.loads(x) for x in Path(reference).read_text().splitlines()]
            need(len(self.reference)==240,'Incomplete paired input reference')
        self.original_event=session.events.step
        def event_step(b,ns,drive,light):
            self.original_event(b,ns,drive,light)
            import cupy as cp
            adapter=session.adapter
            self.adapter.append(dict(time_ns=int(b.time_ns),ns=int(ns),**{
                k:digest(cp.asnumpy(getattr(adapter,k))) for k in ('drive','light','boundary','held')}))
        session.events.step=event_step
        self.original_body=None;self.original_command=None

    def begin(self,phase,step):
        self.phase,self.step=phase,step;self.adapter=[];self.body=[];self.neural=[]

    def isolate_body(self):
        obj=self.obj
        need(obj.active and obj.withdrawals==1,'Expected already active physical prosthesis')
        self.original_command=obj.controller.set_command
        def command(*,forward_mm_s,yaw_rate_rad_s):
            need(self.phase=='ensayo','Body tape used outside trial')
            self.neural.append([float(forward_mm_s),float(yaw_rate_rad_s)])
            j=40+self.step-1
            self.original_command(forward_mm_s=float(self.donor['command_forward_mm_s'][j]),
                yaw_rate_rad_s=float(self.donor['command_yaw_rate_rad_s'][j]))
        obj.controller.set_command=command
        self.original_body=obj.body.advance
        def body(torque,nsteps=1):
            need(type(nsteps) is int and nsteps>0,'Physical step count')
            for _ in range(nsteps):
                need(obj.active and obj.withdrawals==1,'Physical ownership changed')
                self.original_body(torque,1)
                b=obj.body
                self.body.append(dict(step=int(b.steps),time=float(b.data.time),
                    integration=digest(b.integration_state()),force=digest(obj.last_force),
                    command=[float(obj.controller.forward_mm_s),float(obj.controller.yaw_rate_rad_s)]))
        obj.body.advance=body;obj.body_advance=body

    def finish(self):
        o=self.obj;c=o.core;h=c.hybrid
        # Every 125us accepted exchange has a discarded 62.5us predictor.
        # Observe both; their timestamps intentionally overlap after restore.
        need([x['ns'] for x in self.adapter]==[62500,125000]*8,'Coupling boundary count')
        if self.phase=='ensayo':
            need(len(self.body)>0 and len(self.neural)==len(self.body),'Missing physical observations')
        record=dict(phase=self.phase,step=self.step,adapter=self.adapter,body=self.body,
            sensory=digest(c.pending_sensors),
            proprioception={k:digest(v) for k,v in c.pending_proprioception.items() if isinstance(v,np.ndarray)},
            cxhp8={k:(digest(v) if isinstance(v,np.ndarray) else v) for k,v in h.cxhp8_pending.items()})
        index=len(self.rows)
        # Preserve evidence before refusing an unmatched interval.
        self.rows.append(record)
        with (self.out/'INPUTS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if self.reference is not None:
            need(record==self.reference[index],f'Physical or copied sensory input differs at {self.phase}:{self.step}')
        return np.asarray(self.neural[-1] if self.neural else [np.nan,np.nan])

    def close(self):
        self.session.events.step=self.original_event
        if self.original_body is not None:
            self.obj.body.advance=self.original_body;self.obj.body_advance=self.original_body
            self.obj.controller.set_command=self.original_command
