"""A declared slow graded effector with an exact delayed linear-input cascade.

This numerical instrument does not identify a transmitter, efficacy, receptor
mixture, or a G-protein mechanism. Both time constants must be supplied with
provenance. A downstream local conductance map owns the physical amplitude.
"""
import copy,hashlib,json
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from pn_apl_graded_receptor import DelayedGradedReceptor
from pn_coupled_ionic import GAMMA

SCHEMA='PN_delayed_graded_cascade_v1'


@lru_cache(maxsize=128)
def transition(h_s,tau_input_s,tau_effector_s):
    a=h_s/tau_input_s;b=h_s/tau_effector_s
    # Last two coordinates are the driver and its change across this interval.
    A=np.array([[-a,0.,a,0.],[b,-b,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]])
    p=expm(A);p.setflags(write=False);return p


class DelayedCascadeReceptor:
    def __init__(self,names,*,initial_driver,time_ns,tau_input_s,tau_effector_s,delay_ns,provenance):
        if not np.isscalar(tau_effector_s) or not np.isfinite(tau_effector_s) or tau_effector_s<=0:
            raise ValueError('Positive declared effector time constant required')
        self.driver=DelayedGradedReceptor(names,initial_driver=initial_driver,time_ns=time_ns,
            tau_s=tau_input_s,delay_ns=delay_ns,provenance=provenance)
        self.tau_effector_s=float(tau_effector_s);self.occupancy=np.zeros(len(self.driver.names))
        self.identity=hashlib.sha256(json.dumps(dict(schema=SCHEMA,driver=self.driver.identity,
            tau_effector_s=self.tau_effector_s),sort_keys=True).encode()).hexdigest()

    @property
    def time_ns(self):return self.driver.time_ns

    def _at(self,offset_ns,history):
        x=self.driver.occupancy.copy();y=self.occupancy.copy();cursor=0.;stop=float(offset_ns)
        def evolve(h_ns,r0,change):
            nonlocal x,y
            p=transition(h_ns*1e-9,self.driver.tau_s,self.tau_effector_s)
            x,y=p[0,0]*x+p[0,1]*y+p[0,2]*r0+p[0,3]*change,p[1,0]*x+p[1,1]*y+p[1,2]*r0+p[1,3]*change
        for s in history:
            start=s['start_ns']-self.time_ns;end=s['end_ns']-self.time_ns
            if end<=cursor:continue
            if start>cursor:
                gap=min(float(start),stop)-cursor
                if gap>0:evolve(gap,0.,0.);cursor+=gap
            if cursor>=stop:break
            right=min(float(end),stop);left=max(cursor,float(start))
            if right<=left:continue
            change=s['right']-s['left'];r0=s['left']+change*((left-start)/(end-start))
            evolve(right-left,r0,change*((right-left)/(end-start)));cursor=right
        if cursor<stop:evolve(stop-cursor,0.,0.)
        return self.driver._driver(y)

    def preview(self,dt_ns,before,after):
        _,driver_next=self.driver.preview(dt_ns,before,after)
        history=copy.deepcopy(self.driver.history)
        history.append(dict(start_ns=self.time_ns+self.driver.delay_ns,end_ns=self.time_ns+dt_ns+self.driver.delay_ns,
            left=self.driver._driver(before),right=self.driver._driver(after)))
        stages=[self._at(c*dt_ns,history) for c in (GAMMA,1.)]
        return stages,dict(schema=SCHEMA,identity=self.identity,driver=driver_next,occupancy=stages[-1].copy())

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,driver=self.driver.state_dict(),occupancy=self.occupancy.copy())

    def load_state_dict(self,state):
        if set(state)!={'schema','identity','driver','occupancy'} or state['schema']!=SCHEMA or state['identity']!=self.identity:
            raise ValueError('Wrong graded cascade identity')
        self.driver.validated(state['driver']);y=self.driver._driver(state['occupancy'])
        if state['driver']['time_ns']==self.driver.origin_ns and np.any(y):raise ValueError('Effector predates explicit activation')
        self.driver.load_state_dict(state['driver']);self.occupancy=y
