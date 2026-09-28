"""Explicit, peak-normalized conductance shape driven by counted spike events.

No rate cap, anatomical contact multiplier, release probability or depression
is inferred. Physical gain belongs to each declared connection outside this
unit-peak filter. Events supplied to advance occur at the interval start.
"""
import numpy as np
from dm1_pn_membrane import _clock

SCHEMA='biexponential_counted_events_v1'


class SynapticEventFilter:
    def __init__(self,source_ids,rise_ns,decay_ns,*,time_ns=0):
        ids=np.asarray(source_ids)
        if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<=0)
                or np.any(ids>np.iinfo(np.int64).max) or len(np.unique(ids))!=len(ids)):
            raise ValueError('Require unique integer source IDs')
        self.ids=ids.astype(np.int64)
        for value in [rise_ns,decay_ns]:
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
                raise ValueError('Time constants must be finite positive real nanoseconds')
        self.rise_ns=float(rise_ns);self.decay_ns=float(decay_ns)
        if self.rise_ns>=self.decay_ns:raise ValueError('Rise must precede decay')
        r,d=float(self.rise_ns),float(self.decay_ns)
        peak=np.log(d/r)/(1/r-1/d);self.norm=np.exp(-peak/d)-np.exp(-peak/r)
        self.time_ns=_clock(time_ns);self.fast=np.zeros(len(ids));self.slow=np.zeros(len(ids))

    def _counts(self,counts):
        n=np.asarray(counts)
        if n.shape!=self.fast.shape or n.dtype.kind not in 'iu' or np.any(n<0):
            raise ValueError('Require a nonnegative integer event count per source')
        return n

    def add_events(self,counts):
        """Add events at the current boundary without advancing the clock."""
        n=self._counts(counts);f=self.fast+n;s=self.slow+n
        if not np.isfinite(f+s).all():raise ValueError('Event state overflow')
        self.fast=f;self.slow=s

    def advance(self,dt_ns,counts):
        dt=_clock(dt_ns,positive=True);end=_clock(self.time_ns+dt);n=self._counts(counts)
        f=self.fast+n;s=self.slow+n
        midpoint=(s*np.exp(-dt/(2*self.decay_ns))-f*np.exp(-dt/(2*self.rise_ns)))/self.norm
        f=f*np.exp(-dt/self.rise_ns);s=s*np.exp(-dt/self.decay_ns)
        if not np.isfinite(f+s+midpoint).all():raise ValueError('Event state overflow')
        self.fast=f;self.slow=s;self.time_ns=end
        return midpoint

    def state_dict(self):
        return dict(schema=SCHEMA,source_ids=self.ids.copy(),rise_ns=self.rise_ns,decay_ns=self.decay_ns,
            time_ns=self.time_ns,fast=self.fast.copy(),slow=self.slow.copy())

    def load_state_dict(self,state):
        ids=np.asarray(state.get('source_ids'))
        if (state.get('schema')!=SCHEMA or ids.dtype.kind not in 'iu' or not np.array_equal(ids,self.ids)
                or state.get('rise_ns')!=self.rise_ns or state.get('decay_ns')!=self.decay_ns):
            raise ValueError('Wrong event filter identity or kinetics')
        clock=_clock(state.get('time_ns'));f=np.asarray(state.get('fast'),dtype=float);s=np.asarray(state.get('slow'),dtype=float)
        if f.shape!=self.fast.shape or s.shape!=f.shape or not np.isfinite(f+s).all() or np.any(f<0) or np.any(s<f):
            raise ValueError('Invalid conductance state')
        self.fast=f.copy();self.slow=s.copy();self.time_ns=clock
