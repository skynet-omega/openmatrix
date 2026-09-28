"""Open-loop canonical KCγ receiver for timestamped additional DM1 PN spikes.

Inherited background inputs are held; the new conductance is a perturbation
on existing pairs, not a second copy of the CNS or an admitted PN replacement.
"""
from pathlib import Path
import copy
import numpy as np
from dm1_pn_membrane import _clock
from kc_projected_batch import ProjectedKcBatch
from synaptic_event_filter import SynapticEventFilter
from session_io import read_state,sha256

SCHEMA='dm1_canonical_KCgamma_event_receiver_state_v1'


class Dm1KcEventReceiver:
    def __init__(self,source):
        self.source=Path(source);self.source_hashes={s:sha256(self.source.with_suffix(s)) for s in ['.json','.npz']}
        self.reference=read_state(self.source);r=self.reference
        if r['schema']!='canonical_dm1_KCgamma_open_loop_receiver_v1':raise ValueError('Wrong receiver artifact')
        self.time_ns=_clock(r['time_ns']);self.cell_ids=r['cell_ids'];self.pn_ids=r['PN_ids']
        self.filter=SynapticEventFilter(self.pn_ids,**r['kinetics'],time_ns=self.time_ns)
        self.batch=self._new_batch(r['membrane_state'])
        self.delivered_events=np.zeros(len(self.pn_ids),dtype=np.int64)
        self._source_index={int(v):j for j,v in enumerate(self.pn_ids)}

    def _new_batch(self,state):
        r=self.reference;p=r['membrane_parameters'];v=p['rest_mV']+r['membrane_state']['delta']@p['observation_basis'].T
        return ProjectedKcBatch(p,v[:,0],r['membrane_state']['q'],r['caps'],r['tau_s'],backend='numpy',state=state)

    def _interval(self,ns,cut):
        activation=self.filter.advance(ns,np.zeros(len(self.pn_ids),dtype=np.int64))
        r=self.reference;ge=r['background_ge_nS'].copy()
        if not cut:np.add.at(ge[:,1],r['post_slot'],r['peak_gain_nS']*activation[r['pre_slot']])
        self.batch.advance(ns,ge,r['background_gi_nS'],inner_step_ns=25_000)

    def advance(self,dt_ns,events,*,cut=False,partition_times_ns=()):
        """Consume each timestamp once in (current time, end time], in ns.

Intervals are split at event times; an event at the endpoint is saved in the
filter and cannot affect the past. Midpoint conductance is held for <=125 us.
"""
        ns=_clock(dt_ns,positive=True);end=_clock(self.time_ns+ns)
        if ns>125_000 or type(cut) is not bool:raise ValueError('Invalid receiver step or cut')
        e=np.asarray(events)
        if e.shape==(0,):e=np.empty((0,2),dtype=np.int64)
        if e.ndim!=2 or e.shape[1]!=2 or e.dtype.kind not in 'iu':raise ValueError('Events require integer [ID,time] pairs')
        if len(e) and (np.any(e[:,1]<=self.time_ns) or np.any(e[:,1]>end)
                or not np.isin(e[:,0],self.pn_ids).all() or len(np.unique(e,axis=0))!=len(e)):
            raise ValueError('Duplicate, unknown, past or future event')
        extra=np.asarray(partition_times_ns)
        if extra.size==0:extra=np.empty(0,dtype=np.int64)
        if extra.ndim!=1 or extra.dtype.kind not in 'iu' or np.any(extra<=self.time_ns) or np.any(extra>end):
            raise ValueError('Invalid common numerical partition')
        boundaries=np.unique(np.r_[e[:,1],extra])
        e=e[np.argsort(e[:,1],kind='stable')];before=self.state_dict();cursor=self.time_ns
        try:
            for clock in boundaries:
                clock=int(clock)
                if clock>cursor:self._interval(clock-cursor,cut)
                counts=np.zeros(len(self.pn_ids),dtype=np.int64)
                for id_ in e[e[:,1]==clock,0]:counts[self._source_index[int(id_)]]+=1
                if np.any(self.delivered_events>np.iinfo(np.int64).max-counts):raise OverflowError('Event counter overflow')
                self.filter.add_events(counts);self.delivered_events+=counts;cursor=clock
            if cursor<end:self._interval(end-cursor,cut)
            if not np.isfinite(self.batch.delta).all():raise FloatingPointError('Nonfinite KC response')
            self.time_ns=end
        except BaseException:
            self.load_state_dict(before);raise

    def state_dict(self):
        return dict(schema=SCHEMA,source_hashes=self.source_hashes.copy(),time_ns=self.time_ns,
            filter=self.filter.state_dict(),membrane=self.batch.state_dict(),delivered_events=self.delivered_events.copy())

    def load_state_dict(self,state):
        if state.get('schema')!=SCHEMA or state.get('source_hashes')!=self.source_hashes:raise ValueError('Wrong receiver source/schema')
        clock=_clock(state.get('time_ns'));r=self.reference
        if clock<r['time_ns']:raise ValueError('Receiver clock predates extraction')
        filt=SynapticEventFilter(self.pn_ids,**r['kinetics']);filt.load_state_dict(state['filter'])
        expected=r['membrane_state']['elapsed_ns']+clock-r['time_ns']
        if filt.time_ns!=clock or state['membrane']['elapsed_ns']!=expected:raise ValueError('Inconsistent membrane/filter clocks')
        counts=np.asarray(state.get('delivered_events'))
        if counts.dtype!=np.int64 or counts.shape!=self.delivered_events.shape or np.any(counts<0):raise ValueError('Invalid event count')
        batch=self._new_batch(state['membrane'])
        self.filter=filt;self.batch=batch;self.delivered_events=counts.copy();self.time_ns=clock
