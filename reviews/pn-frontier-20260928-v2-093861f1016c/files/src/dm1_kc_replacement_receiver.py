"""External PN->KC replacement boundary preserving inherited synaptic tails.

Only the selected held DM1 contribution is removed. Its fast/slow filter state
is analytically mapped to the counted-event filter before any new events. APL
and all other held inputs remain. This is not yet a full CNS migration.
"""
import copy
from pathlib import Path
import numpy as np
from dm1_kc_event_receiver import Dm1KcEventReceiver, SCHEMA as BASE_SCHEMA
from session_io import read_state, sha256

SCHEMA = 'dm1_KCgamma_replacement_receiver_v1'


def cascade_to_events(x, y, gain_ratio, rise_ns, decay_ns, norm):
    """Map zero-future-input dx=-x/tr, dy=(x-y)/td to two exponentials.

gain_ratio is old rate-filter gain / unit-peak event gain, separately per PN.
It carries the existing rate-cap and kernel-area conversion, without fitting.
"""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);k=np.asarray(gain_ratio,dtype=float)
    if (x.ndim!=1 or y.shape!=x.shape or k.shape!=x.shape or not np.isfinite(x+y+k).all()
            or np.any(x<0) or np.any(y<0) or np.any(k<=0)
            or not np.isfinite([rise_ns,decay_ns,norm]).all() or not 0<rise_ns<decay_ns or norm<=0):
        raise ValueError('Invalid cascade migration')
    fast=norm*k*x*rise_ns/(decay_ns-rise_ns)
    slow=norm*k*y+fast
    if not np.isfinite(fast+slow).all():raise ValueError('Migration overflow')
    return fast,slow


class Dm1KcReplacementReceiver(Dm1KcEventReceiver):
    def __init__(self,source,migration):
        super().__init__(source);p=Path(migration);m=read_state(p)
        if m.get('schema')!='dm1_KC_filter_migration_v1' or m['receiver_hashes']!=self.source_hashes:
            raise ValueError('Migration belongs to another KC receiver')
        if m['time_ns']!=self.time_ns or not np.array_equal(m['PN_ids'],self.pn_ids):
            raise ValueError('Migration PN identity/time differs')
        self.source_hashes.update({'migration'+s:sha256(p.with_suffix(s)) for s in ['.json','.npz']})
        ge=np.asarray(m['background_without_DM1_ge_nS']);old=np.asarray(m['old_DM1_ge_nS'])
        if (ge.shape!=self.reference['background_ge_nS'].shape or old.shape!=ge.shape
                or not np.isfinite(ge+old).all() or np.any(ge<0) or np.any(old<0)):
            raise ValueError('Invalid conductance decomposition')
        np.testing.assert_allclose(ge+old,self.reference['background_ge_nS'],rtol=1e-12,atol=1e-13)
        fast,slow=cascade_to_events(m['old_fast'],m['old_slow'],m['gain_ratio'],
            self.filter.rise_ns,self.filter.decay_ns,self.filter.norm)
        state=self.filter.state_dict();state.update(fast=fast,slow=slow);self.filter.load_state_dict(state)
        actual=np.zeros_like(old)
        activation=(slow-fast)/self.filter.norm
        np.add.at(actual[:,1],self.reference['post_slot'],self.reference['peak_gain_nS']*activation[self.reference['pre_slot']])
        np.testing.assert_allclose(actual,old,rtol=1e-12,atol=1e-13)
        self.reference=copy.deepcopy(self.reference);self.reference['background_ge_nS']=ge.copy()

    def advance(self,dt_ns,events,*,cut=False,partition_times_ns=()):
        # Validate the caller's event list even when cutting its transmission.
        # The base class also validates the clock and common boundaries.
        if type(cut) is not bool:raise ValueError('Explicit cut flag required')
        e=np.asarray(events)
        if e.shape==(0,):e=np.empty((0,2),dtype=np.int64)
        if (e.ndim!=2 or e.shape[1]!=2 or e.dtype.kind not in 'iu'
                or np.any(e[:,1]<=self.time_ns) or np.any(e[:,1]>self.time_ns+dt_ns)
                or not np.isin(e[:,0],self.pn_ids).all() or len(np.unique(e,axis=0))!=len(e)):
            raise ValueError('Invalid replacement event list')
        partitions=np.asarray(partition_times_ns)
        if partitions.size==0:partitions=np.empty(0,dtype=np.int64)
        if cut:
            # Preserve the old filter tail. Cut new events only, with identical
            # numerical partitions in the paired control.
            if partitions.ndim!=1 or partitions.dtype.kind not in 'iu':raise ValueError('Invalid partitions')
            partitions=np.r_[partitions,e[:,1]]
            e=np.empty((0,2),dtype=np.int64)
        return super().advance(dt_ns,e,cut=False,partition_times_ns=partitions)

    def state_dict(self):
        s=super().state_dict();s['schema']=SCHEMA;return s

    def load_state_dict(self,state):
        if state.get('schema')!=SCHEMA:raise ValueError('Not a replacement receiver state')
        s=state.copy();s['schema']=BASE_SCHEMA;return super().load_state_dict(s)
