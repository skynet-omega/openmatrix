"""External site-resolved PN->KC boundary preserving a selected PN's old tail.

The extracted KC preparation and other inputs are held. Only the chosen PN's
future rate input is removed here, and its existing filter tail is retained.
This is not a current-CNS migration. Site gains are explicitly supplied; their
biological identification is a separate requirement from anatomical routing.
"""
import hashlib
import numpy as np
from dm1_kc_event_receiver import Dm1KcEventReceiver
from dm1_kc_replacement_receiver import cascade_to_events
from pn_calcium_release import SiteToTargetConductance
from session_io import read_state,sha256
from synaptic_event_filter import SynapticEventFilter
from pathlib import Path

SCHEMA='PN_site_KC_external_boundary_v1'

class PnSiteKcBoundary:
    def __init__(self,source,migration,pn_id,site_ids,relation_sites,target_ids,gain_nS,*,gain_provenance):
        base=Dm1KcEventReceiver(source);r=base.reference;m=read_state(migration)
        if (m.get('schema')!='dm1_KC_filter_migration_v1' or m['receiver_hashes']!=base.source_hashes
                or m['time_ns']!=base.time_ns or not np.array_equal(m['PN_ids'],base.pn_ids)
                or pn_id not in base.pn_ids or not isinstance(gain_provenance,str) or not gain_provenance):
            raise ValueError('Explicit matching receiver, migration, PN and gain provenance required')
        self.reference=r;self.batch=base.batch;self._new_batch=base._new_batch
        self.route=SiteToTargetConductance(site_ids,relation_sites,target_ids,gain_nS)
        selected=int(np.flatnonzero(base.pn_ids==pn_id)[0]);self.pair_mask=r['pre_slot']==selected
        destinations=r['cell_ids'][r['post_slot'][self.pair_mask]]
        if not np.array_equal(np.sort(destinations),self.route.target_ids):
            raise ValueError('Every selected canonical KC target must be represented exactly once in the target set')
        self.target_slot=np.array([int(np.flatnonzero(r['cell_ids']==x)[0]) for x in self.route.target_ids])
        self.time_ns=base.time_ns;self.legacy=SynapticEventFilter([pn_id],**r['kinetics'],time_ns=self.time_ns)
        fast,slow=cascade_to_events(m['old_fast'][selected:selected+1],m['old_slow'][selected:selected+1],
            m['gain_ratio'][selected:selected+1],self.legacy.rise_ns,self.legacy.decay_ns,self.legacy.norm)
        state=self.legacy.state_dict();state.update(fast=fast,slow=slow);self.legacy.load_state_dict(state)
        self.background_ge=r['background_ge_nS'].copy()
        self.old_at_handoff=self._legacy_ge((slow-fast)/self.legacy.norm)
        self.background_ge-=self.old_at_handoff
        if np.any(self.background_ge<0):raise ValueError('Selected PN subtraction exceeds inherited conductance')
        np.testing.assert_allclose(self.background_ge+self.old_at_handoff,r['background_ge_nS'],rtol=1e-13,atol=1e-14)
        h=hashlib.sha256(SCHEMA.encode()+str(pn_id).encode()+gain_provenance.encode())
        for key,value in sorted(base.source_hashes.items()):h.update(key.encode()+value.encode())
        for suffix in ['.npz','.json']:h.update(sha256(Path(migration).with_suffix(suffix)).encode())
        for a in [self.route.site_ids,self.route.slot,self.route.target_ids,self.route.target_slot,self.route.gain]:h.update(a.tobytes())
        self.identity=h.hexdigest();self.pn_id=int(pn_id)

    def _legacy_ge(self,activation):
        r=self.reference;g=np.zeros_like(r['background_ge_nS']);m=self.pair_mask
        np.add.at(g[:,1],r['post_slot'][m],r['peak_gain_nS'][m]*activation[0]);return g

    def advance(self,dt_ns,site_midpoint_activation):
        if type(dt_ns) is not int or not 0<dt_ns<=125000:raise ValueError('Receiver steps are integer ns up to 125us')
        addition=self.route.evaluate(site_midpoint_activation);before=self.state_dict()
        try:
            old=self.legacy.advance(dt_ns,np.zeros(1,dtype=np.int64))
            ge=self.background_ge+self._legacy_ge(old)
            np.add.at(ge[:,1],self.target_slot,addition)
            self.batch.advance(dt_ns,ge,self.reference['background_gi_nS'],inner_step_ns=min(dt_ns,25000))
            if not np.isfinite(self.batch.delta).all():raise FloatingPointError('Nonfinite KC response')
            self.time_ns+=dt_ns
        except BaseException:
            self.load_state_dict(before);raise

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,
                    legacy=self.legacy.state_dict(),membrane=self.batch.state_dict())

    def advance_calcium(self,chemistry,dt_ns,inward_calcium_pA,*,release_enabled=True):
        """Atomic two-component step using explicitly supplied local Ca flux.

        Midpoint release filters drive the inherited electrical KC model.
        No voltage-to-calcium conversion or PN membrane feedback is implied.
        """
        if (type(dt_ns) is not int or dt_ns<=0 or chemistry.time_ns!=self.time_ns
                or not np.array_equal(chemistry.ids,self.route.site_ids)):
            raise ValueError('Matched site identity/clock and positive integer ns step required')
        activation=chemistry.midpoint_activation(dt_ns,inward_calcium_pA,release_enabled=release_enabled)
        proposal=chemistry.preview(dt_ns,inward_calcium_pA,release_enabled=release_enabled)
        before=self.state_dict()
        try:
            self.advance(dt_ns,activation);chemistry.commit(proposal)
        except BaseException:
            self.load_state_dict(before);raise
        return proposal['release_increment']

    def load_state_dict(self,state):
        if state.get('schema')!=SCHEMA or state.get('identity')!=self.identity:raise ValueError('Wrong site receiver identity')
        clock=state.get('time_ns');r=self.reference
        if (type(clock) is not int or clock<r['time_ns'] or state['legacy']['time_ns']!=clock
                or state['membrane']['elapsed_ns']!=r['membrane_state']['elapsed_ns']+clock-r['time_ns']):
            raise ValueError('Inconsistent receiver clocks')
        legacy=SynapticEventFilter([self.pn_id],**r['kinetics']);legacy.load_state_dict(state['legacy'])
        batch=self._new_batch(state['membrane'])
        self.legacy=legacy;self.batch=batch;self.time_ns=clock
