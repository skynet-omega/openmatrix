"""Address ORN conductance to PN contacts without multiplying population gain.

Native contact coordinates are inputs. Equal splitting within each ORN->PN
pair and inherited population shares are explicit transfer hypotheses, not
measured conductance per EM contact. This module does not advance the CNS.
"""
import hashlib
import inspect
import numpy as np
from olfactory_synapse_candidate import FAST,SLOW,_advance_component

def _filters(values,caps):
    a=np.asarray(values); c=np.asarray(caps)
    if (a.dtype.kind not in 'fiu' or c.dtype.kind not in 'fiu' or c.ndim!=1
            or a.shape!=(4,len(c)) or not len(c) or not np.isfinite(a).all()
            or not np.isfinite(c).all() or np.any(c<=0) or np.any(a<0) or np.any(a>1)):
        raise ValueError('Finite [A_fast,A_slow,z_fast,z_slow] and positive source caps required')
    return a.astype(float,copy=False),c.astype(float,copy=False)

def preview_source_filters(values,caps_hz,rates_hz,duration_s):
    """Exact held-rate preview of inherited source filters; never commits them."""
    a,c=_filters(values,caps_hz); rates=np.asarray(rates_hz)
    if (rates.dtype.kind not in 'fiu' or rates.shape!=c.shape or not np.isfinite(rates).all()
            or np.any(rates<0) or np.any(rates>c) or isinstance(duration_s,(bool,np.bool_))
            or not np.isscalar(duration_s) or np.iscomplexobj(duration_s)
            or not np.isfinite(duration_s) or duration_s<0):
        raise ValueError('Explicit held rates within caps and a nonnegative duration required')
    out=np.empty_like(a)
    for channel,p in enumerate((FAST,SLOW)):
        area=p.k_ns_per_spike*p.tau_g_s
        for i in range(len(c)):
            A,g=_advance_component(a[channel,i],area*c[i]*a[channel+2,i],rates[i],duration_s,p)
            out[channel,i]=A;out[channel+2,i]=g/(area*c[i])
    return out

class SpatialOrnAllocation:
    """Conserving allocation for one PN, with every presynaptic ID retained."""
    def __init__(self,source_ids,contact_pre_ids,contact_nodes,pair_shares,*,full_size):
        ids=np.asarray(source_ids);pre=np.asarray(contact_pre_ids);nodes=np.asarray(contact_nodes)
        shares=np.asarray(pair_shares)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)==0 or np.any(ids<=0)
                or np.any(ids[1:]<=ids[:-1]) or pre.ndim!=1 or pre.dtype.kind not in 'iu'
                or nodes.shape!=pre.shape or nodes.dtype.kind not in 'iu' or not len(nodes)
                or type(full_size) is not int or full_size<=0 or np.any(nodes<0) or np.any(nodes>=full_size)
                or not np.array_equal(np.unique(pre),ids) or shares.dtype.kind not in 'fiu'
                or shares.shape!=ids.shape or not np.isfinite(shares).all() or np.any(shares<=0)
                or not np.isclose(shares.sum(),1.,rtol=0,atol=1e-12)):
            raise ValueError('Complete anatomical source/contact map and explicit shares summing to one required')
        def frozen(x,dtype):
            a=np.asarray(x,dtype=dtype);return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
        self.source_ids=frozen(ids,np.int64); self.contact_pre_ids=frozen(pre,np.int64)
        self.contact_nodes=frozen(nodes,np.int64); self.pair_shares=frozen(shares,np.float64)
        self.source_slot=frozen(np.searchsorted(ids,pre),np.int64)
        self.counts=frozen(np.bincount(self.source_slot,minlength=len(ids)),np.int64)
        unique,slots=np.unique(nodes,return_inverse=True)
        self.nodes=frozen(unique,np.int64);self.node_slot=frozen(slots,np.int64)
        self.full_size=full_size

    def spread(self,pair_conductance_nS):
        """Each pair total appears once, even with many contacts at one node."""
        g=np.asarray(pair_conductance_nS)
        if (g.dtype.kind not in 'fiu' or g.shape!=self.source_ids.shape
                or not np.isfinite(g).all() or np.any(g<0)):
            raise ValueError('Finite nonnegative pair conductances required')
        contact_g=g[self.source_slot]/self.counts[self.source_slot]
        return np.bincount(self.node_slot,weights=contact_g,minlength=len(self.nodes))

    def sample_filters(self,values,caps_hz):
        """Read inherited bounded source filters as the declared physical prior.

        The published gain is allocated once per population using pair_shares.
        This is not identification of the old CNS signed-weight current in nS.
        """
        a,c=_filters(values,caps_hz)
        if c.shape!=self.source_ids.shape:raise ValueError('Source/filter count mismatch')
        pair=self.pair_shares*c*(FAST.k_ns_per_spike*FAST.tau_g_s*a[2]
                                      +SLOW.k_ns_per_spike*SLOW.tau_g_s*a[3])
        return self.spread(pair)


class HeldRateOrnDrive:
    """Bench input with atomic source-filter/PN advancement at a shared clock.

    Rates are explicitly held during each step. This does not implement live
    CNS feedback or equate a separately prepared PN with the CNS history.
    Persist state_dict alongside the PN state; ionic and synaptic charge are
    distinct. Per-step synaptic charge is returned by the electrical solver.
    """
    def __init__(self,allocation,values,caps_hz,*,pn_identity,time_ns,preparation,reversal_mV=-10.):
        a,c=_filters(values,caps_hz)
        if (a.shape[1]!=len(allocation.source_ids) or type(time_ns) is not int or time_ns<0
                or not isinstance(pn_identity,str) or not pn_identity
                or not isinstance(preparation,str) or not preparation
                or not np.isscalar(reversal_mV) or np.iscomplexobj(reversal_mV)
                or not np.isfinite(reversal_mV)):
            raise ValueError('Explicit compatible source state, PN identity, clock and preparation required')
        self.allocation=allocation;self.values=a.copy();self.time_ns=time_ns
        self.caps_hz=np.frombuffer(c.astype(np.float64).tobytes(),dtype=np.float64)
        self.pn_identity=pn_identity;self.preparation=preparation;self.reversal_mV=float(reversal_mV)
        h=hashlib.sha256()
        for x in (allocation.source_ids,allocation.contact_pre_ids,allocation.contact_nodes,
                  allocation.pair_shares,self.caps_hz):h.update(x.tobytes())
        h.update(repr((allocation.full_size,pn_identity,preparation,self.reversal_mV,FAST,SLOW)).encode())
        h.update(inspect.getsource(preview_source_filters).encode())
        self.identity=h.hexdigest()

    def state_dict(self):
        return dict(identity=self.identity,pn_identity=self.pn_identity,time_ns=self.time_ns,
                    source_filters=self.values.copy())

    def load_state_dict(self,state):
        if (state.get('identity')!=self.identity or state.get('pn_identity')!=self.pn_identity
                or type(state.get('time_ns')) is not int or state['time_ns']<0):
            raise ValueError('Incompatible source state identity or clock')
        a,_=_filters(state['source_filters'],self.caps_hz)
        self.values=a.copy();self.time_ns=state['time_ns']

    def advance(self,pn,dt_ns,rates_hz,*,additional_synaptic_stages=None,**solver_options):
        from pn_coupled_ionic import GAMMA
        if (pn.identity!=self.pn_identity or pn.time_ns!=self.time_ns
                or len(pn.voltage)!=self.allocation.full_size or type(dt_ns) is not int or dt_ns<=0):
            raise ValueError('Matching prepared PN/source state and positive integer step required')
        if 'synaptic_stages' in solver_options:raise ValueError('The drive owns its synaptic stage inputs')
        samples=[preview_source_filters(self.values,self.caps_hz,rates_hz,c*dt_ns*1e-9)
                 for c in (GAMMA,1.)]
        stages=[dict(nodes=self.allocation.nodes,conductance_nS=self.allocation.sample_filters(a,self.caps_hz),
                     reversal_mV=self.reversal_mV) for a in samples]
        if additional_synaptic_stages is not None:
            # Extra receptors are explicitly presampled by their owner at the
            # same two stage times. Their kinetics/history are NOT inferred
            # from ORN filters or canonical signed weights. The caller commits
            # and persists that upstream state only after acceptance.
            from pn_local_conductance import combine_conductance_stages
            if not isinstance(additional_synaptic_stages,(list,tuple)) or len(additional_synaptic_stages)!=2:
                raise ValueError('Two explicitly presampled additional synaptic stages required')
            stages=[combine_conductance_stages([orn,extra])
                    for orn,extra in zip(stages,additional_synaptic_stages)]
        report=pn.advance(dt_ns,pn.cp.zeros_like(pn.voltage),synaptic_stages=stages,**solver_options)
        if report['accepted']:
            self.values=samples[1];self.time_ns+=dt_ns
        return report
