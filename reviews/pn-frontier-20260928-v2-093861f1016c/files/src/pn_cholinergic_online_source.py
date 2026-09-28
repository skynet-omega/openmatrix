"""Add declared cholinergic pair-rate filters to the existing live PN source.

Inherited PN/Ca/ORN/output histories remain exact. Additional receptor history
starts at zero on explicit activation of these routes, not at an inferred rest.
"""
import copy,hashlib
import numpy as np
from pn_online_orn_source import OnlineOrnPnSource,SCHEMA as ORN_SCHEMA
from pn_cholinergic_rate_prior import CholinergicPairRatePrior,REVERSAL_MV
from pn_local_conductance import LocalPairConductance,combine_conductance_stages
from pn_cns_orn_stages import canonical_orn_stages
from pn_prepared_kc_boundary import zero_release_tail
from pn_coupled_ionic import GAMMA

SCHEMA='PN_cholinergic_online_source_v1'


class CholinergicOnlineSource(OnlineOrnPnSource):
    @classmethod
    def adopt(cls,base,mapping,*,caps_hz,connected,provenance,origin_ns=None):
        if type(base) is not OnlineOrnPnSource or type(connected) is not bool:
            raise ValueError('Exact live ORN source and explicit ACh current connection required')
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.ach_map=LocalPairConductance(**mapping)
        origin=base.time_ns if origin_ns is None else origin_ns
        if type(origin) is not int or origin<0 or origin>base.time_ns:raise ValueError('Receptor origin must precede or equal current source clock')
        obj.ach=CholinergicPairRatePrior(obj.ach_map.source_ids,time_ns=origin,provenance=provenance)
        if obj.ach_map.full_size!=len(base.pn.voltage) or np.intersect1d(obj.ach_map.source_ids,base.allocation.source_ids).size:
            raise ValueError('ACh sites must be on the same PN and exclude existing ORN pairs')
        if np.any(base.pn.C[obj.ach_map.nodes]<=0):raise ValueError('ACh receptors require physical membrane capacitance')
        caps=np.asarray(caps_hz,dtype=float)
        if caps.shape!=obj.ach.source_ids.shape or not np.isfinite(caps).all() or np.any(caps<=0):raise ValueError('Named inherited rate caps required')
        obj.ach_caps=np.frombuffer(caps.tobytes(),dtype=float);obj.ach_connected=connected;obj.ach_charge_pC=0.;obj.orn_charge_pC=0.
        obj._orn_identity=base.identity
        h=hashlib.sha256((SCHEMA+base.identity+obj.ach.identity+obj.ach_map.identity+str(connected)).encode());h.update(caps.tobytes());obj.identity=h.hexdigest()
        return obj

    def _frame_rates(self,frame):
        ids=np.asarray(frame['source_ids']);rates=np.asarray(frame['source_legacy_rate_proxy_hz'])
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or rates.shape!=ids.shape or rates.dtype.kind not in 'fiu' or len(np.unique(ids))!=len(ids):raise ValueError('Complete named canonical source frame required')
        order=np.argsort(ids);pos=np.searchsorted(ids[order],self.ach.source_ids)
        if np.any(pos>=len(ids)) or not np.array_equal(ids[order][pos],self.ach.source_ids):raise ValueError('Missing cholinergic source')
        rate=rates[order][pos]
        if not np.isfinite(rate).all() or np.any(rate<0) or np.any(rate>self.ach_caps):raise ValueError('Rate proxy outside inherited canonical cap')
        return rate

    def state_dict(self):
        base=OnlineOrnPnSource.state_dict(self);base['identity']=self._orn_identity
        return dict(schema=SCHEMA,identity=self.identity,base=base,ach=self.ach.state_dict(),ach_charge_pC=self.ach_charge_pC,orn_charge_pC=self.orn_charge_pC)

    def load_state_dict(self,saved):
        if (set(saved)!={'schema','identity','base','ach','ach_charge_pC','orn_charge_pC'} or saved['schema']!=SCHEMA or saved['identity']!=self.identity
            or saved['base']['identity']!=self._orn_identity or saved['base']['schema']!=ORN_SCHEMA
            or saved['base']['time_ns']!=saved['ach']['time_ns']):raise ValueError('Wrong joint source identity or receptor clock')
        x,y=self.ach._validated(saved['ach']);qa,qorn=saved['ach_charge_pC'],saved['orn_charge_pC']
        if not np.isscalar(qa) or not np.isscalar(qorn) or not np.isfinite([qa,qorn]).all():raise ValueError('Finite receptor charge histories required')
        base=dict(saved['base'],identity=self.identity)
        OnlineOrnPnSource.load_state_dict(self,base)
        self.ach.fast=x;self.ach.slow=y;self.ach.time_ns=saved['ach']['time_ns'];self.ach_charge_pC=float(qa);self.orn_charge_pC=float(qorn)

    def advance(self,dt_ns,before,after,*,connected=True,**solver_options):
        if (before.get('time_ns')!=self.time_ns or self.ach.time_ns!=self.time_ns
            or self.pn.time_ns-self.source_origin_ns!=self.time_ns-self.cns_origin_ns):
            raise ValueError('CNS/receptor/source interval disagreement')
        if 'stage_observation_nodes' in solver_options:raise ValueError('Receptor source owns its charge observation nodes')
        orn=canonical_orn_stages(self.allocation,self.caps,before,after,dt_ns,connected=connected)
        g,ach_next=self.ach.preview(dt_ns,self._frame_rates(before),self._frame_rates(after))
        ach=[self.ach_map.sample(x if self.ach_connected else np.zeros_like(x),REVERSAL_MV) for x in g]
        combined=[combine_conductance_stages([o,a]) for o,a in zip(orn,ach)]
        nodes=combined[0]['nodes'];np.testing.assert_array_equal(nodes,combined[1]['nodes'])
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise,self.decay,dt_ns);previous=self.state_dict()
        try:
            report=self.pn.advance(dt_ns,self.pn.cp.zeros_like(self.pn.voltage),synaptic_stages=combined,stage_observation_nodes=nodes,**solver_options)
            if not report['accepted']:return report
            observed=report['stage_observations'];np.testing.assert_array_equal(observed['nodes'],nodes)
            voltages=np.asarray(observed['voltage_mV']);charges=[]
            for groups in (orn,ach):
                current=[float(np.sum(x['conductance_nS']*(v[np.searchsorted(nodes,x['nodes'])]-x['reversal_mV']))) for x,v in zip(groups,voltages)]
                charges.append(dt_ns*1e-9*((1-GAMMA)*current[0]+GAMMA*current[1]))
            error=abs(sum(charges)-report['synaptic_outward_charge_increment_pC'])
            if not np.isfinite(charges).all() or error>1e-11+1e-12*sum(abs(q) for q in charges):raise ValueError('ORN/ACh charge decomposition failed')
            self.fast=fast;self.slow=slow;self.time_ns+=dt_ns;self.ach.load_state_dict(ach_next)
            self.orn_charge_pC+=charges[0];self.ach_charge_pC+=charges[1];self.output_nS()
            report['local_receptor_charge']=dict(orn_pC=charges[0],ach_pC=charges[1],sum_error_pC=error)
            return report
        except BaseException:
            self.load_state_dict(previous);raise
