"""Explicit voltage-gated Ca current and local chemistry for fine PN stages.

No default physiological parameter is supplied. The generic gate family has
logistic steady state and a supplied constant tau. Its use and transfers must
be declared by the caller; it does not identify native PN terminal channels.
Positive electrical current is outward. Sites share each physical-node current
through explicit fractions summing to one, rather than duplicating it per site.
"""
import hashlib,copy
import numpy as np
from pn_calcium_release import CalciumReleaseSites,calcium_hazard,FARADAY_C_PER_MOL

METHOD='local_Ca_and_resource_SDIRK2_v1'
GAMMA=1.-1./np.sqrt(2.)

def frozen(x,dtype=float):
    a=np.asarray(x,dtype=dtype);return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)

def chemistry_stages(chemistry,dt_ns,inward_stages_pA,*,release_enabled=True):
    """The SAME RK quadrature as membrane, including clearance and resources."""
    j=np.asarray(inward_stages_pA,dtype=float);n=len(chemistry.ids)
    if type(dt_ns) is not int or dt_ns<=0 or j.shape!=(2,n) or not np.isfinite(j).all() or np.any(j<0):
        raise ValueError('Two finite inward Ca stages required; outward transport not implemented')
    p=chemistry.parameters;dt=dt_ns*1e-9;q=GAMMA*dt;ratio=(1-GAMMA)/GAMMA
    alpha=1e9/(2*FARADAY_C_PER_MOL*p['volume_um3']*(1+p['buffer_capacity']))
    old=chemistry.state_dict();names=['calcium_uM','available','fast','slow'];base={k:old[k] for k in names};stages=[]
    for i in range(2):
        ca=(base['calcium_uM']+q*(p['rest_uM']/p['clearance_s']+alpha*j[i]))/(1+q/p['clearance_s'])
        hazard=calcium_hazard(ca,p['Kd_uM'],p['cooperativity'],p['maximum_hazard_per_s'])
        if not release_enabled:hazard=np.zeros_like(hazard)
        pool=(base['available']+q/p['recovery_s'])/(1+q*(1/p['recovery_s']+hazard));flow=hazard*pool
        row=dict(calcium_uM=ca,available=pool,fast=(base['fast']+q*flow)/(1+q/p['rise_s']),
                 slow=(base['slow']+q*flow)/(1+q/p['decay_s']),flow=flow)
        if (any(not np.isfinite(row[k]).all() or np.any(row[k]<0) for k in names)
                or np.any(pool>1) or np.any(row['slow']<row['fast'])):
            raise FloatingPointError('Invalid chemical stage; reject without clipping')
        stages.append(row)
        if i==0:base={k:old[k]+ratio*(row[k]-old[k]) for k in names}
    weights=np.array([1-GAMMA,GAMMA]);Q=dt*(weights@j)
    released=dt*sum(w*r['flow'] for w,r in zip(weights,stages))
    cleared=dt*sum(w*(r['calcium_uM']-p['rest_uM'])/(p['clearance_s']*alpha) for w,r in zip(weights,stages))
    balance=(stages[1]['calcium_uM']-old['calcium_uM'])/alpha+cleared-Q
    recovered=dt*sum(w*(1-r['available'])/p['recovery_s'] for w,r in zip(weights,stages))
    pool_balance=stages[1]['available']-old['available']-recovered+released
    state=dict(schema=old['schema'],identity=old['identity'],time_ns=old['time_ns']+dt_ns,
               **{k:stages[1][k] for k in names},released=old['released']+released,
               calcium_charge_pC=old['calcium_charge_pC']+Q)
    if not all(np.isfinite(v).all() for v in [Q,released,cleared,balance,pool_balance]):raise FloatingPointError('Nonfinite chemistry')
    return dict(parent_time_ns=old['time_ns'],parent_state_hash=chemistry._state_hash(),state=state,
                release_increment=released,calcium_charge_increment_pC=Q,cleared_charge_pC=cleared,
                calcium_balance_pC=balance,pool_balance=pool_balance,
                output_activation_stages=np.array([(r['slow']-r['fast'])/chemistry.norm for r in stages]))

class CalciumPort:
    def __init__(self,nodes,gbar_nS,reversal_mV,gate_half_mV,gate_slope_mV,gate_tau_s,gate_power,
                 site_nodes,site_fractions,chemistry,*,provenance,enabled,release_enabled=True):
        nodes=np.asarray(nodes);bar=np.asarray(gbar_nS);half=np.asarray(gate_half_mV);slope=np.asarray(gate_slope_mV)
        tau=np.asarray(gate_tau_s);power=np.asarray(gate_power);site_nodes=np.asarray(site_nodes);fraction=np.asarray(site_fractions)
        if (nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or not len(nodes) or np.any(nodes<0) or np.any(nodes[1:]<=nodes[:-1])
                or bar.shape!=nodes.shape or np.any(bar<0) or half.ndim!=2 or half.shape[0]!=len(nodes)
                or half.shape!=slope.shape or half.shape!=tau.shape or half.shape[1]==0 or power.shape!=(half.shape[1],)
                or np.any(slope==0) or np.any(tau<=0) or np.any(power<=0) or np.any(power!=np.floor(power))
                or site_nodes.shape!=chemistry.ids.shape or site_nodes.dtype.kind not in 'iu' or fraction.shape!=site_nodes.shape
                or np.any(fraction<=0) or not np.array_equal(np.unique(site_nodes),nodes)
                or not isinstance(provenance,str) or not provenance or type(enabled) is not bool or type(release_enabled) is not bool
                or not np.isscalar(reversal_mV) or not np.isfinite(reversal_mV)
                or any(not np.isfinite(a).all() for a in [bar,half,slope,tau,power,fraction])):
            raise ValueError('Complete explicit Ca channel/site parameters and provenance required')
        slots=np.searchsorted(nodes,site_nodes)
        if not np.allclose(np.bincount(slots,weights=fraction,minlength=len(nodes)),1.,rtol=0,atol=1e-12):
            raise ValueError('Each physical Ca current must be allocated exactly once')
        self.nodes=frozen(nodes,np.int64);self.gbar=frozen(bar);self.reversal=float(reversal_mV)
        self.half=frozen(half);self.slope=frozen(slope);self.tau=frozen(tau);self.power=frozen(power,np.int64)
        self.site_slot=frozen(slots,np.int64);self.fractions=frozen(fraction)
        self.chemistry=chemistry;self.time_ns=chemistry.time_ns;self.enabled=enabled;self.release_enabled=release_enabled;self.provenance=provenance
        h=hashlib.sha256((METHOD+chemistry.identity+provenance+repr(self.reversal)).encode())
        for a in [self.nodes,self.gbar,self.half,self.slope,self.tau,self.power,self.site_slot,self.fractions]:h.update(a.tobytes())
        self.identity=h.hexdigest()
        self._definition=(self.nodes,self.gbar,self.half,self.slope,self.tau,self.power,self.site_slot,self.fractions,
                          self.chemistry,self.reversal,self.provenance,self.identity,chemistry.identity)
        self._chem_definition=tuple(sorted(chemistry.parameters.items()))
        self._chem_ids=chemistry.ids;self._chem_provenance=chemistry.provenance
        chemistry.norm=frozen(chemistry.norm);self._chem_norm=chemistry.norm
        self.gates=np.zeros_like(self.half);self.outward_charge_pC=np.zeros(len(nodes))

    def assert_model(self):
        current=(self.nodes,self.gbar,self.half,self.slope,self.tau,self.power,self.site_slot,self.fractions,self.chemistry)
        if (any(a is not b for a,b in zip(current,self._definition[:9]))
                or (self.reversal,self.provenance,self.identity,self.chemistry.identity)!=self._definition[9:]
                or set(self.chemistry.parameters)!=set(k for k,v in self._chem_definition)
                or any(self.chemistry.parameters[k] is not v for k,v in self._chem_definition)
                or self.chemistry.norm is not self._chem_norm or self.chemistry.ids is not self._chem_ids
                or self.chemistry.provenance!=self._chem_provenance):
            raise ValueError('Ca definition changed; construct and migrate explicitly')

    def assert_state(self):
        self.assert_model()
        self.validated_state(self.state_dict())

    def steady(self,voltage_mV):
        v=np.asarray(voltage_mV);z=(v[:,None]-self.half)/self.slope;e=np.exp(-abs(z))
        return np.where(z>=0,1/(1+e),e/(1+e))

    def stage(self,voltage_mV,base,duration_s):
        v=np.asarray(voltage_mV);steady=self.steady(v)
        x=(self.tau*base+duration_s*steady)/(self.tau+duration_s)
        dx=duration_s*steady*(1-steady)/(self.slope*(self.tau+duration_s))
        factors=x**self.power;product=np.prod(factors,axis=1);derivative=np.zeros(len(v))
        for k,power in enumerate(self.power):
            other=np.prod(np.delete(factors,k,axis=1),axis=1)
            derivative+=power*x[:,k]**(power-1)*dx[:,k]*other
        g=self.gbar*product if self.enabled else np.zeros(len(v));dg=self.gbar*derivative if self.enabled else np.zeros(len(v))
        current=g*(v-self.reversal);jac=g+dg*(v-self.reversal)
        residual=x-base-duration_s*(steady-x)/self.tau
        if (not np.isfinite(x).all() or np.any(x<0) or np.any(x>1)
                or not np.isfinite(current+jac).all()):raise FloatingPointError('Invalid Ca gate stage')
        return dict(gates=x,current=current,jacobian=jac,frozen_conductance=g,gate_error=float(abs(residual).max()))

    def final_proposal(self,dt_ns,first,second):
        outward=np.array([first['current'],second['current']])
        inward=-outward[:,self.site_slot]*self.fractions
        proposal=chemistry_stages(self.chemistry,dt_ns,inward,release_enabled=self.release_enabled)
        dq=dt_ns*1e-9*((1-GAMMA)*outward[0]+GAMMA*outward[1])
        if not np.isfinite(self.outward_charge_pC+dq).all():raise FloatingPointError('Nonfinite Ca charge')
        error=float(abs(dq.sum()+proposal['calcium_charge_increment_pC'].sum()))
        if error>1e-12 or np.max(abs(proposal['calcium_balance_pC']))>1e-12 or np.max(abs(proposal['pool_balance']))>1e-12:
            raise FloatingPointError('Ca current/site/resource conservation failed')
        return dict(gates=second['gates'],dq=dq,chemistry=proposal,allocation_error_pC=error)

    def commit(self,proposal):
        self.chemistry.commit(proposal['chemistry'])
        self.gates=proposal['gates'];self.outward_charge_pC+=proposal['dq'];self.time_ns=self.chemistry.time_ns

    def state_dict(self):
        return dict(identity=self.identity,time_ns=self.time_ns,enabled=self.enabled,release_enabled=self.release_enabled,gates=self.gates.copy(),
                    outward_charge_pC=self.outward_charge_pC.copy(),chemistry=self.chemistry.state_dict())

    def validated_state(self,s):
        self.assert_model()
        if (set(s)!={'identity','time_ns','enabled','release_enabled','gates','outward_charge_pC','chemistry'} or s.get('identity')!=self.identity or type(s.get('time_ns')) is not int or s['time_ns']<0
                or s['chemistry']['time_ns']!=s['time_ns'] or type(s.get('enabled')) is not bool or type(s.get('release_enabled')) is not bool):raise ValueError('Ca identity/clock mismatch')
        x=np.asarray(s['gates']);q=np.asarray(s['outward_charge_pC'])
        if x.dtype.kind!='f' or q.dtype.kind!='f' or x.shape!=self.half.shape or q.shape!=self.nodes.shape or not np.isfinite(x).all() or not np.isfinite(q).all() or np.any(x<0) or np.any(x>1):raise ValueError('Invalid Ca state')
        chem=copy.copy(self.chemistry);chem.load_state_dict(s['chemistry'])
        allocated=np.bincount(self.site_slot,weights=chem.calcium_charge_pC,minlength=len(self.nodes))
        if np.any(q>0) or np.any(abs(q+allocated)>1e-12+1e-12*allocated):
            raise ValueError('Electrical Ca and accumulated site charge disagree')
        return x.copy(),q.copy(),chem,s['time_ns'],s['enabled'],s['release_enabled']

    def apply_validated(self,values):
        x,q,chem,time_ns,enabled,release_enabled=values
        for k in ['calcium_uM','available','fast','slow','released','calcium_charge_pC']:
            setattr(self.chemistry,k,getattr(chem,k))
        self.chemistry.time_ns=time_ns
        self.gates=x;self.outward_charge_pC=q;self.time_ns=time_ns;self.enabled=enabled;self.release_enabled=release_enabled

    def load_state_dict(self,s):
        self.apply_validated(self.validated_state(s))
