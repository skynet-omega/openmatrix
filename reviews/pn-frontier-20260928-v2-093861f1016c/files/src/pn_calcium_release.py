"""Local calcium/resource boundary with explicit, required model parameters.

This is a candidate chemistry module, not an identified PN calcium channel.
The caller supplies nonnegative inward calcium current in pA. No voltage is
silently converted to calcium, no EM contact is declared to be one vesicle,
and no default PN kinetics or conductance is invented here.

Free Ca has linear clearance and an explicit rapid-buffer factor. Release is
a calcium-dependent hazard on a normalized replenishing pool. The midpoint
hazard is held within a step; pool evolution and its two output filters are
then integrated analytically. Distinct partners never duplicate pool updates.
"""
import hashlib
import numpy as np

FARADAY_C_PER_MOL=96485.33212
SCHEMA='PN_site_calcium_resource_candidate_v1'

def _vector(value,n,name,*,positive=False):
    a=np.asarray(value)
    if (a.shape!=(n,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all()
            or np.any(a<=0 if positive else a<0)):
        raise ValueError('Invalid '+name)
    return a.astype(np.float64,copy=True)

def calcium_advance(c,influx_pA,dt_s,rest,volume_um3,buffer_capacity,clearance_s):
    # pA/(2F) -> mol/s, um3 -> liters, mol/L -> micromolar.
    alpha=1e9/(2*FARADAY_C_PER_MOL*volume_um3*(1+buffer_capacity))
    amount=-np.expm1(-dt_s/clearance_s)
    return c+(rest-c+alpha*influx_pA*clearance_s)*amount

def calcium_hazard(calcium_uM,Kd_uM,cooperativity,maximum_per_s):
    # Logistic form avoids overflow from C**n. C=0 gives exactly zero.
    c=np.asarray(calcium_uM);out=np.zeros_like(c,dtype=float);active=c>0
    z=cooperativity[active]*(np.log(c[active])-np.log(Kd_uM[active]))
    e=np.exp(-abs(z));fraction=np.where(z>=0,1/(1+e),e/(1+e))
    out[active]=maximum_per_s[active]*fraction
    return out

def _phi(rate,dt):return -np.expm1(-rate*dt)/rate

def _convolution(a,b,dt):
    d=abs(a-b);same=d==0;out=np.empty_like(d)
    out[same]=dt*np.exp(-a[same]*dt)
    out[~same]=np.exp(-np.minimum(a[~same],b[~same])*dt)*(-np.expm1(-d[~same]*dt))/d[~same]
    return out

class CalciumReleaseSites:
    def __init__(self,site_ids,parameters,*,provenance,time_ns=0):
        ids=np.asarray(site_ids)
        keys={'rest_uM','volume_um3','buffer_capacity','clearance_s','Kd_uM',
              'cooperativity','maximum_hazard_per_s','recovery_s','rise_s','decay_s'}
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids<0)
                or np.any(ids[1:]<=ids[:-1]) or set(parameters)!=keys
                or not isinstance(provenance,str) or not provenance or type(time_ns) is not int or time_ns<0):
            raise ValueError('Sorted site identities and complete explicit parameters/provenance required')
        self.ids=np.frombuffer(ids.astype(np.int64).tobytes(),dtype=np.int64);self.parameters={}
        for name,value in parameters.items():
            a=_vector(value,len(ids),name,positive=name not in {'rest_uM','buffer_capacity','maximum_hazard_per_s'})
            self.parameters[name]=np.frombuffer(a.tobytes(),dtype=np.float64)
        p=self.parameters
        if np.any(p['rise_s']>=p['decay_s']):raise ValueError('Output rise must precede decay')
        h=hashlib.sha256(SCHEMA.encode()+self.ids.tobytes()+provenance.encode())
        for k,v in sorted(p.items()):h.update(k.encode());h.update(v.tobytes())
        self.identity=h.hexdigest();self.provenance=provenance;self.time_ns=time_ns
        peak=np.log(p['decay_s']/p['rise_s'])/(1/p['rise_s']-1/p['decay_s'])
        self.norm=np.exp(-peak/p['decay_s'])-np.exp(-peak/p['rise_s'])
        self.calcium_uM=p['rest_uM'].copy();self.available=np.ones(len(ids))
        self.fast=np.zeros(len(ids));self.slow=np.zeros(len(ids))
        self.released=np.zeros(len(ids));self.calcium_charge_pC=np.zeros(len(ids))

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,
                    **{k:getattr(self,k).copy() for k in ['calcium_uM','available','fast','slow','released','calcium_charge_pC']})

    def load_state_dict(self,state):
        if (state.get('schema')!=SCHEMA or state.get('identity')!=self.identity
                or type(state.get('time_ns')) is not int or state['time_ns']<0):raise ValueError('Incompatible release state')
        values={k:_vector(state[k],len(self.ids),k) for k in ['calcium_uM','available','fast','slow','released','calcium_charge_pC']}
        if np.any(values['available']>1) or np.any(values['slow']<values['fast']):raise ValueError('Invalid pool/filter state')
        for k,v in values.items():setattr(self,k,v)
        self.time_ns=state['time_ns']

    def preview(self,dt_ns,inward_calcium_pA,*,release_enabled=True):
        if type(dt_ns) is not int or dt_ns<=0 or type(release_enabled) is not bool:raise ValueError('Integer step and explicit release switch required')
        return self._preview_elapsed(dt_ns,inward_calcium_pA,release_enabled)

    def midpoint_activation(self,dt_ns,inward_calcium_pA,*,release_enabled=True):
        # Internal stage time may be a half ns; persisted clocks stay integers.
        if type(dt_ns) is not int or dt_ns<=0 or type(release_enabled) is not bool:raise ValueError('Invalid midpoint interval')
        state=self._preview_elapsed(dt_ns/2,inward_calcium_pA,release_enabled)['state']
        return (state['slow']-state['fast'])/self.norm

    def _preview_elapsed(self,dt_ns,inward_calcium_pA,release_enabled):
        influx=_vector(inward_calcium_pA,len(self.ids),'local inward calcium current');p=self.parameters;dt=dt_ns*1e-9
        mid=calcium_advance(self.calcium_uM,influx,dt/2,p['rest_uM'],p['volume_um3'],p['buffer_capacity'],p['clearance_s'])
        end=calcium_advance(self.calcium_uM,influx,dt,p['rest_uM'],p['volume_um3'],p['buffer_capacity'],p['clearance_s'])
        hazard=calcium_hazard(mid,p['Kd_uM'],p['cooperativity'],p['maximum_hazard_per_s'])
        if not release_enabled:hazard=np.zeros_like(hazard)
        # Extended intermediates protect small positive integrals from loss
        # through cancellation; all persisted physical states remain FP64.
        rate=hazard.astype(np.longdouble);rec=(1/p['recovery_s']).astype(np.longdouble)
        k=rec+rate;steady=rec/k;r0=self.available.astype(np.longdouble);h=np.longdouble(dt)
        ph=_phi(k,h);available=steady+(r0-steady)*np.exp(-k*h)
        x=k*h;one_minus_phi=np.empty_like(x);small=abs(x)<1e-4
        one_minus_phi[small]=x[small]/2-x[small]**2/6+x[small]**3/24-x[small]**4/120
        one_minus_phi[~small]=1-ph[~small]/h
        released=rate*(r0*ph+steady*h*one_minus_phi)
        filters=[]
        for name,tau in [('fast',p['rise_s']),('slow',p['decay_s'])]:
            a=(1/tau).astype(np.longdouble);conv=_convolution(a,k,h)
            value=getattr(self,name)*np.exp(-a*h)+rate*(r0*conv+steady*(_phi(a,h)-conv))
            filters.append(np.asarray(value,dtype=np.float64))
        values=dict(calcium_uM=end,available=np.asarray(available,dtype=np.float64),fast=filters[0],slow=filters[1],
                    released=self.released+np.asarray(released,dtype=np.float64),calcium_charge_pC=self.calcium_charge_pC+influx*dt)
        candidate=dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns+dt_ns,**values)
        if (any(not np.isfinite(v).all() or np.any(v<0) for v in values.values())
                or np.any(values['available']>1) or np.any(values['slow']<values['fast'])):
            raise FloatingPointError('Invalid candidate chemistry; no clipping or commit')
        return dict(parent_time_ns=self.time_ns,parent_state_hash=self._state_hash(),state=candidate,
                    release_increment=np.asarray(released,dtype=np.float64),midpoint_hazard_per_s=hazard)

    def _state_hash(self):
        h=hashlib.sha256(self.identity.encode()+str(self.time_ns).encode())
        for k,v in sorted(self.state_dict().items()):
            if isinstance(v,np.ndarray):h.update(k.encode());h.update(v.tobytes())
        return h.hexdigest()

    def commit(self,proposal):
        if proposal.get('parent_time_ns')!=self.time_ns or proposal.get('parent_state_hash')!=self._state_hash():
            raise ValueError('Stale or repeated presynaptic update')
        candidate=proposal['state']
        if (type(candidate.get('time_ns')) is not int or candidate['time_ns']<=self.time_ns
                or np.any(np.asarray(candidate['released'])<self.released)
                or np.any(np.asarray(candidate['calcium_charge_pC'])<self.calcium_charge_pC)):
            raise ValueError('Chemistry commit must advance time and cumulative counters')
        self.load_state_dict(candidate)

    def activation(self):return (self.slow-self.fast)/self.norm

class SiteToTargetConductance:
    """Distribute a site's shared activation through explicit target gains.

    Gains are required nS per unit of this model's release, not deduced from
    EM counts. A single site may address several targets with distinct gains.
    Repeated site/target relations are anatomical multiplicity, not new pools.
    """
    def __init__(self,site_ids,relation_sites,target_ids,gain_nS):
        ids=np.asarray(site_ids);sites=np.asarray(relation_sites);targets=np.asarray(target_ids)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids[1:]<=ids[:-1])
                or sites.ndim!=1 or sites.dtype.kind not in 'iu' or not len(sites)
                or targets.shape!=sites.shape or targets.dtype.kind not in 'iu' or np.any(targets<=0)
                or not np.isin(sites,ids).all()):raise ValueError('Explicit site/target identities required')
        self.site_ids=ids.copy();self.slot=np.searchsorted(ids,sites);self.target_ids,self.target_slot=np.unique(targets,return_inverse=True)
        self.gain=_vector(gain_nS,len(sites),'relation gain');self.relations=len(sites)

    def evaluate(self,activation):
        a=_vector(activation,len(self.site_ids),'shared site activation')
        g=self.gain*a[self.slot]
        if not np.isfinite(g).all():raise FloatingPointError('Conductance overflow')
        return np.bincount(self.target_slot,weights=g,minlength=len(self.target_ids))
