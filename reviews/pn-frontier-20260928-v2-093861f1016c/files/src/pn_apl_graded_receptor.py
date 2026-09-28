"""Delayed graded receptor occupancy; no spikes, rate caps, or fitted efficacy.

The driver is a dimensionless regional APL release coordinate before its old
generic synaptic filter. A new receptor starts at zero on declared activation;
unknown earlier release is not reconstructed. Physical efficacy is supplied
separately by the local conductance map, once for the whole anatomical pair.
"""
import hashlib,json,copy
import numpy as np
from pn_coupled_ionic import GAMMA

SCHEMA='APL_delayed_graded_receptor_v1'


class DelayedGradedReceptor:
    def __init__(self,region_names,*,initial_driver,time_ns,tau_s,delay_ns,provenance):
        names=tuple(region_names)
        if (not names or len(set(names))!=len(names) or not all(isinstance(v,str) and v for v in names)
            or type(time_ns) is not int or time_ns<0 or type(delay_ns) is not int or delay_ns<=0
            or not np.isscalar(tau_s) or not np.isfinite(tau_s) or tau_s<=0 or not isinstance(provenance,str) or not provenance):
            raise ValueError('Named graded regions, positive physical kinetics, explicit origin and provenance required')
        self.names=names;self.origin_ns=time_ns;self.time_ns=time_ns;self.tau_s=float(tau_s);self.delay_ns=delay_ns;self.provenance=provenance
        self.last_driver=self._driver(initial_driver);self.occupancy=np.zeros(len(names));self.history=[]
        definition=dict(schema=SCHEMA,names=names,origin_ns=time_ns,tau_s=self.tau_s,delay_ns=delay_ns,provenance=provenance,initial_driver=self.last_driver.tolist())
        self.identity=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()

    def _driver(self,value):
        x=np.asarray(value)
        if x.shape!=(len(self.names),) or x.dtype.kind not in 'fiu' or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
            raise ValueError('One finite bounded graded value per named region required')
        return x.astype(float,copy=True)

    def _at(self,offset_ns,history):
        y=self.occupancy.copy();cursor=0.;stop=float(offset_ns)
        for segment in history:
            start=segment['start_ns']-self.time_ns;end=segment['end_ns']-self.time_ns
            if end<=cursor:continue
            if start>cursor:
                gap=min(float(start),stop)-cursor
                if gap>0:y*=np.exp(-gap*1e-9/self.tau_s);cursor+=gap
            if cursor>=stop:break
            right=min(float(end),stop);left=max(cursor,float(start))
            if right<=left:continue
            total_s=(end-start)*1e-9;h=(right-left)*1e-9
            slope=(segment['right']-segment['left'])/total_s
            r0=segment['left']+slope*((left-start)*1e-9)
            em1=np.expm1(-h/self.tau_s)
            y=y*(1+em1)-r0*em1+slope*(h+self.tau_s*em1);cursor=right
        if cursor<stop:y*=np.exp(-(stop-cursor)*1e-9/self.tau_s)
        return self._driver(y)

    def preview(self,dt_ns,before,after):
        if type(dt_ns) is not int or dt_ns<=0:raise ValueError('Positive integer interval required')
        a,b=self._driver(before),self._driver(after)
        if not np.array_equal(a,self.last_driver):raise ValueError('Graded source history discontinuity')
        history=copy.deepcopy(self.history)
        history.append(dict(start_ns=self.time_ns+self.delay_ns,end_ns=self.time_ns+dt_ns+self.delay_ns,left=a,right=b))
        stages=[self._at(c*dt_ns,history) for c in (GAMMA,1.)];stop=self.time_ns+dt_ns
        state=dict(schema=SCHEMA,identity=self.identity,time_ns=stop,occupancy=stages[1].copy(),last_driver=b,
            history=[s for s in history if s['end_ns']>stop])
        return stages,state

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,occupancy=self.occupancy.copy(),last_driver=self.last_driver.copy(),history=copy.deepcopy(self.history))

    def validated(self,state):
        if (set(state)!={'schema','identity','time_ns','occupancy','last_driver','history'} or state['schema']!=SCHEMA or state['identity']!=self.identity
            or type(state['time_ns']) is not int or state['time_ns']<self.origin_ns or not isinstance(state['history'],list)):
            raise ValueError('Wrong graded receptor identity or clock')
        y=self._driver(state['occupancy']);last=self._driver(state['last_driver']);history=[];previous=None
        for s in state['history']:
            if (set(s)!={'start_ns','end_ns','left','right'} or type(s['start_ns']) is not int or type(s['end_ns']) is not int
                or s['start_ns']<self.origin_ns+self.delay_ns or s['end_ns']<=s['start_ns'] or s['end_ns']<=state['time_ns']):
                raise ValueError('Invalid delayed graded interval')
            a,b=self._driver(s['left']),self._driver(s['right'])
            if previous is not None and (s['start_ns']!=previous['end_ns'] or not np.array_equal(a,previous['right'])):
                raise ValueError('Discontinuous delayed graded history')
            previous=dict(start_ns=s['start_ns'],end_ns=s['end_ns'],left=a,right=b);history.append(previous)
        if state['time_ns']>self.origin_ns:
            if not history or history[-1]['end_ns']!=state['time_ns']+self.delay_ns or not np.array_equal(history[-1]['right'],last):
                raise ValueError('Missing latest delayed source history')
            if history[0]['start_ns']>max(state['time_ns'],self.origin_ns+self.delay_ns):raise ValueError('Gap in current delayed source history')
        elif history or np.any(y):raise ValueError('Nonzero receptor before declared activation')
        return y,last,history

    def load_state_dict(self,state):
        y,last,history=self.validated(state)
        self.occupancy=y;self.last_driver=last;self.history=history;self.time_ns=state['time_ns']
