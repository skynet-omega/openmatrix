"""Historical physical equations unchanged; retain the already-computed post state."""
import math
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def lif_record_with_post(voltage,refractory_left,counts,q,vinf,rate,reset,threshold,caps,filter_tau,dt,refractory=.0022):
    """Same analytic held-input LIF as the parent, additionally retain effective jumps."""
    max_events=2+int(dt/refractory);times=np.full((len(q),max_events),np.nan);jumps=np.zeros_like(times);clipped=0;posts=np.full_like(times,np.nan)
    for k in range(len(voltage)):
        t=0.;v=voltage[k];left=refractory_left[k];value=q[k]*math.exp(-dt/filter_tau[k]);n=0
        while t<dt:
            wait=min(left,dt-t);t+=wait;left-=wait
            if t>=dt:break
            if vinf[k]<=threshold[k]:v=vinf[k]+(v-vinf[k])*math.exp(-rate[k]*(dt-t));t=dt;break
            crossing=math.log1p((threshold[k]-v)/(vinf[k]-threshold[k]))/rate[k]
            if crossing>dt-t:v=vinf[k]+(v-vinf[k])*math.exp(-rate[k]*(dt-t));t=dt;break
            t+=max(0.,crossing);counts[k]+=1
            before=value*math.exp((dt-t)/filter_tau[k]);event_value=before+1./(caps[k]*filter_tau[k])
            if event_value>1.:clipped+=1;event_value=1.
            if n>=max_events:raise ValueError('Event accounting capacity exceeded')
            times[k,n]=t;jumps[k,n]=event_value-before;posts[k,n]=event_value;n+=1
            value=event_value*math.exp(-(dt-t)/filter_tau[k]);v=reset[k];left=refractory
        voltage[k]=v;refractory_left[k]=left;counts[k]+=0;q[k]=value
    return clipped,times,jumps,posts
