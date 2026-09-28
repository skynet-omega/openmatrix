"""Projection-specific conductance interfaces, with explicit unresolved scales."""
import numpy as np
import numba
from scipy.optimize import brentq


def biexponential_parameters(peak_time_s, decay_s):
    """Unit-area cascade with a specified peak time and asymptotic decay.

    Interpreting the paper's reported rise interval as time to peak is an
    explicit waveform assumption, not an assertion of a 10-90% measurement.
    """
    if not 0 < peak_time_s < decay_s: raise ValueError('Invalid waveform timing')
    def tpeak(r): return r*decay_s/(decay_s-r)*np.log(decay_s/r)
    rise = brentq(lambda r:tpeak(r)-peak_time_s,decay_s*1e-8,decay_s*(1-1e-8),xtol=1e-15)
    peak = (np.exp(-peak_time_s/decay_s)-np.exp(-peak_time_s/rise))/(decay_s-rise)
    return rise,1./peak


@numba.njit(cache=True, fastmath=False)
def projection_conductances(rows,local_ptr,mode,fraction,slot,pn_slot,apl_edge_slot,
                           ptr,idx,weights,transmission,sax,caps,visual,connected,
                           receptor,apl_edge_release,scales,group,pair_route_slot,
                           route_fractions,route_apl,nregions,pn_gain_nS,
                           pn_filtered_rate_norm,apl_gbar_nS):
    """Return total and isolated PN-gamma/APL-gamma conductances in nS.

    PN gain includes a declared quantal area and presynaptic rate normalization.
    APL gbar is a stored nS parameter; APL's rate cap is never read on this path.
    Other afferents retain exactly the previous signed-weight interface.
    """
    ge=np.zeros(len(rows));gi=np.zeros(len(rows));pn_ge=ge.copy();apl_gi=ge.copy()
    age=np.zeros((2,nregions));agi=np.zeros((2,nregions))
    for j in range(len(rows)):
        row=rows[j]
        for e in range(ptr[row],ptr[row+1]):
            pre=idx[e]
            if not connected and visual[pre]:continue
            a=local_ptr[j]+e-ptr[row]
            if group[j]==2 and pn_slot[a]>=0:
                value=pn_gain_nS[a]*pn_filtered_rate_norm[pn_slot[a]]
                ge[j]+=value;pn_ge[j]+=value;continue
            if group[j]==2 and apl_edge_slot[a]>=0:
                value=apl_gbar_nS[a]*apl_edge_release[apl_edge_slot[a]]
                gi[j]+=value;apl_gi[j]+=value;continue
            s=transmission[pre]
            if mode[a]==1:s*=1.-fraction[a]
            elif mode[a]==2:s+=fraction[a]*(sax[slot[a]]-s)
            if apl_edge_slot[a]>=0:s=apl_edge_release[apl_edge_slot[a]]
            f=s*caps[pre]
            if pn_slot[a]>=0:f=(400./.375)*receptor[pn_slot[a]]
            sign=0 if weights[e]>=0. else 1
            value=abs(weights[e])*f*scales[group[j],sign]
            if sign==0:ge[j]+=value
            else:gi[j]+=value
            if group[j]==3:
                route=pair_route_slot[a];cell=route_apl[route]
                for k in range(nregions):
                    contribution=value*route_fractions[route,k]
                    if sign==0:age[cell,k]+=contribution
                    else:agi[cell,k]+=contribution
    return ge,gi,age,agi,pn_ge,apl_gi
