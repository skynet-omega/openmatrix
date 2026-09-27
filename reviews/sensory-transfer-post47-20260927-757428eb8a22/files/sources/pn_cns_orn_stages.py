"""Sample continuing canonical ORN filters; never evolve a duplicate history.

Linear interpolation within a CNS interval is a coupling approximation. Its
error must be checked independently of the fine-PN implicit solver error.
Published population gain is allocated once by SpatialOrnAllocation.
"""
import numpy as np
from pn_coupled_ionic import GAMMA


def canonical_orn_stages(allocation,caps_hz,before,after,dt_ns,*,reversal_mV=-10.,connected=True):
    if (type(dt_ns) is not int or dt_ns<=0 or type(connected) is not bool
        or type(before.get('time_ns')) is not int or type(after.get('time_ns')) is not int
        or after['time_ns']-before['time_ns']!=dt_ns
        or not np.isscalar(reversal_mV) or not np.isfinite(reversal_mV)):
        raise ValueError('Explicit matching interval, connection switch and reversal required')
    frames=[]
    for frame in (before,after):
        ids=np.asarray(frame['ORN_source_ids']);f=np.asarray(frame['ORN_filters'])
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)!=len(allocation.source_ids)
            or len(np.unique(ids))!=len(ids) or not np.array_equal(np.sort(ids),allocation.source_ids)
            or f.shape!=(4,len(ids)) or f.dtype.kind not in 'fiu' or not np.isfinite(f).all() or np.any(f<0) or np.any(f>1)):
            raise ValueError('Complete ORN identities and bounded canonical source filters required')
        order=np.argsort(ids);frames.append(f[:,order])
    a,b=frames;out=[]
    for stage in (GAMMA,1.):
        f=(1-stage)*a+stage*b
        g=allocation.sample_filters(f,caps_hz)
        if not connected:g=np.zeros_like(g)
        out.append(dict(nodes=allocation.nodes,conductance_nS=g,reversal_mV=float(reversal_mV)))
    return out
