"""Descriptive audit of the exposed command; no fitting or policy selection."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with np.load(HERE/'inputs/donor_traces.npz',allow_pickle=False) as z:
        ix=np.flatnonzero(z['fase']=='ensayo')
        dq=z['DN_q_usada'][ix]-z['DN_baseline'][ix]
        raw=np.tanh(250*(dq[:,2]-dq[:,3]))*math.radians(5)
        state=0.;states=[];applied=[];alpha=-math.expm1(-.001/.2)
        for u in raw:
            state+=alpha*(u-state);states.append(state)
            applied.append(math.copysign(math.radians(5),state) if abs(state)>=.0005 else 0.)
        states=np.asarray(states);applied=np.asarray(applied)
        for computed,k in [(raw,'neural_command_raw_rad_s'),(states,'motor_filter_state_rad_s'),(applied,'motor_filter_applied_rad_s')]:
            if not np.array_equal(computed,z[k][ix]):raise ValueError('Filter recurrence differs: '+k)
        metrics={}
        for label,v in [('raw',raw),('EMA',states),('applied',applied)]:
            x=v[1020:]
            metrics[label]=dict(net_integral_deg=float(np.rad2deg(sum(x)*.001)),
                positive_integral_deg=float(np.rad2deg(sum(x[x>0])*.001)),
                negative_integral_deg=float(np.rad2deg(sum(x[x<0])*.001)),
                positive_ms=int(sum(x>0)),negative_ms=int(sum(x<0)),zero_ms=int(sum(x==0)))
        odor=z['concentracion_campo'][ix][1020:]
        lr=odor[:,0]-odor[:,1]
    result=dict(schema='exposed_filter_descriptive_audit_v1',raw_EMA_relay_all_exact=True,
        postwind_ms=980,metrics=metrics,odor_L_minus_R_range=[float(lr.min()),float(lr.max())],
        trace_sha256=hashlib.sha256((HERE/'inputs/donor_traces.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Posthoc description of exposed life. No threshold/gain fitted, no prospective hypothesis test or localization of neural cause.',
        interpretation='The code matches the frozen filter exactly. Positive postwind motor integral is already present weakly in raw input and strongly amplified by EMA/relay; this does not identify which circuit generated the signal.')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
