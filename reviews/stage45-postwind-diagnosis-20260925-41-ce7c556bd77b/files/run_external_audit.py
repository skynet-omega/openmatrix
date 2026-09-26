"""Run the unmodified ChatGPT audit on the actual preserved NPZ arrays."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from chatgpt_audit_original import descomponer,relojes,integral_zoh


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with np.load(HERE/'inputs/donor_traces.npz',allow_pickle=False) as z:
        prep=np.flatnonzero(z['fase']=='preparacion')[-1]
        trial=np.flatnonzero(z['fase']=='ensayo')
        ix=np.r_[prep,trial];q=z['qpos'][ix];xy=q[:,:2]*.01
        w,x,y,v=q[:,3:7].T
        yaw=np.arctan2(2*(w*v+x*y),1-2*(y*y+v*v))
        t=z['body_time_ns'][ix].astype(float)*1e-9
        source=np.asarray(json.loads((HERE/'inputs/donor_GAUSSIAN_SPEC.json').read_text())['minus']['source_mm'])*.001
        r=descomponer(t,xy,yaw,source,[(t[0],t[1000]),(t[1000],t[1020]),(t[1020],t[-1])])
        r['clock_audit']=relojes(t,z['CNS_time_ns'][ix].astype(float)*1e-6)
        r['applied_postwind_integral_deg']=float(np.rad2deg(integral_zoh(t,z['command_yaw_rate_rad_s'][trial],t[1020],t[-1])))
        r['trace_sha256']=hashlib.sha256((HERE/'inputs/donor_traces.npz').read_bytes()).hexdigest()
        r['external_code_sha256']=hashlib.sha256((HERE/'chatgpt_audit_original.py').read_bytes()).hexdigest()
    local=json.loads((HERE/'VERIFIED.json').read_text())['arms']['identity'];post=r['ventanas'][-1]
    errors={k:abs(post[e]-local[k]) for k,e in [('delta_bearing_deg','geometria_deg'),('delta_yaw_deg','giro_deg'),('delta_signed_error_deg','delta_error_deg')] if k!='delta_yaw_deg'}
    errors['delta_yaw_deg']=abs(post['giro_deg']+local['delta_yaw_deg'])
    if max(errors.values())>1e-10:raise ValueError('Independent geometric decompositions disagree')
    r['agreement_abs_deg']=errors;r['executed_by']='Codex locally, real recorded trace; ChatGPT executed only synthetic examples'
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))


if __name__=='__main__':main()
