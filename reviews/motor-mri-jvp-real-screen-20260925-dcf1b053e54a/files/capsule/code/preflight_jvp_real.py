"""Freeze one tolerance-scale free-state JVP direction before any GPU replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'JVP_REAL_PLAN_62.json'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out,direction):
    start=time.monotonic()
    need(not out.exists() and not direction.exists(),'output exists')
    p=json.loads(PLAN.read_text());need(p['schema']=='jvp_real_plan_v1','plan')
    for rel,expected in p['frozen_sha256'].items():
        need(sha(ROOT/rel)==expected,'source changed: '+rel)
    j=p['stage_index'];need(j==1,'stage index')
    with np.load(HERE/'mri_slow_inputs_01/MRI_SLOW_INPUTS.npz',allow_pickle=False) as z:
        x=z['full_z'][j].copy();f=z['full_f'][j].copy()
        raw=z['slow_f'][j].copy();mask=z['mask'].copy()
        t=float(z['full_time'][j]);side=str(z['full_side'][j])
        need(abs(t-float(z['slow_time'][j]))<1e-18,'stage time')
    need(x.shape==f.shape==raw.shape==mask.shape==(359373,) and
         np.isfinite(x).all() and np.isfinite(f).all() and np.isfinite(raw).all(),
         'state/rhs layout')
    need(np.all(raw[mask]==0) and side in ('left','right'),'prescribed RHS or side')
    scale=1e-7+1e-5*np.abs(x)
    magnitude=float(np.max(np.abs(raw[~mask])/scale[~mask]))
    need(np.isfinite(magnitude) and magnitude>0,'zero residual direction')
    v=raw/magnitude;v[mask]=0
    plus=x+v;minus=x-v
    need(np.isfinite(plus).all() and np.isfinite(minus).all(),'nonfinite displaced state')
    inside=(plus>=0)&(plus<=1)&(minus>=0)&(minus<=1)
    normal=float(np.max(np.abs(v)/scale))
    need(abs(normal-1)<1e-12,'unit tolerance normalization')
    np.savez_compressed(direction,z=x,v=v,archived_full_f=f,mask=mask,
                        time=np.asarray([t]),side=np.asarray([side],dtype='U5'))
    report={'schema':'jvp_real_preflight_v1','plan_sha256':sha(PLAN),
        'direction_sha256':sha(direction),'status':'READY_FOR_SIX_QUERIES' if inside.all()
             else 'STOP_DOMAIN_CROSSING',
        'stage_index':j,'stage_time_s':t,'side':side,
        'scale_norm':normal,'raw_residual_normalized_max':magnitude,
        'domain_crossing_coordinates':int((~inside).sum()),
        'first_domain_crossings':np.flatnonzero(~inside)[:12].tolist(),
        'nonzero_direction_coordinates':int(np.count_nonzero(v)),
        'wall_s':time.monotonic()-start,
        'scope':'One fixed difference direction from archived real MRI stage; model default domain [0,1].'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','domain_crossing_coordinates',
                                            'nonzero_direction_coordinates','scale_norm')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--direction',type=Path,required=True)
    args=ap.parse_args();run(args.out,args.direction)
