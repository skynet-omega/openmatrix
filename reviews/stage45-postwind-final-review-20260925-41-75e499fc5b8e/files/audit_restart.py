"""Posthoc size of restart artifact using already completed physical replays."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify_clock import raw,geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    continuous=raw(HERE/'run_01/identity/trace.npz')
    restarted=raw(HERE/'run_02/identity/trace.npz')
    for k in ('forward_mm_s','yaw_rad_s','wind_torque_native','step'):
        if not np.array_equal(continuous[k],restarted[k]):raise ValueError('Different command schedule')
    for k in ('qpos','qvel'):
        if not np.array_equal(continuous[k][:1000],restarted[k][:1000]):raise ValueError('Unequal prewind state')
    source=np.asarray(json.loads((HERE/'inputs/donor_GAUSSIAN_SPEC.json').read_text())['minus']['source_mm'])
    ca=geometry(continuous['qpos'],source);re=geometry(restarted['qpos'],source)
    result=dict(scope='Posthoc mechanics of continuous versus historical cold restart with identical commands. No new rollout and not the omitted no-wind control.',
        continuous_final_error_deg=float(ca['abs_error_deg'][-1]),
        historical_restart_final_error_deg=float(re['abs_error_deg'][-1]),
        final_error_difference_deg=float(ca['abs_error_deg'][-1]-re['abs_error_deg'][-1]),
        max_yaw_difference_deg=float(np.max(abs(ca['yaw_deg']-re['yaw_deg']))),
        max_error_difference_deg=float(np.max(abs(ca['abs_error_deg']-re['abs_error_deg']))),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_hashes={n:hashlib.sha256((HERE/n/'identity/trace.npz').read_bytes()).hexdigest() for n in ('run_01','run_02')},
        inference_limit='Difference is physically small in this frozen sequence; it does not certify arbitrary restart continuity or neuronal equivalence.')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
