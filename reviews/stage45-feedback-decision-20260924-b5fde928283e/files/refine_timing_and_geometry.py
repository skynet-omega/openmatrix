"""Second bounded read-only analysis: include instrumentation in cost projections.

The first result retains useful compute-step timings but treated remaining wall
cost as fixed. Elapsed progress intervals include repeated logging/checks too.
Also quantify how forward motion changes source bearing, without simulating CNS.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def main():
    start = time.monotonic()
    original = OUT / 'RAW_FEASIBILITY_01.json'
    r = json.loads(original.read_text())
    for name, run in r['runs'].items():
        inputs = run['inputs']
        for name_in, expected in inputs.items():
            need(sha(ROOT/name_in) == expected, 'changed input: '+name_in)
        progress_path = next(ROOT/p for p in inputs if p.endswith('PROGRESS.jsonl'))
        progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
        trial = [x for x in progress if x['phase'] == 'ensayo']
        need([x['step'] for x in trial] == list(range(1,401)), 'clock coverage')
        avg = (trial[-1]['elapsed_s']-trial[0]['elapsed_s'])/399
        fixed = run['timing']['observed_total_wall_s']-400*avg
        need(fixed >= 0, 'overhead extrapolation')
        run['timing']['elapsed_progress_rate_s_per_sim_ms'] = avg
        run['timing']['inferred_nontrial_wall_s'] = fixed
        run['timing']['projection_not_measurement'] = {
            str(t): {'one_arm_h': (fixed+avg*t*1000)/3600,
                     'three_arms_h': 3*(fixed+avg*t*1000)/3600}
            for t in (2,3)}
        run['timing']['projection_method'] = '(elapsed step400 - elapsed step1)/399; other cost inferred from observed total400ms; not a long-run benchmark'
        # kinematic derivative beta_dot = (dy*vx-dx*vy)/distance^2
        # for nominal forward velocity aligned with initial body yaw.
        err = math.radians(run['heading_error_start_deg'])
        run['nominal_initial_bearing_rate_due_forward_deg_s'] = math.degrees(
            run['forward_command_range_mm_s'][0]*math.sin(err)/run['distance_start_mm'])
        path = next(ROOT/p for p in inputs if p.endswith('traces.npz'))
        with np.load(path, allow_pickle=False) as z:
            indices = np.r_[np.flatnonzero(z['fase']=='preparacion')[-1], np.flatnonzero(z['fase']=='ensayo')]
            w,x,y,zq = z['qpos'][indices,3:7].T
            yaw = np.unwrap(np.arctan2(2*(w*zq+x*y), 1-2*(y*y+zq*zq)))
            yaw_change = math.degrees(yaw[-1]-yaw[0])
            run['mean_observed_heading_rate_deg_s'] = yaw_change/0.4
            error_change = run['heading_error_end_deg']-run['heading_error_start_deg']
            run['mean_observed_source_bearing_rate_deg_s'] = (error_change+yaw_change)/0.4
    r['schema'] = 'stage45_exposed_feasibility_v2'
    r['supersedes_only_timing_extrapolation_of'] = {'file': original.name, 'sha256': sha(original)}
    r['second_analysis_source_sha256'] = sha(Path(__file__))
    r['second_analysis_wall_s'] = time.monotonic()-start
    r['interpretation_limits'] = [
        'Heading error and kinematic rates are exposed descriptive metrics, not preregistered admission.',
        'A longer run may change neural commands; these rates are not long-term predictions.',
        'Rigid antenna rotations are hypothetical inputs, not physically reachable states or feedback responses.',
        'Timing scales measured progress; long-run stiffness, I/O and stability remain unmeasured.'
    ]
    target = OUT/'RAW_FEASIBILITY_02.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(r, f, indent=2, allow_nan=False);f.write('\n')
    print(json.dumps({k:{'cost_projection':x['timing']['projection_not_measurement'],
                         'heading_rate_deg_s':x['mean_observed_heading_rate_deg_s'],
                         'source_bearing_rate_deg_s':x['mean_observed_source_bearing_rate_deg_s']}
                      for k,x in r['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
