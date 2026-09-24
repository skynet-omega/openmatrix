"""Read-only, exposed campaign analysis. No simulation and no admission verdict."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def angle(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def wrapped(x):
    return np.arctan2(np.sin(x), np.cos(x))


def analyze(profile, side):
    campaign = ('etapa4_mirrored_source_20260924_26' if profile == 'native'
                else 'etapa4_reference_budget_20260924_27')
    base = ROOT / 'campanas' / campaign / f'{profile}_{side}_01'
    paths = {name: base / name for name in
             ('traces.npz', 'RESULT.json', 'GAUSSIAN_SPEC.json', 'PROGRESS.jsonl')}
    result = json.loads(paths['RESULT.json'].read_text())
    spec = json.loads(paths['GAUSSIAN_SPEC.json'].read_text())[side]
    with np.load(paths['traces.npz'], allow_pickle=False) as z:
        trial = np.flatnonzero(z['fase'] == 'ensayo')
        prep = np.flatnonzero(z['fase'] == 'preparacion')[-1]
        need(len(trial) == 400 and np.array_equal(z['paso'][trial], np.arange(1, 401)), 'trial clock')
        need(np.all(np.diff(z['body_time_ns'][np.r_[prep, trial]]) == 1_000_000), 'body clock')
        selected = np.r_[prep, trial]
        pos = z['position_mm'][selected, :2]
        q = z['qpos'][selected, 3:7]
        need(np.isfinite(pos).all() and np.isfinite(q).all(), 'pose finite')
        need(np.max(abs(np.sum(q*q, axis=1)-1)) < 1e-10, 'quaternion norm')
        source = np.asarray(spec['source_mm'])
        d = source-pos
        dist = np.linalg.norm(d, axis=1)
        heading = angle(q)
        bearing_error = wrapped(np.arctan2(d[:, 1], d[:, 0])-heading)
        command = np.rad2deg(z['command_yaw_rate_rad_s'][trial])
        need(np.isfinite(command).all() and np.max(abs(command)) <= 5, 'command domain')
        forward = z['command_forward_mm_s'][trial]
        antenna = z['antenas_mm'][trial[-1], :, :2]
        offsets = antenna-pos[-1]
        geometry = []
        # Away from the source for this specific final pose; geometry-only.
        sign = -1 if bearing_error[-1] > 0 else 1
        def concentration(points):
            return np.exp(-np.sum((points-source)**2, axis=-1) / (2*spec['sigma_mm']**2))
        actual = concentration(antenna)
        need(np.max(abs(actual-z['concentracion_campo'][trial[-1]])) < 1e-12, 'field reconstruction')
        for size in (0.25, 1, 5, 30, 45):
            theta = math.radians(sign*size)
            rotation = np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta), math.cos(theta)]])
            hypothetical = concentration(pos[-1]+offsets @ rotation.T)
            geometry.append({'rigid_yaw_away_deg': sign*size,
                             'concentration_before': actual.tolist(),
                             'concentration_hypothetical': hypothetical.tolist(),
                             'change_L_minus_R': float(np.diff(actual)[0]-np.diff(hypothetical)[0])})
        measurements = dict(
            yaw_last_deg=float(z['yaw_delta_deg'][trial[-1]]),
            integrated_command_deg=float(np.sum(command)*0.001),
            last100_integrated_command_deg=float(np.sum(command[-100:])*0.001),
            max_command_abs_deg_s=float(np.max(abs(command))),
            forward_command_range_mm_s=[float(np.min(forward)), float(np.max(forward))],
            distance_start_mm=float(dist[0]), distance_end_mm=float(dist[-1]),
            distance_reduction_mm=float(dist[0]-dist[-1]),
            heading_error_start_deg=float(np.rad2deg(bearing_error[0])),
            heading_error_end_deg=float(np.rad2deg(bearing_error[-1])),
            absolute_heading_error_change_deg=float(np.rad2deg(abs(bearing_error[-1])-abs(bearing_error[0]))),
            hypothetical_rigid_antenna_rotation=geometry)
    progress = [json.loads(line) for line in paths['PROGRESS.jsonl'].read_text().splitlines()]
    steps = [r for r in progress if r['phase'] == 'ensayo']
    need([r['step'] for r in steps] == list(range(1, 401)), 'progress coverage')
    trial_wall = sum(r['step_wall_s'] for r in steps)
    mean = trial_wall / 400
    other = result['wall_total_s']-trial_wall
    need(other >= 0 and result['completed_trial_ms'] == 400, 'wall/counts')
    measurements['timing'] = {
        'observed_trial_wall_s': trial_wall, 'mean_step_s_per_sim_ms': mean,
        'observed_other_wall_s': other, 'observed_total_wall_s': result['wall_total_s'],
        'projection_not_measurement': {
            str(t): {'one_arm_h': (mean*t*1000+other)/3600,
                     'three_arms_h': 3*(mean*t*1000+other)/3600}
            for t in (2, 3)}}
    measurements['inputs'] = {str(p.relative_to(ROOT)): digest(p) for p in paths.values()}
    return measurements


def main():
    started = time.monotonic()
    runs = {f'{p}_{s}': analyze(p, s) for p in ('native', 'reference') for s in ('plus', 'minus')}
    out = {'schema': 'stage45_exposed_feasibility_v1', 'organism_runs': 0,
           'scope': 'Descriptive exposed traces, timing extrapolations and rigid geometry only; no causal admission.',
           'runs': runs,
           'command_envelope_not_passive_body_bound': {
               'cap_deg_s': 5, 'time_min_at_cap_for_30deg_s': 6,
               'time_min_at_cap_for_45deg_s': 9,
               'kick_at_s': 1.5, 'max_remaining_integral_2s_deg': 2.5,
               'max_remaining_integral_3s_deg': 7.5},
           'source_sha256': digest(Path(__file__)), 'plan_sha256': digest(OUT/'PLAN.md'),
           'cpu_wall_s': time.monotonic()-started}
    for p in ('native', 'reference'):
        a,b = runs[p+'_plus'],runs[p+'_minus']
        out[p+'_pair'] = {'yaw_plus_minus_deg': a['yaw_last_deg']-b['yaw_last_deg'],
                         'distance_reduction_plus_minus_mm': a['distance_reduction_mm']-b['distance_reduction_mm']}
    target = OUT/'RAW_FEASIBILITY_01.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(out, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'result': str(target), 'wall_s': out['cpu_wall_s'],
                      'native_pair': out['native_pair'],
                      'runs': {k: {n:v for n,v in r.items() if n not in ('inputs','hypothetical_rigid_antenna_rotation')}
                               for k,r in runs.items()}}, indent=2))


if __name__ == '__main__':
    main()
