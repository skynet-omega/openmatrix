"""CPU-only evidence review. No simulator imports and no neuronal steps."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_npz(name):
    with np.load(HERE/'data'/name, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def verify_reader(s, o, plan, recorded_plan_sha, exact_plan_sha, previous_dn_at_start):
    """Supplemental reconstruction, not a claim of mechanical/brain fidelity."""
    need(recorded_plan_sha == exact_plan_sha, 'Frozen plan identity changed')
    n = plan['duration_ms']
    on, off = plan['odor_on_ms'], plan['odor_off_ms']
    for name, trace in [('sham', s), ('odor', o)]:
        previous = np.asarray(previous_dn_at_start[name])
        need(previous.shape == (4,) and np.isfinite(previous).all(), name+': initial DN shape')
        need(np.array_equal(trace['paso'], np.arange(1, n+1)), name+': time sequence')
        for key, value in trace.items():
            need(value.shape[0] == n, name+': trace length '+key)
            if value.dtype.kind in 'fc':
                need(np.isfinite(value).all(), name+': nonfinite '+key)
        baseline = trace['DN_baseline']
        need(baseline.shape == (n, 4), name+': DN shape')
        need(np.array_equal(trace['DN_q_usada'][0], previous), name+': initial motor lag')
        need(np.array_equal(baseline, np.broadcast_to(baseline[0], (n, 4))), name+': baseline drift')
        need(np.array_equal(trace['DN_q_usada'][1:], trace['DN_q_actual'][:-1]), name+': motor lag')
        raw = np.mean(trace['DN_q_usada'][:, :2]-baseline[:, :2], axis=1)
        need(np.array_equal(trace['command_forward_mm_s'], np.clip(raw, 0., .5)), name+': decoder mismatch')
        need(np.all(trace['command_yaw_rate_rad_s'] == 0.), name+': applied yaw')
        expected = np.zeros((n, 3))
        if name == 'odor':
            expected[on:off, :2] = plan['odor_amplitude']
        need(np.array_equal(trace['sensores_usados'], expected), name+': consumed schedule')
    need(np.array_equal(s['DN_baseline'], o['DN_baseline']), 'Paired baselines differ')
    # DN output published at the end of interval on+1 is available for on+2.
    need(np.array_equal(s['command_forward_mm_s'][:on+1], o['command_forward_mm_s'][:on+1]), 'Premature propulsive response')
    return True


def original_fixture(analyzer, plan, root, command_interval=None, changed_threshold=False):
    """Adversarial synthetic evidence; never labelled a real organism run."""
    n = plan['duration_ms']
    t = {'paso': np.arange(1, n+1), 'position_mm': np.zeros((n, 3)),
         'qvel': np.zeros((n, 108)), 'yaw_delta_deg': np.zeros(n),
         'DN_q_actual': np.zeros((n, 4)), 'DN_q_usada': np.zeros((n, 4)),
         'DN_baseline': np.zeros((n, 4)), 'command_forward_mm_s': np.zeros(n),
         'command_yaw_rate_rad_s': np.zeros(n), 'sensores_usados': np.zeros((n, 3)),
         'ORN_q_L': np.zeros((n, 1)), 'ORN_q_R': np.zeros((n, 1))}
    s, o = copy.deepcopy(t), copy.deepcopy(t)
    o['sensores_usados'][1000:3000, :2] = plan['odor_amplitude']
    o['position_mm'][1000:, 0] = np.arange(1, 3001)*.02*.001
    o['qvel'][1000:, 0] = .02/10.
    if command_interval is not None:
        o['command_forward_mm_s'][command_interval-1] = .1
    fixture_plan = copy.deepcopy(plan)
    if changed_threshold:
        fixture_plan['movement_displacement_difference_mm'] = .1
    root.mkdir()
    save(root/'PLAN.json', fixture_plan)
    r = dict(initial_ns=0, parameters={}, stored_weight_digest='fixture', factors_digest='fixture',
             plan_sha256=sha(HERE/'snapshots/PLAN45.json'))
    for name in ('sham', 'odor'):
        (root/name).mkdir()
        save(root/name/'RESULT.json', r)
    analyzer.HERE = root
    analyzer.read_arm = lambda name: (r, s if name == 'sham' else o)
    # The original prints a full result; keep the reproducibility log concise.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        analyzer.main()
    result = json.loads((root/'RESULTADOS.json').read_text())
    return s, o, result, fixture_plan


def main(out):
    start = time.monotonic()
    cpu = time.process_time()
    out = Path(out)
    out.mkdir(exist_ok=False)
    manifest = json.loads((HERE/'INPUTS.json').read_text())
    for name, digest in manifest['review_files'].items():
        need(sha(HERE/name) == digest, 'Review input changed: '+name)
    a, b, long = (load_npz(name) for name in ['stable2s.npz', 'reviewed2s.npz', 'exploratory12s.npz'])
    threshold = .0005
    alpha = -math.expm1(-.001/.2)
    output = math.radians(5.)
    reconstruction = {}
    for name, trace in [('stable', a), ('reviewed', b), ('exploratory12s', long)]:
        dq = trace['DN_q_usada']-trace['DN_baseline']
        raw = np.asarray([float(np.tanh(250.*(d[2]-d[3]))*output) for d in dq])
        state = 0.
        filtered = []
        for value in raw:
            state += alpha*(float(value)-state)
            filtered.append(state)
        filtered = np.asarray(filtered)
        applied = np.asarray([math.copysign(output, f) if abs(f) >= threshold else 0. for f in filtered])
        need(np.array_equal(raw, trace['neural_command_raw_rad_s']), name+': raw decoder reconstruction')
        need(np.array_equal(filtered, trace['motor_filter_state_rad_s']), name+': filter reconstruction')
        need(np.array_equal(applied, trace['command_yaw_rate_rad_s']), name+': relay reconstruction')
        reconstruction[name] = {'raw_exact': True, 'filter_exact': True, 'relay_exact': True}
    changed = np.flatnonzero(a['command_yaw_rate_rad_s'] != b['command_yaw_rate_rad_s'])
    pair = []
    for i in changed:
        pair.append({'interval_ms': int(i+1),
                     'stable_filter_rad_s': float(a['motor_filter_state_rad_s'][i]),
                     'reviewed_filter_rad_s': float(b['motor_filter_state_rad_s'][i]),
                     'filter_difference_rad_s': float(b['motor_filter_state_rad_s'][i]-a['motor_filter_state_rad_s'][i]),
                     'threshold_rad_s': threshold,
                     'stable_applied_deg_s': float(np.rad2deg(a['command_yaw_rate_rad_s'][i])),
                     'reviewed_applied_deg_s': float(np.rad2deg(b['command_yaw_rate_rad_s'][i]))})
    long_metrics = {'ticks': len(long['paso']), 'DNg100_release_initial': long['DN_q_actual'][0, :2].tolist(),
                    'DNg100_delta_range': np.ptp(long['DN_q_actual'][:, :2], axis=0).tolist(),
                    'DNg100_meaning': 'Numerically subnormal release and no modulation in this exposure, not biological inactivity proof',
                    'forward_target_unique_mm_s': np.unique(long['command_forward_mm_s']).tolist(),
                    'raw_yaw_mean_6_12_deg_s': float(np.rad2deg(long['neural_command_raw_rad_s'][6000:]).mean()),
                    'applied_yaw_unique_6_12_deg_s': np.unique(np.rad2deg(long['command_yaw_rate_rad_s'][6000:])).tolist(),
                    'applied_nonzero_sign_disagrees_raw_count': int(np.sum((long['command_yaw_rate_rad_s'] != 0.) &
                          (np.sign(long['command_yaw_rate_rad_s']) != np.sign(long['neural_command_raw_rad_s']))))}
    sys.path.insert(0, str(HERE/'snapshots'))
    spec = importlib.util.spec_from_file_location('frozen_analysis45', HERE/'snapshots/analyze45.py')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    plan = json.loads((HERE/'snapshots/PLAN45.json').read_text())
    tests = []
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        s, o, result, _ = original_fixture(analyzer, plan, tmp/'body_only')
        no_neural_command = np.all(o['DN_q_actual'] == s['DN_q_actual']) and np.all(o['command_forward_mm_s'] == 0.)
        need(no_neural_command and result['operational_initiation_criterion_pass'], 'Counterexample not reproduced')
        initial = {'sham': np.zeros(4), 'odor': np.zeros(4)}  # Defined synthetic initial state only.
        verify_reader(s, o, plan, sha(HERE/'snapshots/PLAN45.json'), sha(HERE/'snapshots/PLAN45.json'), initial)
        tests.append({'case': 'physical_change_without_DN_or_command', 'original_operational_criterion': True,
                      'reader_contract_consistent': True, 'interpretation': 'Physical criterion alone does not certify DN-driven propulsion; fixture is not a mechanically reproduced run'})
        s, o, result, _ = original_fixture(analyzer, plan, tmp/'premature_command', command_interval=1001)
        try:
            verify_reader(s, o, plan, sha(HERE/'snapshots/PLAN45.json'), sha(HERE/'snapshots/PLAN45.json'), initial)
        except ValueError as exc:
            tests.append({'case': 'command_interval1001_inconsistent_with_DN', 'original_accepts': True,
                          'original_first_command': result['first_positive_command_difference_interval'],
                          'supplement_rejects': str(exc)})
        else:
            raise ValueError('Decoder corruption escaped supplement')
        s, o, result, changed_plan = original_fixture(analyzer, plan, tmp/'changed_criterion', changed_threshold=True)
        need(result['operational_initiation_criterion_pass'] is False, 'Changed criterion counterexample')
        try:
            verify_reader(s, o, changed_plan, sha(HERE/'snapshots/PLAN45.json'), sha(tmp/'changed_criterion/PLAN.json'), initial)
        except ValueError as exc:
            tests.append({'case': 'changed_PLAN_after_acquisition', 'original_accepts_and_verdict_changes': True,
                          'supplement_rejects': str(exc)})
        else:
            raise ValueError('Criterion corruption escaped supplement')
    save(out/'RESULTADOS.json', {'status': 'CPU_REVIEW_COMPLETE', 'neural_steps': 0, 'reconstruction': reconstruction,
         'pair2s_relay_boundary': pair, 'historical_FAIL_preserved': True, 'exploratory12s': long_metrics,
         'adversarial_synthetic_tests': tests, 'no_actual_campaign45_corruption_claim': True,
         'CPU_s': time.process_time()-cpu, 'wall_s': time.monotonic()-start,
         'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    print(json.dumps(json.loads((out/'RESULTADOS.json').read_text()), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
