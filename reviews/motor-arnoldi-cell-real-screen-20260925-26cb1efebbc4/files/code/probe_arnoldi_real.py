"""Two-column, six-query Arnoldi diagnostic on one complete effective real RHS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'ARNOLDI_REAL_PLAN_68.json'
LOCK = HERE / 'ARNOLDI_REAL_EXEC_LOCK_69.json'
CAPSULE = HERE / 'mri_jvp_downloaded_01/capsule'
H = 125e-6


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def orthogonal(w, basis):
    residual = w.copy()
    coefficients = np.zeros(len(basis))
    for _ in range(2):
        for i, q in enumerate(basis):
            value = float(q @ residual)
            coefficients[i] += value
            residual -= value * q
    return coefficients, residual


def phi_action(mat, fraction):
    m = len(mat)
    augmented = np.zeros((m + 1, m + 1))
    augmented[:-1, :-1] = mat
    augmented[0, -1] = 1.0
    return expm(fraction * augmented)[:-1, -1]


def replay(parent, candidate, direction, slow, mask, t, out):
    import cupy as cp
    from effective_oracle import EffectiveOracle
    oracle = EffectiveOracle(parent)
    D = 1e-7 + 1e-5 * np.abs(candidate)
    beta = float(np.linalg.norm(direction / D))
    need(beta > 0 and abs(np.max(np.abs(direction / D)) - 1) < 1e-12,
         'invalid archived one-tolerance direction')
    q1 = direction / D / beta
    expected_b = H * slow / D
    need(np.allclose(expected_b, (H * np.max(np.abs(slow / D)) * beta) * q1,
                     rtol=1e-12, atol=1e-11), 'forcing not collinear with q1')
    base_scale = float(np.linalg.norm(expected_b))
    need(np.isfinite(base_scale) and base_scale > 0, 'forcing magnitude')
    fields = []
    points = []

    def query(x):
        need(len(fields) < 6 and x.shape == candidate.shape and np.isfinite(x).all(),
             'oracle query contract')
        need(np.all(x >= 0) and np.all(x <= 1), 'candidate outside default domain')
        with oracle.stream:
            gpu_x = cp.asarray(x, dtype=cp.float64)
        z, a, r = oracle.query(gpu_x, t)
        with oracle.stream:
            f = cp.asnumpy(r * (a - z), stream=oracle.stream)
        oracle.stream.synchronize()
        need(f.shape == candidate.shape and np.isfinite(f).all(), 'oracle output')
        f[mask] = 0.0
        points.append(x.copy())
        fields.append(f)
        return f

    f0 = query(candidate)
    archived = np.load(CAPSULE / 'data/JVP_REAL_ARRAYS.npz', allow_pickle=False)
    need(np.array_equal(f0, archived['full_f'][0]), 'baseline differs from prior same preparation')
    q = [q1]
    K = np.zeros((3, 2))
    even = []
    for column in range(2):
        amplitude = 1.0 / float(np.max(np.abs(q[column])))
        step = amplitude * D * q[column]
        step[mask] = 0.0
        plus = query(candidate + step)
        minus = query(candidate - step)
        if column == 0:
            need(np.array_equal(plus, archived['full_f'][1]) and
                 np.array_equal(minus, archived['full_f'][2]),
                 'first JVP differs from prior same preparation')
        even.append(float(np.max((H * np.abs((plus + minus) / 2 - f0) / D)[~mask])))
        w = H * (plus - minus) / (2 * amplitude * D)
        h, residual = orthogonal(w, q)
        K[:column + 1, column] = h
        norm = float(np.linalg.norm(residual))
        K[column + 1, column] = norm
        need(np.isfinite(norm) and norm > 0, 'Arnoldi numerical breakdown')
        q.append(residual / norm)
    repeated = query(candidate)
    need(oracle.calls == 6 and np.array_equal(f0, repeated), 'oracle nonrepeatable')
    sampled = []
    for fraction in (.25, .5, .75, 1.0):
        projected = phi_action(K[:2, :2], fraction)
        sampled.append(float(base_scale * np.max(np.abs(K[2, 1] * q[2])) *
                             abs(projected[1])))
    arrays = out / 'ARNOLDI_REAL_ARRAYS.npz'
    np.savez_compressed(arrays, z=candidate, D=D, mask=mask, slow=slow,
                        points=np.stack(points), full_f=np.stack(fields),
                        q=np.stack(q), K=K, even=np.asarray(even),
                        sampled_residual=np.asarray(sampled),
                        sample_fractions=np.asarray([.25, .5, .75, 1.0]),
                        t=np.asarray([t]))
    return {'full_queries': oracle.calls, 'arrays_sha256': sha(arrays),
            'baseline_repeat_exact': True, 'prior_q1_exact': True,
            'k11': float(K[0, 0]), 'k21': float(K[1, 0]),
            'k12': float(K[0, 1]), 'k22': float(K[1, 1]),
            'k32': float(K[2, 1]), 'even_remainders': even,
            'sampled_residuals': sampled,
            'sampled_residual_max': max(sampled),
            'even_gate_pass': max(even) <= .1,
            'sampled_gate_pass': max(sampled) <= .1,
            'scope': 'One fixed state, one two-column projection, sampled residual only; no integration or speed claim.'}


def run(out):
    start = time.monotonic()
    out = Path(out).resolve()
    need(__debug__ and not out.exists(), 'normal Python and unique output required')
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'arnoldi_real_plan_v1', 'plan')
    lock = json.loads(LOCK.read_text())
    need(lock['plan_sha256'] == sha(PLAN) and lock['source_sha256'] == sha(Path(__file__)),
         'execution lock')
    for rel, expected in plan['frozen_sha256'].items():
        need(sha(ROOT / rel) == expected, 'source changed: ' + rel)
    with np.load(CAPSULE / 'data/MRI_SLOW_INPUTS.npz', allow_pickle=False) as archived:
        candidate = archived['full_z'][1].copy()
        slow = archived['slow_f'][1].copy()
        mask = archived['mask'].copy()
        t = float(archived['full_time'][1])
    with np.load(CAPSULE / 'data/JVP_REAL_ARRAYS.npz', allow_pickle=False) as archived:
        direction = archived['v'].copy()
        need(np.array_equal(candidate, archived['z']) and
             np.array_equal(mask, archived['mask']) and t == float(archived['time'][0]),
             'archived stage mismatch')
    import cupy as cp
    sys.path[:0] = [str(ROOT / 'motor_nuevo/epoch_cost_20260923'),
                    str(ROOT / 'motor_nuevo/pipeline_review_20260922'),
                    str(ROOT / 'motor_nuevo/multirate_real_20260924_01')]
    import run_set
    from runtime_session import RuntimeSession
    context = {'physical_epochs': 0, 'tested_blocks': 0, 'diagnostic_error': None}
    original_init = RuntimeSession.__init__

    def installed(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        adapter = session.adapter
        original_build = adapter.build

        def build(drive, light):
            original_build(drive, light)
            core = adapter.core
            old_advance = core.advance
            context['core'] = core
            context['old_advance'] = old_advance

            def advance(*aa, **kk):
                if context['active_epoch'] != 1:
                    return old_advance(*aa, **kk)
                need(context['tested_blocks'] == 0 and aa and aa[0] == 125000,
                     'wrong physical block')
                parent = old_advance(*aa, **kk)
                context['tested_blocks'] = 1
                try:
                    context['probe'] = replay(adapter, candidate, direction, slow, mask, t, out)
                except BaseException:
                    context['diagnostic_error'] = traceback.format_exc()
                return parent

            core.advance = advance

        adapter.build = build
        original_step = session.events.step

        def step(brain, ns, drive, light):
            context['active_epoch'] = context['physical_epochs']
            context['physical_epochs'] += 1
            return original_step(brain, ns, drive, light)

        session.events.step = step

    RuntimeSession.__init__ = installed
    old_argv = sys.argv
    sys.argv = ['run_set.py', '--out', str(out), '--odor', 'sham', '--engine', 'causal_cuda',
                '--ms', '1', '--observe', 'off', '--cuda-profile', 'off', '--profile', 'off',
                '--kc-capture', 'off']
    try:
        code = run_set.main()
    finally:
        sys.argv = old_argv
        RuntimeSession.__init__ = original_init
        if 'core' in context:
            context['core'].advance = context['old_advance']
    parent_result = json.loads((out / 'RESULT.json').read_text())
    report = {'schema': 'arnoldi_real_result_v1', 'plan_sha256': sha(PLAN),
              'source_sha256': sha(Path(__file__)), 'runner_exit_code': code,
              'runner_status': parent_result['status'],
              'physical_epochs': context['physical_epochs'],
              'tested_blocks': context['tested_blocks'],
              'diagnostic_error': context['diagnostic_error'],
              'probe': context.get('probe'),
              'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'disk_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    budget = plan['budget']
    report['budget_ok'] = report['wall_s'] <= budget['wall_s_max'] and \
        report['maxrss_kib'] <= budget['ram_gib_max'] * 1024 ** 2 and \
        report['disk_bytes'] <= budget['disk_bytes_max']
    report['status'] = 'COMPLETE_DIAGNOSTIC_ONLY' if (
        code == 0 and parent_result['status'] == 'COMPLETE' and report['budget_ok'] and
        report['probe'] is not None and context['diagnostic_error'] is None and
        context['tested_blocks'] == 1 and context['probe']['full_queries'] == 6
    ) else 'INCOMPLETE_RETAINED'
    (out / 'ARNOLDI_REAL_RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'tested_blocks', 'wall_s', 'budget_ok')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
