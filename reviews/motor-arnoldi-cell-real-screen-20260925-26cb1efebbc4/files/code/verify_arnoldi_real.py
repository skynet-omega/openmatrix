"""Independently recompute two-column screen and original-organism neutrality."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from verify_cell_kernel_timing import compare_tree, need, sha

HERE = Path(__file__).resolve().parent
BASE = HERE / 'event_step_baseline_v2_01'
H = 125e-6


def same(a, b, label):
    need(np.array_equal(a, b), label + ' changed')


def projection(mat, fraction):
    a = np.zeros((3, 3))
    a[:2, :2] = mat
    a[0, 2] = 1
    return expm(fraction * a)[:2, 2]


def verify(run):
    result = json.loads((run / 'ARNOLDI_REAL_RESULT.json').read_text())
    plan = json.loads((HERE / 'ARNOLDI_REAL_PLAN_68.json').read_text())
    lock = json.loads((HERE / 'ARNOLDI_REAL_EXEC_LOCK_69.json').read_text())
    need(result['schema'] == 'arnoldi_real_result_v1' and
         result['status'] == 'COMPLETE_DIAGNOSTIC_ONLY' and result['budget_ok'] and
         result['diagnostic_error'] is None and result['tested_blocks'] == 1 and
         result['physical_epochs'] == 16 and result['probe']['full_queries'] == 6,
         'run receipt')
    need(sha(HERE / 'probe_arnoldi_real.py') == lock['source_sha256'] == result['source_sha256'] and
         sha(HERE / 'ARNOLDI_REAL_PLAN_68.json') == lock['plan_sha256'] == result['plan_sha256'],
         'execution lock')
    for rel, digest in plan['frozen_sha256'].items():
        need(sha(HERE.parents[1] / rel) == digest, 'frozen source: ' + rel)
    probe = result['probe']
    need(sha(run / 'ARNOLDI_REAL_ARRAYS.npz') == probe['arrays_sha256'], 'array hash')
    with np.load(run / 'ARNOLDI_REAL_ARRAYS.npz', allow_pickle=False) as a, \
         np.load(HERE / 'mri_jvp_downloaded_01/capsule/data/JVP_REAL_ARRAYS.npz',
                 allow_pickle=False) as old, \
         np.load(HERE / 'mri_jvp_downloaded_01/capsule/data/MRI_SLOW_INPUTS.npz',
                 allow_pickle=False) as slow_archived:
        z = a['z']; D = a['D']; mask = a['mask']; slow = a['slow']
        points = a['points']; f = a['full_f']; q = a['q']; K = a['K']
        need(z.shape == D.shape == mask.shape == slow.shape == (359373,) and
             points.shape == f.shape == (6, 359373) and q.shape == (3, 359373) and
             K.shape == (3, 2) and mask.dtype == np.bool_ and
             np.isfinite(z).all() and np.isfinite(D).all() and
             np.isfinite(f).all() and np.isfinite(q).all() and np.isfinite(K).all(),
             'array layout/finitude')
        same(z, old['z'], 'initial state')
        same(z, slow_archived['full_z'][1], 'slow state')
        same(slow, slow_archived['slow_f'][1], 'slow forcing')
        same(mask, old['mask'], 'prescribed mask')
        same(D, 1e-7 + 1e-5 * np.abs(z), 'scale')
        same(points[0], z, 'first input')
        same(points[5], z, 'repeat input')
        same(f[0], old['full_f'][0], 'prior baseline RHS')
        same(f[0], f[5], 'repeat RHS')
        same(f[1], old['full_f'][1], 'prior positive JVP')
        same(f[2], old['full_f'][2], 'prior negative JVP')
        beta = float(np.linalg.norm(old['v'] / D))
        need(np.allclose(q[0], old['v'] / D / beta, rtol=1e-12, atol=1e-12),
             'first Krylov vector')
        need(np.all(q[:, mask] == 0), 'prescribed Krylov coordinates')
        recomputed_K = np.zeros_like(K)
        recomputed_even = []
        for column in range(2):
            amplitude = 1.0 / np.max(np.abs(q[column]))
            step = amplitude * D * q[column]
            step[mask] = 0.0
            same(points[2 * column + 1], z + step, 'positive input')
            same(points[2 * column + 2], z - step, 'negative input')
            need(np.all((z + step >= 0) & (z + step <= 1) &
                        (z - step >= 0) & (z - step <= 1)), 'domain')
            plus = f[2 * column + 1]
            minus = f[2 * column + 2]
            recomputed_even.append(float(np.max((H * np.abs((plus + minus) / 2 - f[0]) / D)[~mask])))
            w = H * (plus - minus) / (2 * amplitude * D)
            residual = w.copy()
            for _ in range(2):
                for j in range(column + 1):
                    coefficient = float(q[j] @ residual)
                    recomputed_K[j, column] += coefficient
                    residual -= coefficient * q[j]
            recomputed_K[column + 1, column] = np.linalg.norm(residual)
            need(np.allclose(q[column + 1], residual / recomputed_K[column + 1, column],
                             rtol=1e-12, atol=1e-12), 'next Krylov vector')
        need(np.allclose(K, recomputed_K, rtol=1e-11, atol=1e-11), 'Hessenberg matrix')
        need(np.allclose(a['even'], recomputed_even, rtol=1e-12, atol=1e-12), 'even remainder arrays')
        need(np.allclose(probe['even_remainders'], recomputed_even,
                         rtol=1e-12, atol=1e-12), 'even remainder receipt')
        forcing = H * slow / D
        need(np.allclose(forcing, np.linalg.norm(forcing) * q[0],
                         rtol=1e-12, atol=1e-11), 'forcing direction')
        fractions = a['sample_fractions']
        same(fractions, np.array([.25, .5, .75, 1.0]), 'sample fractions')
        sampled = np.array([float(np.linalg.norm(forcing) * np.max(np.abs(K[2, 1] * q[2])) *
                                  abs(projection(K[:2, :2], fraction)[1]))
                            for fraction in fractions])
        need(np.allclose(sampled, a['sampled_residual'], rtol=1e-11, atol=1e-11),
             'sampled residual arrays')
        need(np.allclose(sampled, probe['sampled_residuals'], rtol=1e-11, atol=1e-11) and
             math.isclose(float(max(sampled)), probe['sampled_residual_max'], rel_tol=1e-11),
             'sampled residual receipt')
        need(probe['even_gate_pass'] == (max(recomputed_even) <= .1) and
             probe['sampled_gate_pass'] == (max(sampled) <= .1), 'prospective gates')
        corrupted = sampled.copy()
        corrupted[0] += .5
        need(not np.allclose(corrupted, a['sampled_residual']), 'corruption detector')
    need(json.loads((BASE / 'EVENT_AUDIT.json').read_text()) ==
         json.loads((run / 'EVENT_AUDIT.json').read_text()), 'parent events changed')
    for rel in ('traces.npz', 'final_state/published.json', 'final_state/published.npz',
                'final_state/prosthesis.json', 'final_state/prosthesis.npz',
                'preparation_inputs/intervenciones_W.npz'):
        need(sha(BASE / rel) == sha(run / rel), 'parent output changed: ' + rel)
    with np.load(BASE / 'final_state/session.npz') as a, \
         np.load(run / 'final_state/session.npz') as b:
        compare_tree(json.loads((BASE / 'final_state/session.json').read_text()),
                     json.loads((run / 'final_state/session.json').read_text()), a, b)
    return {'schema': 'arnoldi_real_verify_v1', 'status': 'PASS_DIAGNOSTIC_ONLY',
            'even_remainder_max': max(recomputed_even),
            'sampled_residual_max': float(max(sampled)),
            'parent_output_exact': True, 'corruption_rejected': True,
            'scope': 'One state and two-column sampled Arnoldi residual; no continuous error, integrated speed or behavioral admission.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'output exists')
    report = verify(args.run)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))
