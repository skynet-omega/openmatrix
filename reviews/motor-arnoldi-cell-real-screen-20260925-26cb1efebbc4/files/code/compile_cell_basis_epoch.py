"""Compile-only certified basis variant of the existing physical epoch kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'CELL_BASIS_COMPILE_PLAN_71.json'


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out):
    start = time.monotonic()
    out = Path(out).resolve()
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'cell_basis_compile_plan_v1' and not out.exists(), 'plan/output')
    for rel, digest in plan['frozen_sha256'].items():
        need(sha(ROOT / rel) == digest, 'source/data changed: ' + rel)
    sys.path[:0] = [str(ROOT / 'campanas/etapa3_motor_nuevo_20260922'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922/vendor'),
                    str(ROOT / 'motor_nuevo/epoch_cost_20260923')]
    from verify_transport import read_state
    from basis_compile import compile_basis, compile_warp, residual_certificate
    import device_cell
    import cupy as cp
    state = read_state(ROOT / 'motor_nuevo/native_hybrid_20260922/baseline_01/brain_final')
    cell = state['kc_spatial_manifest']['cell']
    G = np.asarray(cell['channel_G_nS']).reshape(51, 17, 17)
    b = np.asarray(cell['channel_b_nS']).reshape(51, 17)
    ena = np.tile(np.asarray([60., 60., -80.]) - cell['rest_mV'], 17)
    cg, cb, groups, summary = compile_basis(G, b)
    need(len(groups) == 15 and summary['original_terms'] == 51 and
         summary['max_relative_basis_residual'] < summary['admission_relative_bound'],
         'historical basis changed')
    certificate = residual_certificate(G, b, cg, cb, groups, ena)
    original = device_cell.source()
    candidate = compile_warp(original, groups, unroll=False, certificate=certificate)
    # The replaced 51-term channel loop contained one now-absent inner pragma.
    need(original.count('#pragma unroll') == 10 and candidate.count('#pragma unroll') == 9,
         'unrelated loop policy changed')
    t = time.monotonic()
    kernel = cp.RawKernel(candidate, 'cell_epoch', options=('--fmad=false',))
    kernel.compile()
    report = {'schema': 'cell_basis_compile_result_v1', 'plan_sha256': sha(PLAN),
              'source_sha256': sha(Path(__file__)),
              'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'candidate_source_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
              'basis': summary, 'compiled_terms': len(groups),
              'compile_s': time.monotonic() - t, 'kernel_attributes': kernel.attributes,
              'candidate_local_spill': kernel.attributes['local_size_bytes'] > 0,
              'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report['budget_ok'] = report['wall_s'] <= plan['budget']['wall_s_max'] and \
        report['maxrss_kib'] <= plan['budget']['ram_gib_max'] * 1024 ** 2
    out.mkdir(parents=True)
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    need(report['budget_ok'] and (out / 'RESULT.json').stat().st_size <= plan['budget']['disk_bytes_max'],
         'budget')
    print(json.dumps({'compiled_terms': len(groups), 'attributes': kernel.attributes,
                      'candidate_local_spill': report['candidate_local_spill']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
