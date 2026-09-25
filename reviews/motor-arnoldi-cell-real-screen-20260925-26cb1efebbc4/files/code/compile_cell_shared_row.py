"""Preflight PTX resource use for the real shared-row cell kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from cell_shared_row_source import shared_row_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'CELL_SHARED_ROW_PLAN_73.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out):
    start = time.monotonic()
    out = Path(out).resolve()
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'cell_shared_row_plan_v1' and not out.exists(), 'plan/output')
    for rel, digest in plan['frozen_sha256'].items():
        need(sha(ROOT / rel) == digest, 'source changed: ' + rel)
    import cupy as cp
    sys.path[:0] = [str(ROOT / 'motor_nuevo/epoch_cost_20260923'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922/vendor')]
    import device_cell
    original = device_cell.source()
    candidate = shared_row_source(original)
    kernel = cp.RawKernel(candidate, 'cell_epoch', options=('--fmad=false',))
    t = time.monotonic()
    kernel.compile()
    attrs = kernel.attributes
    report = {'schema': 'cell_shared_row_compile_v1', 'plan_sha256': sha(PLAN),
              'generator_sha256': sha(HERE / 'cell_shared_row_source.py'),
              'source_sha256': sha(Path(__file__)),
              'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'candidate_source_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
              'compile_s': time.monotonic() - t, 'kernel_attributes': attrs,
              'compile_gate_pass': attrs['local_size_bytes'] == 0 and attrs['num_regs'] < 200,
              'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report['budget_ok'] = report['wall_s'] <= plan['budget']['wall_s_max'] and \
        report['maxrss_kib'] <= plan['budget']['ram_gib_max'] * 1024 ** 2
    out.mkdir(parents=True)
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    need(report['budget_ok'] and (out / 'RESULT.json').stat().st_size <= plan['budget']['disk_bytes_max'],
         'budget')
    print(json.dumps({'attributes': attrs, 'compile_gate_pass': report['compile_gate_pass']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
