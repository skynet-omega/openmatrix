"""Compile-only resource screen of the existing cell kernel and no-forced-unroll variant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'CELL_LOOP_SCHEDULE_PLAN_64.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out):
    start = time.monotonic()
    out = Path(out).resolve()
    need(not out.exists(), 'output must be unique')
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'cell_loop_schedule_plan_v1', 'plan schema')
    for rel, expected in plan['frozen_sha256'].items():
        need(sha(ROOT / rel) == expected, 'source changed: ' + rel)
    import cupy as cp
    sys.path[:0] = [str(ROOT / 'motor_nuevo/epoch_cost_20260923'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922'),
                    str(ROOT / 'motor_nuevo/native_hybrid_20260922/vendor')]
    import device_cell
    original = device_cell.source()
    marker = '\n#pragma unroll\n'
    need(original.count(marker) == 10, 'expected ten injected directives')
    variant = original.replace(marker, '\n')
    need(variant.count('#pragma unroll') == 0, 'unexpected residual directive')
    out.mkdir(parents=True)
    records = []
    for name, source in [('control', original), ('no_forced_unroll', variant)]:
        elapsed = time.monotonic() - start
        need(elapsed < plan['budget']['compile_wall_s_max'], 'wall budget')
        kernel = cp.RawKernel(source, 'cell_epoch', options=('--fmad=false',))
        t = time.monotonic()
        kernel.compile()
        records.append({'name': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'compile_s': time.monotonic() - t, 'attributes': kernel.attributes})
    report = {'schema': 'cell_loop_schedule_compile_v1', 'plan_sha256': sha(PLAN),
              'source_sha256': sha(Path(__file__)), 'removed_directives': 10,
              'variants': records, 'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report['budget_ok'] = report['wall_s'] <= plan['budget']['compile_wall_s_max'] and \
        report['maxrss_kib'] <= plan['budget']['ram_gib_max'] * 1024 ** 2
    need(report['budget_ok'], 'budget exceeded')
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= plan['budget']['disk_bytes_max'],
         'disk budget exceeded')
    print(json.dumps({'variants': records, 'wall_s': report['wall_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
