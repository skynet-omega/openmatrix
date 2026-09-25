"""Read-only whole-organism run followed by isolated cell-kernel clone launches."""
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
PLAN = HERE / 'CELL_ISOLATED_REPLAY_PLAN_65.json'


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(out):
    start = time.monotonic()
    out = Path(out).resolve()
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'cell_isolated_replay_plan_v1' and not out.exists(), 'plan/output')
    for rel, expected in plan['frozen_sha256'].items():
        need(sha(ROOT / rel) == expected, 'source changed: ' + rel)
    import cupy as cp
    sys.path[:0] = [str(ROOT / 'motor_nuevo/epoch_cost_20260923'),
                    str(ROOT / 'motor_nuevo/pipeline_review_20260922')]
    import run_set
    from runtime_session import RuntimeSession
    old_init = RuntimeSession.__init__
    context = {'kernel_calls': 0}

    def installed(session, *args, **kwargs):
        old_init(session, *args, **kwargs)
        import device_cell
        cls = device_cell.DeviceCell
        original_advance = cls.advance
        context['cls'] = cls
        context['original_advance'] = original_advance

        def captured_advance(owner, *a, **kw):
            if not hasattr(owner, '_axioma_capture_kernel'):
                kernel = owner.kernel
                context['kernel'] = kernel
                context['owner'] = owner

                def captured(grid, block, arguments):
                    context['grid'] = grid
                    context['block'] = block
                    context['arguments'] = arguments
                    context['kernel_calls'] += 1
                    return kernel(grid, block, arguments)

                owner.kernel = captured
                owner._axioma_capture_kernel = True
            return original_advance(owner, *a, **kw)

        cls.advance = captured_advance

    RuntimeSession.__init__ = installed
    old_argv = sys.argv
    sys.argv = ['run_set.py', '--out', str(out), '--odor', 'sham', '--engine', 'causal_cuda',
                '--ms', '1', '--observe', 'off', '--cuda-profile', 'off', '--profile', 'off',
                '--kc-capture', 'off']
    try:
        code = run_set.main()
    finally:
        sys.argv = old_argv
        RuntimeSession.__init__ = old_init
        if 'cls' in context:
            context['cls'].advance = context['original_advance']
    result = json.loads((out / 'RESULT.json').read_text())
    need(code == 0 and result['status'] == 'COMPLETE' and context['kernel_calls'] == 16,
         'original organism did not complete sixteen physical epochs')
    owner = context['owner']
    raw = context['arguments']
    need(len(raw) == 51 and context['grid'] == (owner.b.n,) and context['block'] == (32,),
         'cell kernel signature changed')
    cp.cuda.Device().synchronize()
    report_replays = []
    for repeat in range(plan['budget']['replay_launches_max']):
        clones = {}
        args = []
        with owner.stream:
            for x in raw:
                if isinstance(x, cp.ndarray):
                    ident = id(x)
                    if ident not in clones:
                        clones[ident] = x.copy()
                    args.append(clones[ident])
                else:
                    args.append(x)
            for index in (33, 34, 35, 36, 37, 48, 49, 50):
                args[index].fill(0)
        owner.stream.synchronize()
        before = cp.cuda.Event()
        after = cp.cuda.Event()
        host_start = time.perf_counter()
        with owner.stream:
            before.record(owner.stream)
            context['kernel'](context['grid'], context['block'], tuple(args))
            after.record(owner.stream)
        after.synchronize()
        host_ms = (time.perf_counter() - host_start) * 1000
        event_ms = float(cp.cuda.get_elapsed_time(before, after))
        status = cp.asnumpy(args[50])
        count = cp.asnumpy(args[48])
        report_replays.append({'index': repeat, 'event_ms': event_ms, 'enclosing_host_ms': host_ms,
                               'status_nonzero': int(np.count_nonzero(status)),
                               'accepted': int(count[:, 0].sum()),
                               'rejected': int(count[:, 1].sum()),
                               'timing_consistent': event_ms <= host_ms + 1.0})
    report = {'schema': 'cell_isolated_replay_result_v1', 'plan_sha256': sha(PLAN),
              'probe_sha256': sha(Path(__file__)), 'organism_result_sha256': sha(out / 'RESULT.json'),
              'original_kernel_calls': context['kernel_calls'],
              'original_owner_wall_s': result['runtime']['cell']['native_wall_s'],
              'original_step_wall_s': sum(json.loads(line)['step_wall_s'] for line in
                                          (out / 'PROGRESS.jsonl').read_text().splitlines()),
              'kernel_attributes': context['kernel'].attributes,
              'replay_post_state_only': True, 'replays': report_replays,
              'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'disk_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    report['budget_ok'] = report['wall_s'] <= plan['budget']['wall_s_max'] and \
        report['maxrss_kib'] <= plan['budget']['ram_gib_max'] * 1024 ** 2 and \
        report['disk_bytes'] <= plan['budget']['disk_bytes_max']
    report['status'] = 'COMPLETE_DIAGNOSTIC_ONLY' if report['budget_ok'] and all(
        x['timing_consistent'] and x['status_nonzero'] == 0 for x in report_replays
    ) else 'INCOMPLETE_RETAINED'
    (out / 'CELL_ISOLATED_REPLAY_RESULT.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'original_owner_wall_s', 'replays', 'wall_s')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
