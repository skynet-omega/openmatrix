"""One bounded process for an uninterrupted or resumed OFF-tail segment."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import resource
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from resume49 import build, need, save, sha, HERE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ms', type=int, choices=(2, 4), required=True)
    p.add_argument('--split', type=int, choices=(0, 2), default=0)
    args = p.parse_args()
    need(not args.out.exists(), 'Preserve previous evidence')
    started, cpu = time.monotonic(), time.process_time()
    def stop(*_):
        raise TimeoutError('Finite qualification budget')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(900)
    resource.setrlimit(resource.RLIMIT_CPU, (700, 710))
    # Runtime guard avoids killing healthy WSL siblings or altering GPU policy.
    report = dict(status='STARTED', attempted_ms=0, committed_ms=0, source=str(args.source),
                  runner_sha256=sha(__file__), schema_harness_sha256=sha(HERE/'resume49.py'),
                  restorer_sha256=sha(HERE/'restore48.py'), equations_changed=False)
    run = None
    try:
        run = build(args.source, args.out)
        import cupy as cp
        from source_inventory import imported, verify
        from resume49 import C48
        lock = json.loads((C48/'SOURCES.json').read_text())
        current = imported()
        changed = [name for name, value in current.items() if name in lock and lock[name] != value]
        unknown = [name for name in current if name not in lock and not Path(name).is_relative_to(HERE)]
        need(not changed and not unknown, 'Unexpected execution sources: ' + repr(changed + unknown))
        save(args.out/'EXECUTED_SOURCES.json', current)
        report.update(initial=run.initial, appendix=run.appendix, load_s=time.monotonic()-started)
        print(json.dumps(dict(status='RESTORED_EXACT', elapsed_s=report['load_s'])), flush=True)
        rows = []
        for j in range(1, args.ms+1):
            report['attempted_ms'] += 1
            save(args.out/'STATUS.json', report)
            row = run.step()
            rows.append(row)
            report['committed_ms'] += 1
            free, total = cp.cuda.runtime.memGetInfo()
            need(total-free < 14*1024**3, 'VRAM budget')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 24*1024**3, 'RAM budget')
            print(json.dumps(dict(status='STEP', tail_ms=j, consumed_ms=run.stimulus.k,
                                  elapsed_s=time.monotonic()-started)), flush=True)
            if j == args.split:
                run.save(args.out/'split_state')
        np.savez_compressed(args.out/'traces.npz', **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
        run.observer.flush(args.out/'dng100_observed.npz')
        run.save(args.out/'final_state')
        save(args.out/'EVENTS.json', run.session.events.audit)
        report['runtime'] = run.session.report()
        report['observer'] = run.observer.report()
        verify(current)
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        if run is not None:
            try:
                run.close()
            except BaseException as exc:
                report.update(status='FAILED', cleanup_error=repr(exc))
        report.update(wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        args.out.mkdir(parents=True, exist_ok=True)
        save(args.out/'RESULT.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'attempted_ms', 'committed_ms', 'wall_s', 'CPU_s', 'peak_RSS_bytes')}), flush=True)
        signal.alarm(0)
    if report['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
