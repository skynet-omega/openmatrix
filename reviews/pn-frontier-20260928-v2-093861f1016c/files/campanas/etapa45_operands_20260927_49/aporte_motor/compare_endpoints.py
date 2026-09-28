"""Exact split/continuous comparison of all serialized scientific owners."""
import argparse
import json
import time
import resource
import sys
from pathlib import Path

from resume49 import compare_trees, save, verify_checkpoint, reference
sys.path.insert(0, str(reference.OLD/'src'))
from session_io import read_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('continuous', type=Path)
    p.add_argument('resumed', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    left, right = verify_checkpoint(a.continuous), verify_checkpoint(a.resumed)
    checks = {}
    owners = ('session', 'prosthesis', 'published', 'stored_operator', 'effective_operator',
              'input_owner', 'interval_owner', 'motor_owner')
    for name in owners:
        checks[name] = compare_trees(read_state(a.continuous/name), read_state(a.resumed/name))
        print(name, checks[name]['exact'], checks[name]['different_paths'], flush=True)
    checks['boundary'] = compare_trees(json.loads((a.continuous/'boundary.json').read_text()),
                                      json.loads((a.resumed/'boundary.json').read_text()))
    result = dict(exact=all(v['exact'] for v in checks.values()), trees=checks,
                  continuous_manifest=left['time_ns'], resumed_manifest=right['time_ns'],
                  method='Direct recursive dtype/shape/value comparison; no tolerance or ignored paths.',
                  new_neural_ms=0, wall_s=time.monotonic()-started,
                  CPU_s=time.process_time()-cpu,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save(a.out, result)
    print(json.dumps({k:result[k] for k in ('exact', 'wall_s', 'CPU_s', 'peak_RSS_bytes')}))
    if not result['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
