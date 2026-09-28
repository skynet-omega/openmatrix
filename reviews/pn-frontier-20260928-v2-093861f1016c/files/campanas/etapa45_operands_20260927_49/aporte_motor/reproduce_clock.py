"""Reproduce the checkpoint clock rejection without loading CNS or MuJoCo."""
import argparse
import hashlib
import json
import mmap
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/daroch/AXIOMA_ASTRA')
C48 = ROOT/'campanas/etapa45_composicion_20260927_48'
PREPARED = ROOT/'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state'


def clocks(folder):
    with (folder/'session.json').open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        start = re.search(rb'(?m)^  "body":\s*', m).end()
        b, _ = json.JSONDecoder().raw_decode(m[start:start+1000000].decode())
    with np.load(folder/'session.npz', allow_pickle=False) as a:
        t = float(a[b['integration']['__array__']][0])
        parent = float(a[b['parent']['integration']['__array__']][0])
    dt = b['dt_ns']*1e-9
    steps = b['parent']['steps']
    grid = steps*dt
    return dict(path=str(folder), body_time_s=t, parent_time_s=parent, dt_s=dt,
                steps=steps, time_ns=b['rh']['time_ns'], grid_time_s=grid,
                absolute_error_s=abs(t-grid), historical_limit_s=1e-10,
                rejected_by_historical_clock_guard=abs(t-grid)>1e-10,
                checkpoint_manifest_sha256=hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Preserve previous evidence')
    cpu, start = time.process_time(), time.monotonic()
    prepared = clocks(PREPARED)
    records = {arm: clocks(C48/arm/'final_state') for arm in ('sham', 'dm1', 'profile', 'permuted')}
    for row in records.values():
        t = prepared['body_time_s']
        count = row['steps']-prepared['steps']
        for _ in range(count):
            t += row['dt_s']
        row.update(added_steps_from_prepared=count, repeated_addition_from_prepared_s=t,
                   repeated_addition_matches_saved_exactly=t==row['body_time_s'])
    result = dict(schema='checkpoint48_clock_guard_reproduction_v1', prepared=prepared,
                  final_states=records, new_neural_ms=0, new_body_steps=0,
                  CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-start,
                  scope='Scalar clock arithmetic only, no organism or MuJoCo advance')
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
