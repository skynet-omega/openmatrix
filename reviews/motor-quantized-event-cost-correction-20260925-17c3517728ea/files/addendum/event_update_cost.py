"""Count exact-event and thresholded-continuous CSC updates on the same block."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def run(capsule, report_path, plan_path, out):
    start = time.monotonic()
    p = json.loads(plan_path.read_text())
    need(p['schema'] == 'event_update_cost_plan_v1', 'Plan schema')
    need(sha(Path(__file__)) == p['script_sha256'], 'Script changed')
    need(sha(capsule) == p['capsule_sha256'] and
         sha(report_path) == p['report_sha256'], 'Input changed')
    r = json.loads(report_path.read_text())
    with np.load(capsule, allow_pickle=False) as z:
        release = z['release']
        degree = z['outdegree']
        event = z['event_rows']
    n = len(degree)
    need(release.shape == (60, n) and n == 166700 and len(event) == 4062
         and len(np.unique(event)) == len(event), 'Layout')
    is_event = np.zeros(n, dtype=bool)
    is_event[event] = True
    cache = release[0].copy()
    rows = []
    for j in range(1, len(release)):
        d = release[j] - cache
        active_event = is_event & (d != 0)
        active_continuous = ~is_event & (np.abs(d) > p['absolute_release_threshold'])
        e_event = int(degree[active_event].sum(dtype=np.int64))
        e_cont = int(degree[active_continuous].sum(dtype=np.int64))
        need(e_cont == r['per_query'][j]['updated_edges'] and
             int(np.count_nonzero(active_continuous)) ==
             r['per_query'][j]['updated_sources'], 'Published continuous count differs')
        cache[active_event | active_continuous] = release[j, active_event | active_continuous]
        need(np.array_equal(cache[event], release[j, event]), 'Event source not exact')
        rows.append({'query': j, 'event_sources': int(np.count_nonzero(active_event)),
                     'event_edges': e_event, 'continuous_sources':
                     int(np.count_nonzero(active_continuous)),
                     'continuous_edges': e_cont, 'total_edges': e_event + e_cont})
    totals = [x['total_edges'] for x in rows]
    result = {'schema': 'event_update_cost_result_v1',
              'status': 'PASS_FULL_EVENT_COST_RECOUNT',
              'plan_sha256': sha(plan_path), 'capsule_sha256': sha(capsule),
              'queries_after_seed': len(rows),
              'event_edges_median': float(np.median([x['event_edges'] for x in rows])),
              'event_edges_max': max(x['event_edges'] for x in rows),
              'continuous_edges_median': float(np.median([x['continuous_edges'] for x in rows])),
              'total_edges_median': float(np.median(totals)),
              'total_edges_max': max(totals),
              'total_edges_fraction_of_csr_median': float(np.median(totals) / degree.sum()),
              'per_query': rows,
              'scope': 'Counts CSC edge updates for exact event sources plus thresholded continuous sources. Still omits source selection, atomics, owner transforms, state-error and speed costs.',
              'wall_s': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    b = p['budget']
    need(result['wall_s'] <= b['wall_seconds_max'] and
         result['maxrss_kib'] <= b['ram_gib_max'] * 1024**2,
         'Budget exceeded')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    need(out.stat().st_size <= b['output_bytes_max'], 'Output budget')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    r = run(a.capsule, a.report, a.plan, a.out)
    print(json.dumps({k: r[k] for k in ('status', 'event_edges_median',
                                       'continuous_edges_median', 'total_edges_median',
                                       'total_edges_fraction_of_csr_median', 'wall_s')}))
