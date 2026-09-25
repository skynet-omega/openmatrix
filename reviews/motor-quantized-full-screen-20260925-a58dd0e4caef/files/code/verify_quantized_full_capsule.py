"""Recompute bounded release-cache and selected full-coefficient evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(capsule, report_path, plan_path, out):
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    manifest = json.loads(capsule.with_name('QUANTIZED_CAPSULE_MANIFEST.json').read_text())
    need(plan['schema'] == 'quantized_full_capsule_plan_v1', 'Plan schema')
    need(sha(capsule) == manifest['capsule_sha256'], 'Capsule digest')
    need(sha(report_path) == plan['result_sha256'], 'Result digest')
    need(report['full_coefficient_arrays_sha256'] == manifest['full_arrays_sha256']
         == plan['full_arrays_sha256'], 'Full archive provenance')
    with np.load(capsule, allow_pickle=False) as z:
        a = {key: z[key] for key in z.files}
    n, ts = int(plan['n_sources']), int(plan['transmission_start'])
    selection = list(plan['selected_query_indices'])
    need(a['selected_query_indices'].tolist() == selection, 'Selection')
    need(a['release'].shape == (60, n) and a['release'].dtype == np.float64,
         'Release shape')
    need(a['outdegree'].shape == (n,) and int(a['outdegree'].sum()) == 25582938,
         'Edge census')
    event = a['event_rows']
    need(event.shape == (4062,) and np.all((event >= 0) & (event < n)),
         'Event source cohort')
    active_source = np.ones(n, dtype=bool)
    active_source[event] = False
    cache = None
    selected_cache = {}
    all_edges = []
    for j in range(60):
        release = a['release'][j]
        if cache is None:
            cache = release.copy()
            active = np.zeros(n, dtype=bool)
        else:
            active = active_source & (np.abs(release - cache) > plan['absolute_release_threshold'])
            cache[active] = release[active]
            cache[event] = release[event]
        if j in selection:
            selected_cache[j] = cache.copy()
        edges = int(a['outdegree'][active].sum(dtype=np.int64))
        row = report['per_query'][j]
        need(row['query'] == j and row['updated_sources'] == int(np.count_nonzero(active))
             and row['updated_edges'] == edges, 'Release update accounting')
        need(float.fromhex(row['time_hex']) == a['query_s'][j]
             and row['trial_step_s'] == a['trial_step_s'][j], 'Query clocks')
        if j:
            all_edges.append(edges)
    need(float(np.median(all_edges)) == report['median_updated_edges'], 'Median edges')
    selected_proxy = {}
    for slot, j in enumerate(selection):
        arrays = [a['selected_' + name][slot] for name in
                  ('consumed_z', 'consumed_target', 'consumed_rate',
                   'baseline_z', 'baseline_target', 'baseline_rate',
                   'candidate_z', 'candidate_target', 'candidate_rate')]
        need(all(x.shape == (359373,) and x.dtype == np.float64 and np.isfinite(x).all()
                 for x in arrays), 'Selected array layout')
        z, target, rate, bz, ba, br, qz, qa, qr = arrays
        need(np.array_equal(z, bz) and np.array_equal(target, ba)
             and np.array_equal(rate, br), 'Selected baseline equality')
        need(np.array_equal(z[ts:ts+n], a['release'][j]), 'Selected source state')
        expected_z = z.copy()
        expected_z[ts:ts+n] = selected_cache[j]
        need(np.array_equal(expected_z, qz), 'Selected quantized state')
        need(np.array_equal(qa[ts:ts+n], target[ts:ts+n])
             and np.array_equal(qr[ts:ts+n], rate[ts:ts+n]),
             'Source dynamics equality')
        h = float(a['trial_step_s'][j])
        f_base = rate * (target - z)
        f_candidate = qr * (qa - z)
        denominator = 3 * (plan['atol'] + plan['rtol'] *
                           np.maximum(np.abs(z), np.abs(z + h * f_base)))
        need(np.isfinite(denominator).all() and np.all(denominator > 0), 'Scale')
        proxy = float(np.max(h * np.abs(f_candidate - f_base) / denominator))
        row = report['per_query'][j]
        need(abs(proxy - row['local_rhs_proxy_max']) <= plan['verification_max_abs'],
             'Selected RHS proxy')
        selected_proxy[str(j)] = proxy
    need(selected_proxy['24'] == report['maximum_local_rhs_proxy'], 'Selected global maximum')
    result = {'status': 'PASS_CAPSULE_RECOMPUTE_LIMITED',
              'capsule_sha256': sha(capsule), 'queries_release': 60,
              'queries_full_coefficients': selection,
              'median_updated_edges': float(np.median(all_edges)),
              'selected_proxy': selected_proxy,
              'scope': 'Recomputes update counts for all 60 and full coefficient proxy for three queries. Does not independently recheck other 57 coefficient outputs or integrated state.'}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.capsule, a.report, a.plan, a.out), indent=2))
