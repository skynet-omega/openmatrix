"""Make a bounded review capsule from the full same-run coefficient archive."""
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


def run(source, out, plan):
    need(not out.exists(), 'Unique capsule output required')
    p = json.loads(plan.read_text())
    need(p['schema'] == 'quantized_full_capsule_plan_v1', 'Plan schema')
    need(sha(Path(__file__)) == p['script_sha256'], 'Maker changed')
    report_path = source / 'QUANTIZED_FULL_RESULT.json'
    archive_path = source / 'FULL_COEFFICIENT_ARRAYS.npz'
    need(sha(report_path) == p['result_sha256'], 'Result changed')
    need(sha(archive_path) == p['full_arrays_sha256'], 'Full arrays changed')
    report = json.loads(report_path.read_text())
    selected = np.asarray(p['selected_query_indices'], dtype=np.int64)
    need(selected.tolist() == [0, 24, 59], 'Query selection changed')
    need(report['per_query'][24]['local_rhs_proxy_max'] == report['maximum_local_rhs_proxy'],
         'Selected maximum changed')
    with np.load(archive_path, allow_pickle=False) as z:
        ts = int(p['transmission_start'])
        n = int(p['n_sources'])
        fields = {'release': z['consumed_z'][:, ts:ts+n],
                  'outdegree': z['outdegree'], 'event_rows': z['event_rows'],
                  'query_s': z['query_s'], 'trial_step_s': z['trial_step_s'],
                  'selected_query_indices': selected}
        for name in ('consumed_z', 'consumed_target', 'consumed_rate',
                     'baseline_z', 'baseline_target', 'baseline_rate',
                     'candidate_z', 'candidate_target', 'candidate_rate'):
            fields['selected_' + name] = z[name][selected]
        need(fields['release'].shape == (60, 166700), 'Release cohort')
        need(all(np.isfinite(v).all() for v in fields.values() if v.dtype.kind == 'f'),
             'Nonfinite capsule data')
        np.savez_compressed(out, **fields)
    size = out.stat().st_size
    need(size <= p['budget']['capsule_bytes_max'], 'Capsule over budget')
    result = {'status': 'CAPSULE_COMPLETE', 'capsule_sha256': sha(out),
              'capsule_bytes': size, 'full_arrays_sha256': sha(archive_path),
              'selected_query_indices': selected.tolist(),
              'scope': 'All 60 release vectors and edge counts; full coefficients only at queries 0, 24, 59. Full 60-query verification remains local.'}
    out.with_name('QUANTIZED_CAPSULE_MANIFEST.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.source, a.out, a.plan), indent=2))
