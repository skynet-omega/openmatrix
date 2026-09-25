"""Check that profiling preserved the organism and report clock attribution limits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE/'event_step_baseline_v2_01'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_tree(a, b, a_npz, b_npz, path=''):
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) == {'__array__'}:
        x, y = a_npz[a['__array__']], b_npz[b['__array__']]
        need(x.shape == y.shape and x.dtype == y.dtype, 'array layout: '+path)
        need(np.array_equal(x, y, equal_nan=x.dtype.kind in 'fc'), 'array mismatch: '+path)
    elif isinstance(a, dict) and isinstance(b, dict):
        need(set(a) == set(b), 'dictionary shape: '+path)
        for key in a:
            compare_tree(a[key], b[key], a_npz, b_npz, path+'/'+str(key))
    elif isinstance(a, list) and isinstance(b, list):
        need(len(a) == len(b), 'list length: '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, a_npz, b_npz, path+'/'+str(i))
    else:
        need(a == b, 'scalar mismatch: '+path)


def verify_one(name):
    folder = HERE/name
    report = json.loads((folder/'CELL_KERNEL_TIMING_RESULT.json').read_text())
    need(report['status'] == 'COMPLETE_DIAGNOSTIC_ONLY', 'run incomplete')
    need(report['calls'] == 16 and report['wrapped_owners'] == 1, 'coverage')
    need(report['run_result_sha256'] == sha(folder/'RESULT.json'), 'result hash')
    need(report['plan_sha256'] == sha(HERE/('CELL_KERNEL_PLAN_47.json' if name.endswith('02') else 'CELL_KERNEL_PLAN_46.json')),
         'plan hash')
    need(json.loads((BASE/'EVENT_AUDIT.json').read_text()) ==
         json.loads((folder/'EVENT_AUDIT.json').read_text()), 'event log altered')
    for rel in ('traces.npz', 'final_state/published.json',
                'final_state/published.npz', 'final_state/prosthesis.json',
                'final_state/prosthesis.npz', 'preparation_inputs/intervenciones_W.npz'):
        need(sha(BASE/rel) == sha(folder/rel), 'output altered: '+rel)
    base_state = BASE/'final_state'
    new_state = folder/'final_state'
    with np.load(base_state/'session.npz') as x, np.load(new_state/'session.npz') as y:
        compare_tree(json.loads((base_state/'session.json').read_text()),
                     json.loads((new_state/'session.json').read_text()), x, y)
    kernel = np.asarray(report['per_call_kernel_ms'], dtype=float)
    need(kernel.shape == (16,) and np.isfinite(kernel).all() and (kernel>0).all(),
         'kernel timings invalid')
    need(math.isclose(float(kernel.sum())/1000, report['sum_kernel_s'], rel_tol=1e-10),
         'kernel sum')
    outcome = {'run': name, 'scientific_outputs_exact': True,
               'sum_kernel_cuda_event_s': report['sum_kernel_s'],
               'native_owner_wall_s': report['native_owner_wall_s'],
               'kernel_exceeds_owner_wall_s': report['sum_kernel_s']-report['native_owner_wall_s']}
    if 'per_call_host_advance_ms' in report:
        host = np.asarray(report['per_call_host_advance_ms'], dtype=float)
        owner = np.asarray(report['per_call_owner_delta_ms'], dtype=float)
        need(host.shape == owner.shape == kernel.shape and np.isfinite(host).all() and
             np.isfinite(owner).all(), 'host timing array')
        need(math.isclose(float(host.sum())/1000, report['sum_host_advance_s'], rel_tol=1e-10),
             'host sum')
        need(math.isclose(float(owner.sum())/1000, report['native_owner_wall_s'], rel_tol=1e-10),
             'owner sum')
        outcome['sum_host_advance_s'] = report['sum_host_advance_s']
        outcome['kernel_exceeds_owner_call_count'] = int((kernel>owner).sum())
    return outcome


def verify():
    results = [verify_one(n) for n in ('cell_kernel_timing_01','cell_kernel_timing_02')]
    need(all(x['kernel_exceeds_owner_wall_s'] > 0 for x in results), 'clock contradiction vanished')
    return {'schema':'cell_kernel_timing_verification_v1',
            'status':'SCIENTIFIC_OUTPUT_EXACT__KERNEL_FRACTION_UNVERIFIED',
            'runs':results,
            'interpretation':'CUDA event interval exceeds containing native-owner wall in both runs; use a profiler timeline before attributing exact kernel fraction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    need(not a.out.exists(), 'output exists')
    result = verify()
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'], 'runs':result['runs']}))
