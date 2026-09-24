"""Verify the new saved raw pstats against paired complete scientific states.

The instrumented runner failed after writing raw pstats because Path was
passed to pstats.Stats. This verifier does not resume or rerun the organism.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROUND = Path(__file__).resolve().parent
ROOT = ROUND.parents[1]
CONTROL = ROUND / 'NONCNS_pstats_off_01'
PROFILE = ROUND / 'NONCNS_pstats_on_01'
PLAN = ROUND / 'NONCNS_PSTATS_REPAIR_PLAN_02.json'
COMPARE_SOURCE = ROOT / 'motor_nuevo/epoch_cost_20260923/compare_profile.py'
EXTRACT_SOURCE = ROUND / 'NONCNS_pstats_arcs.py'
COMPARE_SHA256 = '240dd392a4d27a6a9b472a265837a5c5c371c397115bff424df907478ed5c745'
EXTRACT_SHA256 = 'bc6aa3ecb8d42e08a557a7aa0d752e21fb20cb85c7875a0059b49f3c43463b1b'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load frozen source {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arc_gate(rows):
    targets = {r['kind']: r for r in rows
               if r['kind'] in ('cupy.ndarray.get', 'numpy.ndarray.tolist')}
    if set(targets) != {'cupy.ndarray.get', 'numpy.ndarray.tolist'}:
        return False
    return all(r['calls'] > 0 and r['arc_call_sum'] == r['calls']
               and sum(a['calls'] for a in r['arcs']) == r['calls']
               for r in targets.values())


def main():
    plan = json.loads(PLAN.read_text())
    if plan['schema'] != 'noncns_pstats_offline_repair_plan_v1':
        raise RuntimeError('Wrong offline repair plan')
    if digest(CONTROL / 'RESULT.json') != plan['prior_control_result_sha256'] or \
            digest(PROFILE / 'RESULT.json') != plan['prior_profile_result_sha256']:
        raise RuntimeError('Original run results changed')
    if digest(COMPARE_SOURCE) != COMPARE_SHA256 or digest(EXTRACT_SOURCE) != EXTRACT_SHA256:
        raise RuntimeError('Frozen comparator or extractor changed')
    compare = load_module('frozen_profile_compare', COMPARE_SOURCE).compare
    extractor = load_module('frozen_pstats_extract', EXTRACT_SOURCE)

    status = {arm: json.loads((base / 'RESULT.json').read_text())
              for arm, base in [('control', CONTROL), ('profile', PROFILE)]}
    if status['control']['status'] != 'COMPLETE' or status['control']['completed_trial_ms'] != 2:
        raise RuntimeError('Control did not finish two ms')
    err = status['profile']['error']
    if (status['profile']['status'] != 'INCOMPLETE' or
            status['profile']['completed_trial_ms'] != 2 or not isinstance(err, dict) or
            err.get('type') != 'TypeError' or
            not err.get('message','').startswith('Cannot create or construct a <class \'pstats.Stats\'> object from PosixPath(')):
        raise RuntimeError('Unexpected profile failure or incomplete organism step')
    raw = PROFILE / 'NONCNS_RAW.pstats'
    if digest(raw) != plan['raw_pstats_sha256'] or raw.stat().st_size > 16 * 1024 * 1024:
        raise RuntimeError('Frozen raw profile cap exceeded')
    extracted = extractor.extract(str(raw))
    arcs = extracted['rows']
    encoded = (json.dumps(extracted, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if len(encoded) > plan['limits']['derived_json_MiB_max']*1024*1024:
        raise RuntimeError('Derived caller-arc file exceeds budget')
    gate = arc_gate(arcs)
    if not gate:
        raise RuntimeError('Raw get/tolist caller arcs incomplete')
    flat = json.loads((PROFILE / 'PROFILE.json').read_text())['rows']
    flat_by_name = {r['function']: r for r in flat}
    for row in arcs:
        if row['kind'] not in ('cupy.ndarray.get', 'numpy.ndarray.tolist'):
            continue
        if flat_by_name[row['callee']['function']]['calls'] != row['calls']:
            raise RuntimeError('Raw pstats and flat profile count disagree')

    state = {}
    docs = {}
    packs = {}
    for arm, base in [('control', CONTROL), ('profile', PROFILE)]:
        final = base / 'final_state'
        docs[arm] = json.loads((final / 'session.json').read_text())
        packs[arm] = np.load(final / 'session.npz', allow_pickle=False)
    try:
        if set(docs['control']) != set(docs['profile']):
            raise RuntimeError('Scientific state top-level keys differ')
        for field in docs['control']:
            differences = []
            compare(docs['control'][field], docs['profile'][field],
                    packs['control'], packs['profile'], field, differences)
            state[field] = {'equal': not differences, 'different_paths': differences[:8],
                            'different_path_count': len(differences)}
        tick = docs['control']['ticks']
        docs['control']['ticks'] = tick + 1
        mutant = []
        compare(docs['control']['ticks'], docs['profile']['ticks'],
                packs['control'], packs['profile'], 'ticks', mutant)
        docs['control']['ticks'] = tick
        if not mutant:
            raise RuntimeError('Deliberately corrupted tick escaped comparator')
    finally:
        for pack in packs.values():
            pack.close()

    exact_files = {}
    for rel in ('final_state/prosthesis.json', 'final_state/prosthesis.npz',
                'final_state/published.json', 'final_state/published.npz',
                'final_state/boundary.json', 'traces.npz', 'EVENT_AUDIT.json'):
        exact_files[rel] = digest(CONTROL / rel) == digest(PROFILE / rel)
    mutant_arcs = [dict(r) for r in arcs]
    for r in mutant_arcs:
        if r['kind'] == 'cupy.ndarray.get':
            r['calls'] += 1
            break
    if arc_gate(mutant_arcs):
        raise RuntimeError('Deliberately corrupted arc total escaped gate')
    exact = all(v['equal'] for v in state.values()) and all(exact_files.values())
    if not exact:
        raise RuntimeError('Profile changed scientific state')

    targets = {r['kind']: r for r in arcs
               if r['kind'] in ('cupy.ndarray.get', 'numpy.ndarray.tolist')}
    result = {
        'schema': 'noncns_pstats_path_repair_verification_v1',
        'classification': 'SALVAGED_DIAGNOSTIC_FROM_PATH_TYPE_ERROR',
        'runner_failure': err['message'],
        'organism_ms_per_arm': 2,
        'scientific_state_exact': exact,
        'state_fields': state,
        'auxiliary_files_exact': exact_files,
        'raw_profile_sha256': digest(raw),
        'raw_profile_bytes': raw.stat().st_size,
        'complete_caller_arcs': gate,
        'get_calls': targets['cupy.ndarray.get']['calls'],
        'tolist_calls': targets['numpy.ndarray.tolist']['calls'],
        'deliberate_state_mutation_detected': True,
        'deliberate_arc_mutation_detected': True,
        'compare_source_sha256': COMPARE_SHA256,
        'extractor_source_sha256': EXTRACT_SHA256,
        'repair_plan_sha256': digest(PLAN),
        'limitation': ('Only second millisecond of a two-ms sham run; cProfile times '
                       'include device waits and profiler overhead. Runner status remains '
                       'INCOMPLETE; raw pstats and state were saved before the import failure.'),
    }
    arcs_out = ROUND / 'NONCNS_ARCS_REPAIRED_02.json'
    with arcs_out.open('xb') as f:
        f.write(encoded)
    result['arcs_sha256'] = digest(arcs_out)
    out = ROUND / 'NONCNS_PSTATS_SALVAGED_VERIFY_02.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('classification', 'scientific_state_exact',
                                            'complete_caller_arcs', 'get_calls', 'tolist_calls')}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        print(f'NONCNS saved verification failed: {exc}', file=sys.stderr)
        raise
