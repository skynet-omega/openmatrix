"""Offline caller-arc extraction from a preserved cProfile/pstats file.

This script imports neither CuPy nor the organism. The prior flat PROFILE.json
cannot be used as input because its caller dictionaries were discarded.
"""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats


def _func(f):
    return {'file': f[0], 'line': f[1], 'function': f[2]}


def _kind(name):
    if "method 'get' of 'cupy._core.core._ndarray_base'" in name:
        return 'cupy.ndarray.get'
    if "method 'tolist' of 'numpy.ndarray'" in name:
        return 'numpy.ndarray.tolist'
    if name == 'asnumpy':
        return 'asnumpy'
    if name == 'host':
        return 'host'
    return None


def extract(profile):
    stats = pstats.Stats(profile).stats
    rows = []
    for callee, (primitive, calls, self_s, cumulative_s, callers) in stats.items():
        kind = _kind(callee[2])
        if kind is None:
            continue
        arcs = []
        for caller, value in callers.items():
            if isinstance(value, tuple) and len(value) == 4:
                arc_calls, arc_primitive, arc_self_s, arc_cumulative_s = value
            else:
                # Old profiler formats contain only call counts; avoid
                # inventing per-edge time if such a file is supplied.
                arc_calls, arc_primitive = int(value), None
                arc_self_s = arc_cumulative_s = None
            arcs.append({'caller': _func(caller), 'calls': arc_calls,
                         'primitive_calls': arc_primitive,
                         'self_s': arc_self_s, 'cumulative_s': arc_cumulative_s})
        arcs.sort(key=lambda a: (-a['calls'], a['caller']['file'], a['caller']['line']))
        rows.append({'kind': kind, 'callee': _func(callee), 'calls': calls,
                     'primitive_calls': primitive, 'self_s': self_s,
                     'cumulative_s': cumulative_s, 'arcs': arcs,
                     'arc_call_sum': sum(a['calls'] for a in arcs)})
    rows.sort(key=lambda r: (-r['self_s'], r['kind'], r['callee']['file']))
    return {'schema': 'noncns_pstats_caller_arcs_v1', 'rows': rows,
            'caveat': ('Arcs identify calling functions, not individual call '
                       'expressions or array sizes. Self/cumulative times may '
                       'include GPU synchronization and profiler overhead.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_pstats', type=Path)
    parser.add_argument('out_json', type=Path)
    args = parser.parse_args()
    result = extract(args.raw_pstats)
    data = json.dumps(result, sort_keys=True, allow_nan=False, indent=2)
    if len(data.encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('Frozen 4-MiB derived output cap exceeded')
    args.out_json.write_text(data + '\n', encoding='utf-8')


def _cpu_fixture():
    # Offline CPU-only check that pstats retains real NumPy caller arcs.
    import numpy as np

    def caller():
        return np.arange(4).tolist()

    profiler = cProfile.Profile()
    profiler.enable()
    caller()
    profiler.disable()
    rows = extract(profiler)['rows']
    target = next(r for r in rows if r['kind'] == 'numpy.ndarray.tolist')
    if target['calls'] != 1 or target['arc_call_sum'] != 1 or not any(
            a['caller']['function'] == 'caller' for a in target['arcs']):
        raise RuntimeError('CPU pstats caller-arc fixture failed')
    print('CPU-only pstats caller-arc fixture PASS')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        _cpu_fixture()
    else:
        main()
