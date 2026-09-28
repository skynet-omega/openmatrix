"""Domain and temporal-coverage checks before a three-arm signal intervention.

This module never reconstructs an unrecorded RHS history, runs a CNS, or claims
causal mediation. Per-bin first/last values are distinct descriptive samples.
"""
from pathlib import Path
import json
import time
import numpy as np
from session_io import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decompose(none, left, right):
    arrays = [np.asarray(x, dtype=np.float64) for x in (none, left, right)]
    require(arrays[0].ndim == 2 and arrays[0].size > 0, 'time by identity matrix required')
    require(all(x.shape == arrays[0].shape and np.isfinite(x).all() for x in arrays),
            'finite aligned matrices required')
    n, l, r = arrays
    mean = (l + r) / 2
    return {'common': mean - n, 'lateral': (l - r) / 2,
            'null_lateral': mean, 'null_common_L': n + (l - r) / 2,
            'null_common_R': n - (l - r) / 2}


def domain_summary(values, lower, upper):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.size and np.isfinite(values).all(), 'invalid signal')
    bad = (values < lower) | (values > upper)
    return {'min': float(values.min()), 'max': float(values.max()),
            'out_of_domain_entries': int(bad.sum()),
            'affected_identities': int(np.any(bad, axis=0).sum()),
            'affected_bins': int(np.any(bad, axis=1).sum()),
            'domain_valid': not bool(np.any(bad))}


def validate_summary(summary, shape, lower, upper):
    require(set(summary) == {'first', 'last', 'lo', 'hi', 'counts'}, 'summary fields changed')
    require(all(v.shape == shape for v in summary.values()), 'summary shape changed')
    counts = summary['counts']
    require(counts.dtype.kind in 'iu' and np.all(counts > 0), 'missing integer RHS counts')
    require(np.array_equal(counts, np.broadcast_to(counts[:, :1], shape)),
            'identity coverage differs within bin')
    for key in ('first', 'last', 'lo', 'hi'):
        require(np.isfinite(summary[key]).all(), 'nonfinite summary ' + key)
    lo, hi = summary['lo'], summary['hi']
    require(np.all(lo <= hi) and np.all(lo >= lower) and np.all(hi <= upper),
            'recorded signal outside declared domain')
    for key in ('first', 'last'):
        require(np.all((lo <= summary[key]) & (summary[key] <= hi)),
                'sample outside recorded extrema')


def evaluate(inputs, parameters, output):
    started = time.process_time()
    require(not parameters, 'all decisions belong in the frozen contract')
    c = json.loads(Path(inputs['contract']).read_text())
    require(c['schema'] == 'matrix_signal_contrast_preflight_v1', 'unknown contract')
    require(c['quantity'] == 'PN_generic_CSR_release' and c['unit'] == 'dimensionless',
            'different observable needs an explicit contract')
    require(c['phase'] == 'first_and_last_RHS_in_committed_1ms_bin', 'unknown phase')
    require(c['full_RHS_tape_available'] is False, 'full tapes require a different validator')
    require(set(c['bindings']) == set(inputs) - {'source', 'contract'}, 'undeclared input')
    for key, digest in c['bindings'].items():
        require(sha256(inputs[key]) == digest, 'bound input changed: ' + key)
    lo, hi = c['domain']
    require(np.isfinite([lo, hi]).all() and lo < hi, 'invalid domain')
    n, k = c['duration_ms'], c['identities']
    require(type(n) is int and type(k) is int and n > 0 and k > 0, 'invalid dimensions')
    with np.load(inputs['support'], allow_pickle=False) as z:
        ids, rows = z['ids'].copy(), z['rows'].copy()
    require(ids.shape == rows.shape == (k,) and ids.dtype.kind in 'iu' and rows.dtype.kind in 'iu',
            'invalid identity shape or type')
    require(len(np.unique(ids)) == len(np.unique(rows)) == k and np.all(np.diff(rows) > 0),
            'ambiguous support')
    data, clocks = {}, {}
    for arm in ('none', 'L', 'R'):
        with np.load(inputs[arm], allow_pickle=False) as z:
            data[arm] = {key: z[key] for key in z.files}
        validate_summary(data[arm], (n, k), lo, hi)
        with np.load(inputs[arm + '_identity'], allow_pickle=False) as z:
            require(np.array_equal(z['PN_ids'], ids) and np.array_equal(z['PN_rows'], rows),
                    'arm identity/order mismatch')
        with np.load(inputs[arm + '_clock'], allow_pickle=False) as z:
            clocks[arm] = z['CNS_time_ns'].copy()
            expected = c['start_ns'] + np.arange(1, n + 1, dtype=np.int64) * 1000000
            require(np.array_equal(clocks[arm], expected), 'recorded CNS clock mismatch')
    fields, candidates = {}, {}
    for field in ('first', 'last'):
        parts = decompose(*(data[a][field] for a in ('none', 'L', 'R')))
        fields[field] = {}
        for name in ('null_common_L', 'null_common_R', 'null_lateral'):
            values = parts[name]
            # Casting is part of the interface, not a numerical rescue or clamp.
            fields[field][name] = {'fp64': domain_summary(values, lo, hi),
                'fp32': domain_summary(values.astype(np.float32), lo, hi)}
            candidates[field + '_' + name] = values
        fields[field]['mean_input_sums'] = {a: float(data[a][field].astype(float).sum(1).mean())
                                           for a in ('none', 'L', 'R')}
        fields[field]['lateral_sum_not_forced_zero'] = float(parts['lateral'].sum(1).mean())
    # Interval arithmetic is conservative: it is a sufficient domain check over
    # any values within the bins, not an observed jointly timed counterfactual.
    bounds = {}
    for arm, other in (('L', 'R'), ('R', 'L')):
        low = data['none']['lo'].astype(float) + (data[arm]['lo'].astype(float) - data[other]['hi']) / 2
        high = data['none']['hi'].astype(float) + (data[arm]['hi'].astype(float) - data[other]['lo']) / 2
        bounds[arm] = {'minimum_possible': float(low.min()), 'maximum_possible': float(high.max()),
                      'guaranteed_inside_domain': bool(np.all(low >= lo) and np.all(high <= hi))}
    counts = {a: data[a]['counts'][:, 0] for a in data}
    coverage = {a: {'RHS_calls': int(x.sum()), 'minimum_per_bin': int(x.min()),
                   'maximum_per_bin': int(x.max()), 'bins_with_more_than_two_calls': int((x > 2).sum()),
                   'bins_with_unordered_variation': int(np.any(data[a]['hi'] != data[a]['lo'], axis=1).sum())}
                for a, x in counts.items()}
    domain_ok = all(fields[f][x]['fp32']['domain_valid'] for f in fields
                    for x in ('null_common_L', 'null_common_R'))
    result = {'schema': 'matrix_signal_contrast_preflight_result_v1', 'input_integrity': True,
              'identities': k, 'duration_ms': n, 'domain': [lo, hi], 'fields': fields,
              'null_common_interval_bounds': bounds, 'coverage': coverage,
              'equal_bin_counts_between_conditions': all(np.array_equal(counts['none'], x) for x in counts.values()),
              'sampled_null_common_domain_valid': domain_ok,
              'exact_replay_identifiable_from_inputs': False,
              'ready_for_exact_component_replay': False, 'new_CNS_ms': 0,
              'stage_admission': None, 'CPU_s': time.process_time() - started,
              'scope': c['scope'],
              'limits': ['First/last sample ordinals need not have equal internal timestamps across arms.',
                         'Extrema/counts do not determine order, timestamps, accepted/rejected RHS, or intermediate values.',
                         'Common/lateral across conditions is not pure direction: total dose can differ.',
                         'No causal result, body prediction, or calibration follows from this preflight.']}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'sampled_counterfactuals.npz', ids=ids, rows=rows, **candidates)
    lines = ['# Preflight de frontera PN', '', c['scope'], '',
             f'{k} identidades, {n} intervalos. CNS nuevo: 0 ms.', '',
             f'Dominio válido en contrafactuales muestreados sin componente común: {domain_ok}.',
             'Reconstrucción exacta RHS desde estos archivos: no identificable.', '',
             '| Campo | Operación | Mínimo FP32 | Máximo FP32 | Entradas fuera de dominio |',
             '|---|---|---:|---:|---:|']
    for f in ('first', 'last'):
        for name in ('null_common_L', 'null_common_R', 'null_lateral'):
            d = fields[f][name]['fp32']
            lines.append(f"| {f} | {name} | {d['min']:.9g} | {d['max']:.9g} | {d['out_of_domain_entries']} |")
    lines += ['', 'No recortar valores ni presentar los resúmenes como cinta completa.',
              'La comprobación del dominio y la de cobertura temporal son independientes.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return {'metrics': {'domain_valid': int(domain_ok), 'exact_replay_identifiable': 0,
                        'new_CNS_ms': 0, 'identities': k}, 'assessment': result}
