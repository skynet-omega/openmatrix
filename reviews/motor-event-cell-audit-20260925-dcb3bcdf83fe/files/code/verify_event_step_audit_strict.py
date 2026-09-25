"""Prospective addendum: recheck raw event identities and separate trial phases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_events(a, b):
    aa, bb = a['blocks'], b['blocks']
    need(len(aa) == len(bb), 'block count')
    out = {'records': 0, 'speculative_records': 0, 'confirmed_records': 0,
           'speculative_blocks': 0, 'confirmed_blocks': 0,
           'max_time_s': 0., 'max_jump': 0., 'max_post_q': 0.}
    for x, y in zip(aa, bb):
        need((x['block'], x['start_elapsed_ns'], x['duration_ns']) ==
             (y['block'], y['start_elapsed_ns'], y['duration_ns']), 'block identity')
        need(x['duration_ns'] in (62500, 125000), 'unknown phase')
        need(len(x['events']) == len(y['events']), 'event count')
        phase = 'speculative' if x['duration_ns'] == 62500 else 'confirmed'
        out[phase + '_blocks'] += 1
        out[phase + '_records'] += len(x['events'])
        for u, v in zip(x['events'], y['events']):
            need((u['row'], u['neuron_id'], u['producer']) ==
                 (v['row'], v['neuron_id'], v['producer']), 'event identity')
            for z, duration in ((u, x['duration_ns']), (v, y['duration_ns'])):
                for key in ('time_s', 'jump'):
                    need(math.isfinite(float(z[key])), 'nonfinite ' + key)
                need(0 <= z['time_s'] <= duration * 1e-9 + 1e-15,
                     'event outside block')
                if z['post_q'] is not None:
                    need(math.isfinite(float(z['post_q'])), 'nonfinite post_q')
            need((u['post_q'] is None) == (v['post_q'] is None), 'post_q null')
            out['records'] += 1
            out['max_time_s'] = max(out['max_time_s'], abs(u['time_s']-v['time_s']))
            out['max_jump'] = max(out['max_jump'], abs(u['jump']-v['jump']))
            if u['post_q'] is not None:
                out['max_post_q'] = max(out['max_post_q'], abs(u['post_q']-v['post_q']))
    need(out['speculative_blocks'] == out['confirmed_blocks'] == 160,
         'paired half/full phase')
    starts = [x['start_elapsed_ns'] for x in aa if x['duration_ns'] == 125000]
    need(starts == list(range(starts[0], starts[0] + 160*125000, 125000)),
         'confirmed chronology')
    out['confirmed_span_ms'] = 160*125000/1e6
    return out


def verify():
    p = HERE/'EVENT_STEP_PAIR_LONG_RESULT.json'
    pair = json.loads(p.read_text())
    names = ('event_step_baseline_long_01', 'event_step_candidate_long_01')
    audit = [json.loads((HERE/n/'EVENT_AUDIT.json').read_text()) for n in names]
    found = check_events(*audit)
    need(found['records'] == pair['event_count'], 'reported event count')
    for key, receipt_key in (('max_time_s', 'event_time_max_abs_s'),
                             ('max_jump', 'event_jump_max_abs'),
                             ('max_post_q', 'event_post_q_max_abs')):
        need(math.isclose(found[key], pair[receipt_key], rel_tol=1e-10, abs_tol=1e-15),
             'reported ' + key)
    capsule = np.load(HERE/'EVENT_STEP_LONG_CAPSULE.npz')
    x, y = capsule['baseline_state'], capsule['candidate_state']
    need(np.isfinite(x).all() and np.isfinite(y).all(), 'nonfinite final state')
    rtol, atol = float(capsule['baseline_rtol']), float(capsule['baseline_atol'])
    error = np.abs(x-y)/(atol+rtol*np.maximum(np.abs(x), np.abs(y)))
    index = int(np.argmax(error))
    need(math.isclose(float(error[index]), pair['normalized_state_error_max'],
                      rel_tol=1e-12), 'reported max state error')
    auxiliary = {}
    for name in ('pending_excitation', 'pending_sensors', 'muscles_activation',
                 'plasticity_factors'):
        a, b = capsule['baseline_'+name], capsule['candidate_'+name]
        need(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite '+name)
        auxiliary[name] = float(np.max(np.abs(a-b)))
    # Test each newly protected field without mutating files or the original logs.
    for field, value in (('start_elapsed_ns', -1), ('neuron_id', -1),
                         ('time_s', float('nan')), ('jump', float('nan')),
                         ('post_q', float('nan'))):
        bad = copy.deepcopy(audit[1])
        target = bad['blocks'][0] if field == 'start_elapsed_ns' else bad['blocks'][0]['events'][0]
        target[field] = value
        try:
            check_events(audit[0], bad)
        except ValueError:
            pass
        else:
            raise ValueError('corruption passed: '+field)
    return {'schema': 'event_step_strict_audit_v1', 'pair_sha256': sha(p),
            'audit_sha256': {n: sha(HERE/n/'EVENT_AUDIT.json') for n in names},
            'status': 'PASS_STRICT_EVENT_LOG_CHECK', 'events': found,
            'final_state_max_index': index,
            'final_state_max_normalized_error': float(error[index]),
            'final_state_at_index': [float(x[index]), float(y[index])],
            'auxiliary_endpoint_max_abs': auxiliary,
            'corruptions_rejected': ['start_elapsed_ns','neuron_id','time_s','jump','post_q'],
            'scope': 'Endpoint and logged half/full trials only; no full neural trajectory certification.'}


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--out', type=Path, required=True)
    a = arg.parse_args()
    need(not a.out.exists(), 'output exists')
    report = verify()
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'events':report['events'],
                      'final_state_max_index':report['final_state_max_index']}))
