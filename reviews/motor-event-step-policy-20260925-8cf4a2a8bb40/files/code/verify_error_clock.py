"""Independently recompute the read-only error and event-clock diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(arrays_path, audit_path, limiter_path, clock_path, *, corrupt=False):
    limiter = json.loads(Path(limiter_path).read_text())
    clock = json.loads(Path(clock_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    limit_plan = json.loads((HERE/'ERROR_LIMITER_PLAN_38.json').read_text())
    clock_plan = json.loads((HERE/'EVENT_CLOCK_PLAN_39.json').read_text())
    need(sha(arrays_path) == limiter['arrays_sha256'] == clock_plan['arrays_sha256'], 'Array hash')
    need(sha(audit_path) == clock_plan['audit_sha256'], 'Audit hash')
    need(sha(HERE/'ERROR_LIMITER_PLAN_38.json') == limiter['plan_sha256'], 'Limiter plan hash')
    need(sha(HERE/'EVENT_CLOCK_PLAN_39.json') == clock['plan_sha256'], 'Clock plan hash')
    need(sha(HERE/'analyze_event_clock.py') == clock_plan['script_sha256'], 'Clock script hash')
    need(sha(HERE/'probe_error_limiter.py') == limit_plan['script_sha256'], 'Limiter script hash')
    with np.load(arrays_path, allow_pickle=False) as z:
        times = z['times']
        reference = z['reference_error']
        values = z['block_values']
        indices = z['block_indices']
        winner = z['winner_indices']
        reconstructed = z['reconstructed_error']
    need(times.shape == (191, 2) and values.shape == indices.shape == (191, 1404), 'Shapes')
    need(np.isfinite(times).all() and np.isfinite(values).all() and np.isfinite(reference).all(), 'Nonfinite')
    maxima = np.max(values, axis=1)
    winners = indices[np.arange(191), np.argmax(values, axis=1)]
    if corrupt:
        winners[0] = -1
    need(np.array_equal(winners, winner), 'Winner corruption')
    need(np.array_equal(maxima, reconstructed), 'Maxima corruption')
    need(np.max(np.abs(maxima-reference)) <= limit_plan['maximum_error_reconstruction_abs'], 'Norm reconstruction')
    need(limiter['trials'] == 191 and limiter['limiter_unique_rows'] == len(np.unique(winners)), 'Limiter counts')
    need(limiter['limiter_regions']['soma'] == int(np.count_nonzero(winners < 166700)), 'Soma count')
    for got, want in zip(limiter['h_s_quantiles'], np.quantile(times[:,1], [0,.25,.5,.75,1])):
        need(abs(got-want) < 1e-14, 'h quantiles')
    for got, want in zip(limiter['error_quantiles'], np.quantile(maxima, [0,.25,.5,.75,1])):
        need(abs(got-want) < 1e-14, 'error quantiles')
    starts = [0] + [i for i in range(1,191) if times[i,0] < times[i-1,0]-clock_plan['clock_tolerance_s']] + [191]
    need(len(starts) == 17 and len(audit['blocks']) == 16, 'Epoch partition')
    event_count = end_count = noncut = predictor = accepted = 0
    for block, b in enumerate(audit['blocks']):
        first, last = starts[block:block+2]
        end = b['duration_ns']*1e-9
        ev = np.unique([x['time_s'] for x in b['events']])
        need(abs(times[first,0]) <= clock_plan['clock_tolerance_s'], 'Epoch start')
        need(abs(np.sum(times[first:last,1])-end) <= clock_plan['clock_tolerance_s'], 'Epoch duration')
        for i in range(first,last):
            endpoint = sum(times[i])
            at_event = bool(len(ev) and np.min(np.abs(ev-endpoint)) <= clock_plan['clock_tolerance_s'])
            at_end = bool(abs(endpoint-end) <= clock_plan['clock_tolerance_s'])
            event_count += at_event
            end_count += at_end
            noncut += not at_event and not at_end
            predictor += block % 2 == 0
            accepted += block % 2 == 1
    need((event_count,end_count,noncut,predictor,accepted) ==
         tuple(clock[k] for k in ('event_cut_trials','epoch_end_trials','noncut_trials',
                                  'predictor_trials','accepted_trials')), 'Clock counts')
    need(clock['all_trials'] == 191 and clock['status'] == 'PASS_CLOCK_PARTITION', 'Clock status')
    return {'status':'PASS_RECOMPUTED','trials':191,'event_cuts':event_count,
            'epoch_ends':end_count,'noncut':noncut,'error_max_abs':float(np.max(np.abs(maxima-reference)))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrupt', action='store_true')
    args = ap.parse_args()
    base = HERE/'error_limiter_01'
    print(json.dumps(verify(base/'ERROR_LIMITER_ARRAYS.npz', base/'EVENT_AUDIT.json',
                            base/'ERROR_LIMITER_RESULT.json', HERE/'EVENT_CLOCK_RESULT.json',
                            corrupt=args.corrupt)))
