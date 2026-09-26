"""External audit of saved evidence; never changes the frozen trial or thresholds."""
from pathlib import Path
import argparse, json
import numpy as np
from comparison_math import array_error, load_arrays, saved_tree_errors
from compare_runs import need

HERE = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())

def check_trace(folder, ms, origin, *, reference=False):
    trace = load_arrays(folder/'traces.npz')
    expected = origin + np.arange(1, ms+1, dtype=np.int64)*1_000_000
    for key, values in trace.items():
        need(values.ndim >= 1 and len(values) >= ms, 'Incomplete trace: '+key)
        if not reference:
            need(len(values) == ms, 'Unexpected trace length: '+key)
        array_error(values[:ms], values[:ms])
    need(np.array_equal(trace['paso'][:ms], np.arange(1, ms+1)), 'Invalid sample indices')
    for key in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
        need(np.array_equal(trace[key][:ms], expected), 'Stopped or invalid clock: '+key)
    return {'fields': len(trace), 'rows_checked': ms, 'all_clocks_step_ns': 1_000_000}

def initial_neurons(left, right):
    a = load_arrays(left/'neural_states.npz'); b = load_arrays(right/'neural_states.npz')
    need(set(a) == set(b), 'Initial neural schema differs')
    need(all(array_error(a[k][0], b[k][0])['exact'] for k in a), 'Recorded initial neurons differ')
    return len(a)

def initial_trees(left, right):
    out = {}
    for name in ('pn_state', 'published'):
        rows = saved_tree_errors(left/'state_0ms'/name, right/'state_0ms'/name)
        need(all(row['exact'] for row in rows.values()), 'Recorded initial '+name+' differs')
        out[name] = len(rows)
    return out

def final_trees(folder, ms):
    out = {}
    for name in ('pn_state', 'published'):
        path = folder/f'state_{ms}ms'/name
        need(path.with_suffix('.json').is_file() and path.with_suffix('.npz').is_file(), 'Missing final '+name)
        # Traversal checks schema, referenced arrays and scalar finiteness; it
        # deliberately imposes no new numerical PN/publication tolerance.
        rows = saved_tree_errors(path, path)
        need(bool(rows), 'Empty final state tree '+name)
        # PN's nested owners have different time origins. Compare elapsed
        # duration with each recorded initial clock, never force a global epoch.
        def clocks(tree, prefix=''):
            found = {}
            if isinstance(tree, dict):
                for key, value in tree.items():
                    at = prefix+'/'+key
                    if key == 'time_ns':
                        need(type(value) is int, 'Noninteger saved clock '+at)
                        found[at] = value
                    elif isinstance(value, (dict, list)): found.update(clocks(value, at))
            elif isinstance(tree, list):
                for i, value in enumerate(tree): found.update(clocks(value, prefix+'/'+str(i)))
            return found
        initial = clocks(read(HERE/'control_100ms_01/state_0ms'/f'{name}.json'))
        final = clocks(read(path.with_suffix('.json')))
        need(bool(initial) and set(initial) == set(final), 'Final clock schema differs '+name)
        need(all(final[key] == value+ms*1_000_000 for key, value in initial.items()), 'Stale or invalid final clock '+name)
        out[name] = len(rows)
    return out

def event_payloads(left, right, ms):
    a = read(left/'EVENTS.json')[:16*ms]; b = read(right/'EVENTS.json')[:16*ms]
    need(len(a) == len(b) == 16*ms, 'Incomplete event log')
    out = {kind: dict(control_events=0, candidate_events=0, ordered_pairs=0,
           excluded_control_events=0, excluded_candidate_events=0, excluded_blocks=0,
           jump_changed=0, jump_max_abs=0., post_q_presence_changes=0,
           post_q_pairs=0, post_q_changed=0, post_q_max_abs=0., time_max_abs_s=0.)
           for kind in ('committed', 'predictor')}
    for i, (x, y) in enumerate(zip(a, b)):
        row = out['predictor' if i%2 == 0 else 'committed']
        sx, sy = x['events'], y['events']
        row['control_events'] += len(sx); row['candidate_events'] += len(sy)
        for block in (x, y):
            for event in block['events']:
                need(np.isfinite(event['time_s']) and 0 <= event['time_s'] <= block['duration_ns']*1e-9, 'Invalid event time')
                need(np.isfinite(event['jump']) and (event['post_q'] is None or np.isfinite(event['post_q'])), 'Invalid payload')
        identity = lambda events: [(e['row'], e['neuron_id'], e['producer']) for e in events]
        if identity(sx) != identity(sy):
            row['excluded_blocks'] += 1
            row['excluded_control_events'] += len(sx); row['excluded_candidate_events'] += len(sy)
            continue
        for u, v in zip(sx, sy):
            row['ordered_pairs'] += 1
            row['time_max_abs_s'] = max(row['time_max_abs_s'], abs(u['time_s']-v['time_s']))
            row['jump_changed'] += int(u['jump'] != v['jump'])
            row['jump_max_abs'] = max(row['jump_max_abs'], abs(u['jump']-v['jump']))
            row['post_q_presence_changes'] += int((u['post_q'] is None) != (v['post_q'] is None))
            if u['post_q'] is not None and v['post_q'] is not None:
                row['post_q_pairs'] += 1
                row['post_q_changed'] += int(u['post_q'] != v['post_q'])
                row['post_q_max_abs'] = max(row['post_q_max_abs'], abs(u['post_q']-v['post_q']))
    return {'diagnostic_only': True, 'matching': 'Ordered identical identities inside the same block; all exclusions counted', **out}

def audit(long=False):
    control = HERE/'control_100ms_01'; short = HERE/'candidate_100ms_01'
    historical = HERE.parent/'equivalence_1s_20260925_09/optimized_1000ms_01'
    reference = HERE.parent/'equivalence_1s_20260925_09/reference'
    c = read(control/'RESULT.json'); plan = read(HERE/'PLAN.json')
    origin = int(load_arrays(control/'neural_states.npz')['time_ns'][0])
    folders = [(control, 100), (short, 100)]
    if long: folders.append((HERE/'candidate_1000ms_01', 1000))
    rows = {}
    for folder, ms in folders:
        r = read(folder/'RESULT.json'); initial = read(folder/'INITIAL.json')
        need(r['status'] == 'COMPLETE' and r['completed_ms'] == ms and not r.get('cleanup_errors'), 'Incomplete run')
        need(initial['exact'] and initial['method'].startswith('Direct recursive'), 'Missing direct preparation comparison')
        for key in ('source_checkpoint', 'source_trace_sha256'):
            need(r[key] == c[key], 'Preparation provenance differs: '+key)
        need(initial['checkpoint_manifest_sha256'] == read(control/'INITIAL.json')['checkpoint_manifest_sha256'], 'Checkpoint manifest differs')
        need(r['restoration']['checkpoint_manifest_sha256'] == initial['checkpoint_manifest_sha256'], 'Restoration provenance differs')
        for key in ('advance_total_s', 'wall_total_s'):
            need(np.isfinite(r[key]) and r[key] > 0, 'Invalid measured time')
        budget = plan['short_pair']['wall_s_each_max'] if ms == 100 else plan['confirmation']['wall_s_max']
        need(r['advance_total_s'] <= r['wall_total_s'] <= budget, 'Process exceeded frozen budget')
        row = dict(trace=check_trace(folder, ms, origin), initial_neural_fields=initial_neurons(control, folder),
                   initial_trees=initial_trees(control, folder), final_trees=final_trees(folder, ms),
                   process_s=r['wall_total_s'], budget_s=budget, within_budget=True)
        if ms == 1000:
            row['process_targets_minutes'] = {str(m): r['wall_total_s'] <= 60*m for m in plan['targets_process_minutes_per_simulated_second']}
            prior = read(historical/'RESULT.json')
            row['historical_process_s'] = prior['wall_total_s']
            row['historical_fraction_saved_not_controlled'] = 1-r['wall_total_s']/prior['wall_total_s']
        rows[folder.name] = row
    total = c['wall_total_s'] + read(short/'RESULT.json')['wall_total_s']
    need(total <= plan['short_pair']['wall_s_total_max'], 'Short pair exceeded budget')
    prior = read(historical/'RESULT.json')
    for key in ('source_checkpoint', 'source_trace_sha256'):
        need(prior[key] == c[key], 'Historical preparation differs: '+key)
    need(prior['restoration']['checkpoint_manifest_sha256'] == c['restoration']['checkpoint_manifest_sha256'], 'Historical checkpoint differs')
    historical_initial = initial_neurons(control, historical)
    n = 1000 if long else 100
    final_trees(historical, n); final_trees(reference, n)
    check_trace(historical, n, origin, reference=True); check_trace(reference, n, origin, reference=True)
    target = HERE/'candidate_1000ms_01' if long else short
    payloads = {'parent': event_payloads(historical, target, n), 'stable': event_payloads(reference, target, n)}
    return dict(status='INTEGRITY_CONFIRMED', ms=n, runs=rows, short_pair_process_s=total,
                historical_initial_neural_fields=historical_initial,
                historical_initial_scope='Neural0 and source/checkpoint provenance checked. Historical PN0/publication0 were not saved; no direct comparison of those historical trees is claimed. All new runs compare those initial trees against the fresh control.',
                events=payloads, scope='External data-integrity audit, no new engine execution or numerical thresholds. Functional screen and speed target remain separate decisions.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--long', action='store_true'); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = audit(a.long)
    need(not a.out.exists(), 'New audit output required')
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'ms': result['ms'], 'runs': result['runs']}))
