"""Derive a compact, portable 20-ms DNb05/KC diagnostic from raw saved runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('native_sham_20_01', 'reference_sham_20_01')
IDS = (10176, 10208, 10360, 523769, 10065, 10118)
KC_PATHS = (
    'hybrid/kc_axonal_state/trough',
    'hybrid/kc_axonal_state/previous_slope',
    'hybrid/kc_axonal_state/last_voltage',
    'hybrid/kc_spatial_state/trough',
    'hybrid/kc_spatial_state/previous_slope',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_array(case: str, path: str) -> np.ndarray:
    root = HERE / case / 'final_state'
    node = json.loads((root / 'session.json').read_text())
    for part in path.split('/'):
        node = node[part]
    with np.load(root / 'session.npz', allow_pickle=False) as z:
        return z[node['__array__']].copy()


def main() -> None:
    runs = {c: json.loads((HERE / c / 'RESULT.json').read_text()) for c in CASES}
    if any(r['status'] != 'COMPLETE' or r['completed_trial_ms'] != 20 for r in runs.values()):
        raise ValueError('Incomplete 20-ms run')
    flows = {}
    traces = {}
    for c in CASES:
        with np.load(HERE / c / 'flow/FLOW.npz', allow_pickle=False) as z:
            flows[c] = {k: z[k].copy() for k in z.files}
        with np.load(HERE / c / 'traces.npz', allow_pickle=False) as z:
            traces[c] = {k: z[k].copy() for k in ('DN_q_actual','DN_q_usada','DN_baseline','command_yaw_rate_rad_s','yaw_delta_deg')}
        if tuple(flows[c]['ids']) != IDS or flows[c]['target'].shape != (20, 6):
            raise ValueError('Unexpected focal flow layout')
    a,b = (flows[c] for c in CASES)
    if not np.array_equal(a['time_ns'],b['time_ns']):
        raise ValueError('20-ms flow clocks differ')
    if not all(np.isfinite(z[k]).all() for z in (a,b) for k in ('raw_signed','target')):
        raise ValueError('Nonfinite focal flow')
    kc = {}
    kc_error = {}
    for path in KC_PATHS:
        x,y = (get_array(c,path) for c in CASES)
        if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('KC state layout/finite gate')
        key = path.replace('/','__')
        kc[key+'__native'] = x
        kc[key+'__reference'] = y
        d = np.abs(x-y)
        idx = np.unravel_index(np.argmax(d),d.shape)
        kc_error[path] = {'max_abs':float(d[idx]),'index':list(map(int,idx)),
                          'native_value':float(x[idx]),'reference_value':float(y[idx])}
    np.savez_compressed(HERE/'COMPACT_NUMERIC.npz',
        time_ns=a['time_ns'],ids=a['ids'],
        native_raw=a['raw_signed'],reference_raw=b['raw_signed'],
        native_target=a['target'],reference_target=b['target'],
        native_dn_q=traces[CASES[0]]['DN_q_actual'],
        reference_dn_q=traces[CASES[1]]['DN_q_actual'],
        native_command=traces[CASES[0]]['command_yaw_rate_rad_s'],
        reference_command=traces[CASES[1]]['command_yaw_rate_rad_s'],
        **kc)
    result = {
        'schema':'stage3_dnb05_native_gate_summary_v1',
        'statuses':{c:runs[c]['status'] for c in CASES},
        'wall_s':{c:runs[c]['wall_total_s'] for c in CASES},
        'per_id':{str(identity):{
            'native_target_first':float(a['target'][0,i]),
            'native_target_last':float(a['target'][-1,i]),
            'target_max_abs_between_engines':float(np.max(np.abs(a['target'][:,i]-b['target'][:,i]))),
            'raw_max_abs_between_engines':float(np.max(np.abs(a['raw_signed'][:,i]-b['raw_signed'][:,i])))
        } for i,identity in enumerate(IDS)},
        'dnb05_left_minus_right_target_last':float(a['target'][-1,5]-a['target'][-1,4]),
        'dnb05_left_minus_right_actual_q_last':float(traces[CASES[0]]['DN_q_actual'][-1,2]-traces[CASES[0]]['DN_q_actual'][-1,3]),
        'dna02_both_target_zero_all_samples':bool(np.array_equal(a['target'][:,2:4],np.zeros((20,2)))),
        'kc_errors':kc_error,
        'numerical_gate':json.loads((HERE/'REFERENCE_20_COMPARE.json').read_text())['screen_pass'],
        'observer_on_off_1ms_exact':json.loads((HERE/'SMOKE_COMPARE.json').read_text())['exact_scientific_state'],
        'accepted_events':json.loads((HERE/'ACCEPTED_EVENT_COMPARE.json').read_text()),
        'compact_sha256':sha(HERE/'COMPACT_NUMERIC.npz'),
        'source_sha256':{c:{name:sha(HERE/c/name) for name in ('RESULT.json','flow/FLOW.npz','EVENT_AUDIT.json','final_state/session.json','final_state/session.npz')} for c in CASES},
        'scope':'Two 20-ms sham trajectories, six selected native rate rows, five KC arrays and accepted physical events. No 400-ms error certificate or odor contrast.'
    }
    (HERE/'GATE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'numerical_gate':result['numerical_gate'],
                      'dnb05_target_max_abs':max(result['per_id'][str(i)]['target_max_abs_between_engines'] for i in IDS[-2:]),
                      'kc_trough_max_abs':kc_error[KC_PATHS[0]]['max_abs'],
                      'accepted_events':result['accepted_events']['committed_events_causal']}))


if __name__ == '__main__':
    main()
