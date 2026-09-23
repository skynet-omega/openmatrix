"""Mechanical closure of the fixed-budget native-sideband round."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=read(HERE/'PLAN.json')
    smoke={name:read(HERE/name/'RESULT.json') for name in ('smoke_off_01','smoke_on_01')}
    cmp=read(HERE/'SMOKE_COMPARE.json')
    run=read(HERE/'full_sham_01/RESULT.json')
    flow=read(HERE/'full_sham_01/flow/RESULT.json')
    if any(x['status']!='COMPLETE' or x['completed_trial_ms']!=1 for x in smoke.values()):
        raise ValueError('Unexpected smoke receipts')
    if not cmp['exact_scientific_state'] or flow['max_target_error']>1e-9:
        raise ValueError('Observer gate did not pass')
    if run['status']!='INCOMPLETE' or run['completed_preparation_ms']!=40 or run['completed_trial_ms']!=79:
        raise ValueError('Unexpected full-arm outcome')
    diag=run['error']['numerical_diagnostic']
    if run['error']['message']!='accepted event state outside domain' or diag['sample_indices']!=[28296] or diag['sample_values']!=[1.0000000000000002]:
        raise ValueError('Numerical failure identity changed')
    if flow['samples']!=119 or list(HERE.glob('full_*_01')) != [HERE/'full_sham_01']:
        raise ValueError('Unexpected full runs or flow samples')
    with np.load(HERE/'full_sham_01/traces.npz',allow_pickle=False) as z:
        if len(z['fase'])!=119 or not np.array_equal(z['fase'],['preparacion']*40+['ensayo']*79):
            raise ValueError('Unexpected trace prefix')
    nodes=pd.read_parquet(OLD/'data/male_v10/nodes.parquet',columns=['bodyId','type','somaSide','instance'])
    node=nodes.iloc[28296].to_dict()
    if node['bodyId']!=41645 or not node['type'].startswith("KCa'b'"):
        raise ValueError('Failed state coordinate identity changed')
    refs={name:sha(HERE/name) for name in ('PLAN.json','native_tap.py','run_kernel.py','compare_smoke.py',
            'SMOKE_COMPARE.json','full_sham_01/RESULT.json','full_sham_01/flow/FLOW.npz')}
    result={
        'schema':'stage3_native_sideband_close_v1',
        'classification':'BLOQUEADO_POR_DOMINIO_NUMERICO',
        'stage3_pass':None,
        'instrument_gate':'PASS_1MS_EXACT_SCIENTIFIC_STATE',
        'native_coefficient_target_max_abs':flow['max_target_error'],
        'budget':{'smoke_used':2,'smoke_max':plan['budgets']['organism_smoke_1ms_max'],
                  'full_started':1,'full_max':plan['budgets']['organism_full_400ms_max'],
                  'reference_used':0,'other_full_arms_started':0},
        'full_sham_prefix':{'preparation_ms':40,'trial_ms':79,'flow_samples':119,
            'wall_s':run['wall_total_s'],'failure':run['error']['message'],
            'attempted_state_index':28296,'attempted_value':1.0000000000000002,
            'node':node,'next_step_s':diag['attempted_clock_s'][1]},
        'interpretation':'A one-ulp event-filter domain excess halted the protected run before 320/400 ms. This does not identify a stiff biological circuit. No other odor arms were launched after the first failed gate. Native PN/DNa02 raw flow is observational and not path-decomposed.',
        'cleanup_limitation':run['cleanup_errors'],
        'source_sha256':refs,
    }
    (HERE/'CLOSE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'classification':result['classification'],'trial_ms':79,'node':node},ensure_ascii=False))


if __name__=='__main__':main()
