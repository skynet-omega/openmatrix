"""Reconstruct the four-arm campaign receipt from saved traces and contracts."""
from __future__ import annotations
import json
from pathlib import Path
from analyze_long import main as analyze

HERE=Path(__file__).resolve().parent
ARMS={'sham':'full_sham_01','odor_left':'full_odor_left_01',
      'odor_right':'full_odor_right_01','uniform':'full_uniform_01'}

def get(path):return json.loads(path.read_text())
def require(test,message):
    if not test:raise ValueError(message)

def main():
    analyze()
    plan=get(HERE/'PLAN.json')
    data=get(HERE/'LONG_DIAGNOSTIC.json')
    preparation=get(HERE/'PREPARATION_COMPARE.json')
    smoke=get(HERE/'SMOKE_COMPARE.json')
    require(smoke['exact_scientific_state'],'Observer on/off one-ms smoke differs')
    require(set(data['arms'])==set(ARMS),'Missing or extra odor condition')
    require(all(x['equal'] for x in data['preparation_exact'].values()),
            'Pre-odor trace differs')
    require(all(preparation['arms'][name]['scientific_state_exact'] for name in ARMS if name!='sham'),
            'Pre-odor serialized scientific state differs')
    hashes={name:get(HERE/folder/'FROZEN.json') for name,folder in ARMS.items()}
    require(all(h==hashes['sham'] for h in hashes.values()),'Executed sources differ')
    total=0.
    rows={}
    for name,folder in ARMS.items():
        receipt=get(HERE/folder/'RESULT.json')
        flow=get(HERE/folder/'flow/RESULT.json')
        x=data['arms'][name]
        require(receipt['status']=='COMPLETE' and receipt['error'] is None and not receipt['cleanup_errors'],
                'Incomplete run: '+name)
        require(receipt['completed_preparation_ms']==40 and receipt['completed_trial_ms']==400,
                'Incomplete biological clock: '+name)
        require(receipt['observer']=='on' and receipt['engine']=='causal_cuda',
                'Different execution profile: '+name)
        require(x['odor_exposure_exact'] and x['motor_command_and_one_ms_lag_exact'],
                'Odor or motor decoder check failed: '+name)
        require(flow['samples']==440 and flow['max_target_error']<=1e-9
                and flow['max_rate_error']<=1e-12,'Native tap check failed: '+name)
        require(receipt['wall_total_s']<=plan['budgets']['wall_single_400ms_s_max'],
                'Per-arm wall budget exceeded: '+name)
        total+=receipt['wall_total_s']
        rows[name]={'yaw_320_deg':x['windows']['320']['yaw_delta_deg'],
                    'yaw_400_deg':x['windows']['400']['yaw_delta_deg'],
                    'command_integral_400_deg':x['command_integral_trial_deg'],
                    'DNa02_nonzero_target_samples':x['DNa02_native_target_nonzero_samples'],
                    'native_target_max_error':flow['max_target_error'],
                    'wall_s':receipt['wall_total_s']}
    require(total<=plan['budgets']['wall_total_s_max'],'Aggregate wall budget exceeded')
    right=rows['odor_right']['yaw_400_deg']
    output={'schema':'stage3_four_arm_long_close_v1',
            'classification':'PROMETEDOR_NO_CONFIRMADO',
            'stage3_admission':False,
            'reason':'Right absolute yaw is not negative at 400 ms; sham drifts; long-horizon two-engine fidelity and model PN symmetry are unverified.',
            'right_absolute_yaw_negative_400ms':bool(right<0),
            'frozen_source_file_count':len(hashes['sham']),
            'all_source_manifests_exact':True,
            'all_preodor_states_exact':True,
            'observer_one_ms_exact':True,
            'runs':rows,
            'yaw_minus_sham_400_deg':{
                name:data['contrasts_to_sham'][name]['400']['yaw_minus_sham_deg']
                for name in ARMS if name!='sham'},
            'wall_four_runs_s':total,
            'wall_budget_s':plan['budgets']['wall_total_s_max'],
            'numerical_scope':'Native target reconstruction and one-ms observer neutrality only; 20-ms hidden-KC reference gate remains failed.',
            'biological_scope':'One male connectome and one body state; provisional PN adapters and DNb05 decoder; no equivalence claim.'}
    (HERE/'CLOSE.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'classification':output['classification'],
                      'stage3_admission':False,'wall_four_runs_s':total,
                      'yaw_400_deg':{k:v['yaw_400_deg'] for k,v in rows.items()}},ensure_ascii=False))

if __name__=='__main__':main()
