"""Mechanical campaign stop receipt after a reference timed out at fixed budget."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text())
def need(ok,msg):
    if not ok:raise ValueError(msg)

def main():
    plan=read('PLAN.json');native=read('RAW_VERIFIED_01.json');final=read('RAW_VERIFIED_02.json')
    ref=read('reference_plus_01/RESULT.json')
    prepared=read('PREPARED_COMPARE_causal_cuda.json')
    prefix=read('PREFIX_PARITY_01.json')
    need(native['errors']==native['missing']==[],'Native raw verdict invalid')
    need(native['decision']['classification']=='PROMETEDOR_NO_CONFIRMADO' and
         all(native['decision']['native_checks'].values()),'Native material effect not established')
    need(prepared['status']=='EXACT' and prepared['scientific_state_exact'] is True,'Prepared states differ')
    need(ref['status']=='INCOMPLETE' and ref['error']['type']=='TimeoutError' and
         ref['completed_trial_ms']==391 and ref['completed_preparation_ms']==40 and
         ref['wall_total_s']<=plan['budget']['wall_each_s_max'],'Wrong numerical stop')
    need(final['decision']['classification']=='BLOQUEADO' and
         final['errors']==[{'folder':'campanas/etapa4_mirrored_source_20260924_26/reference_plus_01',
                            'error':'Execution/cleanup failure'}] and final['missing']==[],
         'Final raw verifier mismatch')
    need(prefix['aligned_trial_ms']==391 and prefix['full_gate_admission'] is False,
         'Partial parity mislabeled')
    need(not (HERE/'reference_minus_01').exists(),'Second reference was not stopped after first failure')
    runs={name:read(name+'/RESULT.json') for name in ('native_plus_01','native_minus_01','reference_plus_01')}
    wall=sum(r['wall_total_s'] for r in runs.values())
    need(wall<=plan['budget']['aggregate_wall_s_max'],'Aggregate budget exceeded')
    sources={name:sha(HERE/name) for name in
             ('PLAN.json','CAMPOS.json','SOURCE_LOCK.json','RAW_VERIFIED_01.json',
              'RAW_VERIFIED_02.json','PREPARED_COMPARE_causal_cuda.json',
              'PREFIX_PARITY_01.json','SCALAR_SERIES_01.csv',
              'SCALAR_SERIES_MANIFEST_01.json','TRAJECTORY_DESCRIPTIVE_01.json',
              'PAIR_DIAGNOSTIC.json','CHATGPT_REVIEW_01.md')}
    report={'schema':'stage4_mirrored_source_stop_v1',
            'classification':'BLOQUEADO',
            'native_subresult':'PROMETEDOR_NO_CONFIRMADO',
            'reason':'Native 400-ms mirrored-source-to-command effect is material, but the first strict numerical reference stopped at the registered wall limit after 391/400 ms; the second reference was not run. No full numerical confirmation or closed-loop navigation admission.',
            'stage4_admission':False,'stage5_admission':False,
            'native_metrics':native['decision']['native_pair'],
            'native_checks':native['decision']['native_checks'],
            'prepared_exact':True,
            'reference_plus':{'completed_trial_ms':391,'requested_trial_ms':400,
                              'wall_s':ref['wall_total_s'],'error_type':ref['error']['type'],
                              'prefix_yaw_sup_deg':prefix['yaw_sup_deg'],
                              'prefix_command_L1_deg':prefix['command_L1_deg'],
                              'prefix_only':True},
            'reference_minus':'NOT_LAUNCHED_AFTER_REQUIRED_REFERENCE_PLUS_FAILED',
            'budget':{'organism_runs':3,'native_runs':2,'reference_runs':1,
                      'aggregate_wall_s':wall,'limit_s':plan['budget']['aggregate_wall_s_max']},
            'external_review':{'ChatGPT':'Prelaunch review identified and locally repaired trace-length bug; no execution of organism or final scientific audit.',
                               'Jev':'One bounded task-routing advisory in preceding design campaign; no biological validation.'},
            'sources_sha256':sources,
            'limitations':['Sources were matched in mean only at preparation, not along divergent trajectories.',
                           'Both bodies approached their source mostly through shared forward motion; no online-versus-yoked perturbation.',
                           'Only one prepared state/organism per source side; no animal variability or compatible in-vivo calibration.',
                           'Full strict reference and four-arm numerical parity are absent.',
                           'The body uses an azimuthal roller-assisted effector, not validated six-leg natural gait.'],
            'next_ABC':['A: performance engineering or genuinely faster strict numerical profile at the same equations, with a new prospective budget and parity contract.',
                        'B: prove exact cold continuation of a Gaussian reference from saved full state before using it to complete a long run.',
                        'C: independently validated reference integrator and matched online-versus-replay perturbation; keep numerical and navigation gates separate.']}
    target=HERE/'CLOSE_01.json'
    with target.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'classification':report['classification'],'native_subresult':report['native_subresult'],
                      'reference_completed_ms':391,'aggregate_wall_s':wall},ensure_ascii=False))

if __name__=='__main__':main()
