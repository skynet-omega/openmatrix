"""Cross saved body states and command streams without running the neural model."""
from pathlib import Path
import importlib.util,hashlib,json,time,resource
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'campanas/etapa3_causal_controls_20260923_11/body_replay.py'
spec=importlib.util.spec_from_file_location('preserved_body_replay',SOURCE)
if spec is None or spec.loader is None:raise RuntimeError('Missing body donor')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)

def main():
    if not __debug__:raise RuntimeError('Diagnostic rejects -O')
    plan=json.loads((HERE/'BODY_FACTORIAL_PLAN.json').read_text())
    dest=HERE/'body_factorial_01';dest.mkdir(exist_ok=False)
    start=time.monotonic();prepared={};command={};reference={};provenance={}
    for name,folder in [('parent',ROOT/'campanas/etapa3_largo_diagnostico_20260923_10/full_sham_01'),('child',HERE/'full_sham_01')]:
        replay.SOURCE=folder;replay.PREPARED=folder/'prepared_state'
        # Use the donor's exact saved-state verification, with the current plan hash.
        r=replay.source_receipt();r['factorial_plan_sha256']=hashlib.sha256((HERE/'BODY_FACTORIAL_PLAN.json').read_bytes()).hexdigest()
        session=replay.read_state(folder/'prepared_state/session')
        prosthesis=replay.read_state(folder/'prepared_state/prosthesis')
        prepared[name]=(session['body'],prosthesis['controller'])
        del session,prosthesis
        with np.load(folder/'traces.npz',allow_pickle=False) as z:
            trial=z['fase']=='ensayo'
            if np.count_nonzero(trial)!=400:raise ValueError('Incomplete source')
            command[name]=np.stack((z['command_forward_mm_s'][trial],z['command_yaw_rate_rad_s'][trial]),axis=1)
            reference[name]={k:z[k][trial].copy() for k in ('qpos','qvel','yaw_delta_deg')}
        provenance[name]=r
    cases={}
    for state,cmd in [('parent','parent'),('child','child'),('parent','child'),('child','parent')]:
        if time.monotonic()-start>plan['budget']['CPU_wall_total_s_max']-30:raise TimeoutError('Body factorial budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>plan['budget']['RAM_GiB_max']:raise MemoryError('Body factorial RAM')
        name=state+'_state__'+cmd+'_command'
        result=replay.run_case(name,*prepared[state],command[cmd],dest/name,
                   {'state':provenance[state],'command':provenance[cmd]},reference[state] if state==cmd else None)
        cases[name]=result
        if state==cmd:
            errors=result['reference_errors']
            if max(errors['max_qpos_abs'],errors['max_qvel_abs'])>plan['gates']['matched_qpos_qvel_max_abs'] or errors['max_yaw_deg_abs']>plan['gates']['matched_yaw_max_abs_deg']:
                raise ValueError('Matched body reproduction failed')
    y={(s,c):cases[s+'_state__'+c+'_command']['final_yaw_deg'] for s in prepared for c in prepared}
    effect_command=y['parent','child']-y['parent','parent']
    effect_state=y['child','parent']-y['parent','parent']
    interaction=y['child','child']-y['child','parent']-y['parent','child']+y['parent','parent']
    result={'cases':cases,'plan':plan,'total_change_deg':y['child','child']-y['parent','parent'],
            'command_effect_at_parent_deg':effect_command,'prepared_state_effect_at_parent_deg':effect_state,
            'interaction_deg':interaction,'wall_total_s':time.monotonic()-start,
            'stage3_admission':False,'scope':'Synthetic body/controller causality only; no neuronal or numerical equivalence'}
    if result['wall_total_s']>plan['budget']['CPU_wall_total_s_max']:raise TimeoutError('Aggregate body budget')
    (dest/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','plan')}))
if __name__=='__main__':main()
