"""Recompute the prespecified PN629 stop decision from actual parent/child traces."""
from pathlib import Path
import hashlib,json
import numpy as np
from analyze_long import main as analyze,ARMS
from compare_preparation import main as preparation

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_largo_diagnostico_20260923_10'
def read(p):return json.loads(p.read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stop_rule(parent_right,child_right,material):
    delta=child_right-parent_right
    return abs(delta)<material or delta>0

def main():
    plan=read(HERE/'PLAN.json');analyze();preparation()
    data=read(HERE/'LONG_DIAGNOSTIC.json');prep=read(HERE/'PREPARATION_COMPARE.json')
    smoke=read(HERE/'CAPTURE_NEUTRALITY.json')
    require(smoke['engines']['smoke']['exact_scientific_state'],'Capture failed')
    names=set(data['arms']);require({'sham','odor_right'}<=names,'Need completed sham/right')
    frozen=None;intervention=None;contracts=[];rows={};total=0.
    for name in sorted(names):
        root=HERE/ARMS[name];r=read(root/'RESULT.json');c=read(root/'RUN_CONTRACT.json')
        require(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Incomplete '+name)
        require(r['completed_preparation_ms']==40 and r['completed_trial_ms']==400,'Clock '+name)
        require(r['engine']=='causal_cuda' and r['observer']=='on','Profile '+name)
        require(c['plan_sha256']==sha(HERE/'PLAN.json'),'Plan drift '+name)
        require(r['wall_total_s']<=plan['budget']['wall_single_long_s_max'],'Wall '+name)
        current=read(root/'FROZEN.json')
        for path,digest in current.items():require(sha(root/'executed_sources'/path)==digest,'Executed source modified '+path)
        if frozen is None:frozen=current
        require(current==frozen,'Sources differ '+name)
        x=read(root/'INTERVENTION.json')
        require(x['only_changes']==['general_outputs.enabled','record_sha256'] and x['dynamic_466_enabled'] is True,'Intervention scope '+name)
        require(x['anatomy_changed'] is False and x['motor_decoder_changed'] is False,'Intervention drift '+name)
        if intervention is None:intervention=x
        require(x==intervention,'Intervention differs '+name)
        require(data['preparation_exact'][name]['equal'],'Preparation traces '+name)
        if name!='sham':require(prep['arms'][name]['scientific_state_exact'],'Prepared state '+name)
        flow=read(root/'flow/RESULT.json')
        require(flow['samples']==440 and flow['max_target_error']<=1e-9 and flow['max_rate_error']<=1e-12,'Native flow '+name)
        rows[name]={'windows':data['arms'][name]['windows'],'wall_s':r['wall_total_s'],
                    'native_target_max_error':flow['max_target_error'],'trace_sha256':sha(root/'traces.npz')}
        contracts.append(c);total+=r['wall_total_s']
    require(all(c['checkpoint_manifest_sha256']==contracts[0]['checkpoint_manifest_sha256'] for c in contracts),'Checkpoint origin differs')
    for smoke_name in ('smoke_on_01','smoke_off_01'):
        r=read(HERE/smoke_name/'RESULT.json');require(r['status']=='COMPLETE','Smoke incomplete')
        total+=r['wall_total_s']
    require(total<=plan['budget']['aggregate_wall_s_max'],'Aggregate budget')
    parent={}
    for name in ('sham','odor_right'):
        root=PARENT/ARMS[name]
        with np.load(root/'traces.npz',allow_pickle=False) as z:
            index=np.flatnonzero((z['fase']=='ensayo')&(z['paso']==400))
            require(len(index)==1 and np.isfinite(z['yaw_delta_deg']).all(),'Invalid parent trace')
            parent[name]={'yaw_400_deg':float(z['yaw_delta_deg'][index[0]]),'trace_sha256':sha(root/'traces.npz')}
        c=read(root/'RUN_CONTRACT.json')
        require(c['checkpoint_manifest_sha256']==contracts[0]['checkpoint_manifest_sha256'],'Parent checkpoint differs')
    right=rows['odor_right']['windows']['400']['yaw_delta_deg'];p=parent['odor_right']['yaw_400_deg']
    stop=stop_rule(p,right,plan['material_effect_deg'])
    require(stop or names==set(ARMS),'Favorable right requires left/uniform before closing')
    result={'classification':'DESCARTADO' if stop else 'PROMETEDOR_NO_CONFIRMADO',
        'scope':'PN629-disabled route as sufficient material rescue; not exclusion of downstream, history, or other PN routes',
        'stop_after_sham_right':stop,'stage3_admission':False,'runs':rows,'parent':parent,
        'right_change_vs_parent_deg':right-p,'right_minus_child_sham_deg':right-rows['sham']['windows']['400']['yaw_delta_deg'],
        'preparation_within_child_exact':True,'executed_sources_within_child_exact':True,
        'intervention':intervention,'wall_all_runs_s':total,'wall_budget_s':plan['budget']['aggregate_wall_s_max'],
        'limitations':['466 dynamic left PN outputs retained; this is not full adapter homologation',
                      'Different parent/child prepared neural states are part of intervention/history effect',
                      'No 400ms numerical certificate or biological DNb05 decoder calibration'],
        'plan_sha256':sha(HERE/'PLAN.json')}
    (HERE/'CLOSE.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('classification','stop_after_sham_right','right_change_vs_parent_deg','right_minus_child_sham_deg','wall_all_runs_s')}))
if __name__=='__main__':main()
