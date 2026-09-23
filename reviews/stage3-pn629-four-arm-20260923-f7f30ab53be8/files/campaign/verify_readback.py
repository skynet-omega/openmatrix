"""Portable verification of behavioral/factorial numbers, not hidden CNS state."""
from pathlib import Path
import json,hashlib
import numpy as np
from close import stop_rule,require
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_largo_diagnostico_20260923_10'
def get(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root=HERE,parent_root=PARENT):
    close=get(root/'CLOSE.json');plan=get(root/'PLAN.json')
    require(sha(root/'PLAN.json')==close['plan_sha256'],'Plan mismatch')
    y={}
    for name,row in close['runs'].items():
        folder=root/('full_'+name+'_01');path=folder/'traces.npz'
        require(sha(path)==row['trace_sha256'],'Trace hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            for key in z.files:
                if z[key].dtype.kind in 'fc':require(np.isfinite(z[key]).all(),'Nonfinite trace '+key)
            require(np.array_equal(z['paso'][z['fase']=='ensayo'],np.arange(1,401)),'Trial clock mismatch')
            for ms in plan['readback_ms']:
                indices=np.flatnonzero((z['fase']=='ensayo')&(z['paso']==ms))
                require(len(indices)==1,'Missing observation')
                i=indices[0];require(float(z['yaw_delta_deg'][i])==row['windows'][str(ms)]['yaw_delta_deg'],'Yaw mismatch')
                mask=(z['fase']=='ensayo')&(z['paso']<=ms)
                integral=float(np.rad2deg(np.sum(z['command_yaw_rate_rad_s'][mask])*.001))
                require(integral==row['windows'][str(ms)]['command_integral_deg'],'Command integral mismatch')
            y[name]=row['windows']['400']['yaw_delta_deg']
    p=close['parent']['odor_right'];path=parent_root/'full_odor_right_01/traces.npz'
    require(sha(path)==p['trace_sha256'],'Parent trace hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        i=np.flatnonzero((z['fase']=='ensayo')&(z['paso']==400));require(len(i)==1,'Parent observation')
        parent=float(z['yaw_delta_deg'][i[0]]);require(parent==p['yaw_400_deg'],'Parent yaw mismatch')
    expected=stop_rule(parent,y['odor_right'],plan['material_effect_deg'])
    require(expected or set(y)==set(plan['conditions']),'Favorable result missing required arms')
    require(close['stop_after_sham_right']==expected,'Stop decision mismatch')
    require(close['right_change_vs_parent_deg']==y['odor_right']-parent,'Effect mismatch')
    require(close['right_minus_child_sham_deg']==y['odor_right']-y['sham'],'Sham contrast mismatch')
    require(close['classification']==('DESCARTADO' if expected else 'PROMETEDOR_NO_CONFIRMADO'),'Classification mismatch')
    require(close['stage3_admission'] is False,'Exploratory result promoted improperly')
    body=get(root/'body_factorial_01/RESULT.json');ys={}
    for name,r in body['cases'].items():
        with np.load(root/'body_factorial_01'/name/'trace.npz',allow_pickle=False) as z:
            require(np.isfinite(z['yaw_delta_deg']).all(),'Nonfinite body trace')
            ys[name]=float(z['yaw_delta_deg'][-1]);require(ys[name]==r['final_yaw_deg'],'Body result mismatch')
    a=ys['parent_state__parent_command'];b=ys['parent_state__child_command'];c=ys['child_state__parent_command'];d=ys['child_state__child_command']
    for key,value in [('total_change_deg',d-a),('command_effect_at_parent_deg',b-a),('prepared_state_effect_at_parent_deg',c-a),('interaction_deg',d-c-b+a)]:
        require(body[key]==value,'Factorial mismatch '+key)
    return {'behavioral_numbers_and_stop_rule_rebuilt':True,'factorial_rebuilt':True,
          'scope':'Published behavioral arrays only; full prepared CNS equivalence and organism replay require omitted full checkpoints/assets'}
if __name__=='__main__':print(json.dumps(verify()))
