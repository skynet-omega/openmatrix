"""Portable verification of reported observations, not whole-organism replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_probe import summarize,motor_effect,need


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root):
    root=Path(root)
    data={'plan':json.loads((root/'DYNAMIC_PLAN.json').read_text()),
          'result':json.loads((root/'RESULT.json').read_text()),'arms':{}}
    for condition in data['plan']['conditions']:
        p=root/(condition+'_01')
        arm={'meta':json.loads((p/'probe/METADATA.json').read_text()),
             'input':[json.loads(x) for x in (p/'INPUTS.jsonl').read_text().splitlines()]}
        for key,path in [('panel',p/'probe/PANEL.npz'),('trace',p/'traces.npz')]:
            with np.load(path,allow_pickle=False) as z:arm[key]={k:z[k] for k in z.files}
        data['arms'][condition]=arm
    return data


def verify(data):
    plan,result,arms=data['plan'],data['result'],data['arms']
    need(result['stage4_admission'] is False and result['stage5_admission'] is False,'Unsupported stage claim')
    need(result['classification']=='DESCRIPTIVE_CAUSAL_SCREEN_COMPLETE','Unsupported scope')
    need([r['condition'] for r in result['conditions']]==plan['conditions'],'Condition order')
    need(plan['screen']['source_material_q_effect']==.01 and plan['screen']['panel_material_q_effect']==.001,'Materiality criteria changed')
    need(plan['pulse_ns']==[20_000_000,60_000_000] and plan['fraction_remaining_target']==.25,'Pulse contract changed')
    baseline=arms['sham'];bp=baseline['panel'];bt=baseline['trace']
    rebuilt=[]
    for condition,reported in zip(plan['conditions'],result['conditions']):
        arm=arms[condition];p,z,m=arm['panel'],arm['trace'],arm['meta']
        need(p['q'].shape==(240,22) and p['raw'].shape==(240,4,22),'Panel dimensions')
        need(m['ids']==p['ids'].tolist()==baseline['meta']['ids'],'Cell identity differs')
        need(m['source_ids']==plan['sources'][condition],'Source identity differs')
        need(p['phase'].tolist()==['preparacion']*40+['ensayo']*200,'Phase context differs')
        need(p['step'].tolist()==list(range(1,41))+list(range(1,201)),'Sample index context differs')
        need(np.array_equal(p['time_ns'],bp['time_ns']) and np.array_equal(p['time_ns'],z['CNS_time_ns']),'Clock context differs')
        need(np.array_equal(p['q'][:60],bp['q'][:60]),'Prepulse state differs')
        need(arm['input']==baseline['input'] and len(arm['input'])==240,'Paired input context differs')
        for key in ('qpos','qvel','command_forward_mm_s','command_yaw_rate_rad_s','sensores_usados','sensores_pendientes'):
            need(np.array_equal(z[key],bt[key]),'Imposed body or sensors differ: '+key)
        for key in ('q','raw','maximum','counts'):need(np.isfinite(p[key]).all(),'Nonfinite native observation')
        need(np.all((p['q']>=0)&(p['q']<=1)),'State domain')
        gain,theta=np.asarray(m['gain']),np.asarray(m['theta'])
        reconstructed=np.maximum(0,np.tanh(gain*(p['raw'][:,0]+(p['raw'][:,1]+p['raw'][:,2])-theta)))
        need(float(np.max(abs(reconstructed-p['raw'][:,3])))<=1e-12,'Native target formula differs')
        pulse=p['raw'][:,2]
        need(np.all(pulse[:60]==0) and np.all(pulse[100:]==0),'Wrong pulse times')
        sources=np.isin(p['ids'],m['source_ids'])
        need(np.all(pulse[:,~sources]==0),'Pulse outside source cells')
        if sources.any():
            need(np.all(pulse[60:100,sources]>0) and np.all(pulse[60:100,sources]==pulse[60,sources]),'Pulse not fixed/positive')
        if condition!='sham':
            cells=summarize(bp,p,m,plan)
            need(cells==reported['cells'],'Reported cellular effects do not match raw data')
            yaw=float(np.rad2deg(np.max(abs(z['neural_command_shadow'][40:,1]-bt['neural_command_shadow'][40:,1]))))
            forward=float(np.max(abs(z['neural_command_shadow'][40:,0]-bt['neural_command_shadow'][40:,0])))
            need(yaw==reported['max_abs_neural_yaw_command_difference_deg_s'] and forward==reported['max_abs_neural_forward_command_difference_mm_s'],'Reported motor effects differ')
            need(motor_effect(bt,z)==reported['shadow_motor_effect'],'Signed motor effect differs')
            rebuilt.append(dict(condition=condition,cells=cells,yaw_effect_deg_s=yaw,forward_effect_mm_s=forward))
    return dict(status='PORTABLE_OBSERVATIONS_REPRODUCED',arms=len(arms),effects=rebuilt,
        stage4_admission=False,stage5_admission=False,
        scope='Recomputed22-cell and shadow-command observations plus paired input witnesses. Full operators, organism dynamics and numerical convergence not reproduced by this capsule.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--corruption-checks',action='store_true');a=parser.parse_args()
    need(not a.out.exists(),'Refuse overwrite');data=load(a.root);result=verify(data)
    caught=[]
    if a.corruption_checks:
        for kind in ('false_admission','input_context','cell_effect','criterion'):
            x=copy.deepcopy(data)
            if kind=='false_admission':x['result']['stage4_admission']=True
            elif kind=='input_context':x['arms']['DNp09_bilateral']['input'][60]['step']+=1
            elif kind=='criterion':x['plan']['screen']['panel_material_q_effect']=.0001
            else:
                p=x['arms']['DNp09_bilateral']['panel'];j=list(p['ids']).index(10783);i=int(np.argmax(p['q'][:,j]));p['q'][i,j]+=.01
            try:verify(x)
            except ValueError:caught.append(kind)
            else:raise RuntimeError('Undetected corruption: '+kind)
    result['corruptions_rejected']=caught
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],arms=result['arms'],corruptions_rejected=caught)))


if __name__=='__main__':main()
