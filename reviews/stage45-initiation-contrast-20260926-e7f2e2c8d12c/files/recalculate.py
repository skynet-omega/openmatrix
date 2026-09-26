"""Recalculate selected observations45 from the portable NPZ (NumPy only).

This reproduces reported observation statistics, not an organism restart,
the unrecorded synaptic currents, or the full independent source/physics audit.
"""
from pathlib import Path
import argparse,json
import numpy as np

def need(ok,why):
    if not ok:raise ValueError(why)

def calculate(folder):
    folder=Path(folder)
    with np.load(folder/'OBSERVACIONES45.npz',allow_pickle=False) as z:
        a={name:{k.split('__',1)[1]:z[k].copy() for k in z.files if k.startswith(name+'__')} for name in ('sham','odor')}
    for name,t in a.items():
        need(np.array_equal(t['paso'],np.arange(1,4001)),'Incomplete sampled life')
        for k,v in t.items():need(np.isfinite(v).all(),'Nonfinite '+k)
        expected=np.vstack([t['initial_DN'],t['DN_q_actual'][:-1]])
        need(np.array_equal(t['DN_q_usada'],expected),'Neural output/reader lag differs')
        raw=np.mean(t['DN_q_usada'][:,:2]-t['DN_baseline'][:,:2],axis=1)
        need(np.array_equal(raw,t['forward_unclipped_mm_s']),'Raw propulsion differs')
        need(np.array_equal(np.clip(raw,0,.5),t['command_forward_mm_s']),'Applied propulsion differs')
    s,o=a['sham'],a['odor']
    need(np.array_equal(s['initial_DN'],o['initial_DN']),'Different prepared release')
    body=all(np.array_equal(s[k],o[k]) for k in ('qpos','qvel','contact_force_N'))
    phases={}
    for name,sl in [('baseline',slice(0,1000)),('stimulus',slice(1000,3000)),('poststimulus',slice(3000,4000))]:
        phases[name]={
            'ORN_mean_difference_LR':[float((o[k][sl]-s[k][sl]).mean()) for k in ('ORN_mean_L','ORN_mean_R')],
            'DN_mean_difference':np.mean(o['DN_q_actual'][sl]-s['DN_q_actual'][sl],axis=0).tolist(),
            'PN_legacy_mean_difference':np.mean(o['PN_q_legacy'][sl]-s['PN_q_legacy'][sl],axis=0).tolist(),
            'forward_command_max_difference':float(np.max(abs(o['command_forward_mm_s'][sl]-s['command_forward_mm_s'][sl])))}
    return dict(phases=phases,body_identical=body,total_new_simulated_ms=0,stage4='OPEN',stage5='OPEN',
                claim='Sensory propagation without DNg100-driven initiation in this prepared model/context')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=Path(__file__).parent)
    args=p.parse_args();result=calculate(args.folder)
    need(result==json.loads((args.folder/'METRICAS45.json').read_text()),'Published metrics differ')
    print(json.dumps(dict(status='OBSERVATION_METRICS_REPRODUCED',body_identical=result['body_identical'],neural_steps=0),indent=2))
