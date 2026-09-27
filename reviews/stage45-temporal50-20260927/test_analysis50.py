"""Corruption checks and numerical design controls; no organism execution."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from analyze50 import HERE,ARMS,expected_schedule,contrast,gate,check_trace,check_dng,read_npz,read_json,need,save,sha,PLAN_HASH

def test(folder=HERE,real=False):
    start=time.process_time();folder=Path(folder);results={}
    def reject(name,fn):
        try:
            fn()
        except (ValueError,KeyError) as exc:
            results[name]=dict(rejected=True,error=str(exc))
        else:
            raise ValueError('Corruption escaped: '+name)
    x={a:expected_schedule(a) for a in ARMS}
    need(all(np.array_equal(v.sum(axis=0),[60.,60.]) for v in x.values()),'Marginal input')
    for i in (0,1):
        for j in (0,1):
            c=contrast({a:((v[:,0]==i)&(v[:,1]==j)).astype(float) for a,v in x.items()})
            need(abs(c['mean_J'])<1e-12,'Instantaneous stationary null')
    separable={a:np.cumsum(v[:,0]**2)+2*np.cumsum(v[:,1]**3) for a,v in x.items()}
    need(np.max(np.abs(contrast(separable)['J']))<1e-12,'Separable history null')
    temporal={}
    for a,v in x.items():
        l,r=v.T
        temporal[a]=np.r_[np.zeros(40),l[:-40]]*r-np.r_[np.zeros(40),r[:-40]]*l
    need(abs(contrast(temporal)['mean_J'])>.1,'Delayed positive control')
    drift={a:np.exp(-np.arange(140)/60.)*v[:,0]*v[:,1] for a,v in x.items()}
    need(abs(contrast(drift)['mean_J'])>.01,'Known time-varying instantaneous counterexample')
    need(gate(1.6e-5,.02)['screen_pass'],'Prospective inclusive boundary')
    need(not gate(np.nextafter(1.6e-5,0),.02)['screen_pass'],'Neural boundary')
    need(not gate(1.6e-5,np.nextafter(.02,0))['screen_pass'],'Command boundary')
    need(not gate(0,5)['screen_pass'] and not gate(1,0)['screen_pass'],'Both gates required')
    need(sha(folder/'PLAN.json')==PLAN_HASH,'Frozen criteria hash')
    if real:
        a='p00';t=read_npz(folder/a/'traces.npz');o=read_npz(folder/a/'input_and_observers.npz');owner=read_json(folder/a/'TEMPORAL_OWNER.json')
        initial=read_npz(folder/'reference/initial.npz');prefix=read_npz(folder/'reference/prefix49.npz');spec=read_npz(folder/'reference/input_spec.npz')
        def check(tt=t,oo=o,own=owner):check_trace(a,tt,oo,own,initial,prefix,spec)
        check()
        for name,key,index in [('clock','CNS_time_ns',(30,)),('latency','DN_q_usada',(30,2)),('baseline','DN_baseline',(30,2)),('decoder','neural_yaw_unapplied_rad_s',(30,))]:
            tt={k:v.copy() for k,v in t.items()};tt[key][index]+=1
            reject(name,lambda tt=tt:check(tt=tt))
        oo={k:v.copy() for k,v in o.items()};oo['nominal_Hz'][30,0]+=1
        reject('input',lambda:check(oo=oo))
        oo={k:v.copy() for k,v in o.items()};oo['ORN_ids'][0]+=1
        reject('identity',lambda:check(oo=oo))
        own=copy.deepcopy(owner);own['instantaneous_state_clamped']=True
        reject('flag',lambda:check(own=own))
        dd=read_npz(folder/a/'dng100_observed.npz');check_dng(dd,t)
        changed={k:v.copy() for k,v in dd.items()};changed['committed'][0]=True
        reject('context',lambda:check_dng(changed,t))
        changed={k:v.copy() for k,v in dd.items()};changed['records'][0,138]=1-changed['records'][0,138]
        reject('acceptance',lambda:check_dng(changed,t))
        changed={k:v.copy() for k,v in dd.items()};changed['records'][0,10]+=1
        reject('margin',lambda:check_dng(changed,t))
    return dict(schema='temporal50_corruptions_v1',real_record_checks=real,cases=results,
        design_controls=True,threshold_boundaries=True,known_instantaneous_drift_J=contrast(drift)['mean_J'],
        CPU_s=time.process_time()-start,new_neural_ms=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=HERE);p.add_argument('--real',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    r=test(a.folder,a.real)
    if a.output:
        need(not a.output.exists(),'Preserve earlier test result');save(a.output,r)
    print(json.dumps(r))
