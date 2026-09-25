"""CPU input/verification tests against the actual donor, never a new organism."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import replay_boundary as r
import verify_pair as v

def run():
    donorpath=r.PRIOR/'native_minus_02/traces.npz'
    donor=v.load_trace(donorpath)
    sha=v.sha(donorpath)
    spec=json.loads((r.PRIOR/'CAMPOS.json').read_text())['minus']
    source=np.array(spec['source_mm'])
    full=np.vstack((donor['sensores_usados'][40:],donor['sensores_pendientes'][-1:]))
    tapes={mode:r.Tape(donorpath,sha,mode) for mode in ('identity','no_contrast')}
    for mode,t in tapes.items():
        expected=full.copy()
        if mode=='no_contrast':expected[:,:2]=np.mean(expected[:,:2],axis=1)[:,None]
        actual=np.array([t.at(t.origin+i*r.DT) for i in range(1001)])
        v.need(np.array_equal(actual,expected),'Incorrect clocked tape')
        v.need(np.array_equal(actual[:,2],full[:,2]),'Third channel changed')
        v.need(np.array_equal(actual[:,:2].mean(axis=1),full[:,:2].mean(axis=1)),'Common intensity changed')
        for time in (t.origin-1,t.origin+1,t.origin+1001*r.DT):
            try:t.at(time)
            except ValueError:pass
            else:raise ValueError('Invalid time accepted')
    # Real traces are used only as verifier fixtures, not fabricated outcomes.
    z={k:x.copy() for k,x in donor.items()}
    z['concentracion_fisica']=z['concentracion_campo'].copy()
    v.verify_arrays(z,donor,'identity',source,spec['sigma_mm'])
    rejected=[]
    corruptions={
        'extra_row':lambda a:a.update(command_yaw_rate_rad_s=np.r_[a['command_yaw_rate_rad_s'],0.]),
        'clock':lambda a:a['CNS_time_ns'].__setitem__(500,a['CNS_time_ns'][500]+1),
        'used_side':lambda a:a['sensores_usados'].__setitem__((500,slice(0,2)),a['sensores_usados'][500,1::-1]),
        'pending_delay':lambda a:a['sensores_pendientes'].__setitem__(500,a['sensores_pendientes'][501]),
        'baseline':lambda a:a['DN_baseline'].__setitem__((500,0),a['DN_baseline'][500,0]+.01),
        'reader':lambda a:a['command_yaw_rate_rad_s'].__setitem__(500,.01),
        'physical_field':lambda a:a['concentracion_fisica'].__setitem__((500,0),0.),
        'quaternion':lambda a:a['qpos'].__setitem__((500,3),2.),
        'nonfinite':lambda a:a['DN_q_actual'].__setitem__((500,0),np.nan)}
    for name,change in corruptions.items():
        a=copy.deepcopy(z);change(a)
        try:v.verify_arrays(a,donor,'identity',source,spec['sigma_mm'])
        except ValueError:rejected.append(name)
        else:raise ValueError('Corruption accepted: '+name)
    v.need(len(rejected)==len(corruptions),'Corruption detection incomplete')
    print(json.dumps(dict(status='PASS_CPU_INPUT_PORT_AND_VERIFIER',sample_times=1001,
        modes=list(tapes),rejected=rejected,donor_sha256=sha,organism_executed=False),sort_keys=True))

if __name__=='__main__':run()
