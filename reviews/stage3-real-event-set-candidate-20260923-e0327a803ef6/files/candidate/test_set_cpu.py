"""CPU gate: physical producer unchanged; SET preserves its declared post."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from event_waveform import Waveform,lif_record,kernel

HERE=Path(__file__).resolve().parent
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922/event_waveform.py')


def original():
    spec=importlib.util.spec_from_file_location('original_event_waveform',OLD)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    old=original()
    x=json.loads((HERE.parent/'etapa3_evento_20260923_06/full_sham_capture_01/DOMAIN_CAPTURE.json').read_text())
    p=x['port'];q0=p['q0'];s0=p['s0'];tau=p['tau_q_s'];ts=p['tau_s_s'];te=p['times_s'][0];jump=p['jumps'][0];duration=x['owner']['duration_s']
    w=Waveform([q0],[s0],[tau],ts,duration)
    w.add([te],np.array([0]),[jump],posts=[1.])
    wo=old.Waveform([q0],[s0],[tau],ts,duration)
    wo.add([te],np.array([0]),[jump])
    at=np.array([z[0] for z in w.at(te)])
    legacy=np.array([z[0] for z in wo.at(te)])
    end=np.array([z[0] for z in w.at(duration)])
    s_before=s0*math.exp(-te/ts)+q0*float(kernel(te,tau,ts))
    if at[0]!=1. or legacy[0]!=1.0000000000000002 or abs(at[1]-s_before)>1e-15 or not 0<=end[0]<=1:
        raise ValueError('Actual event SET/continuity gate failed')
    add=Waveform([q0],[s0],[tau],ts,duration);add.add([te],np.array([0]),[.1])
    add_old=old.Waveform([q0],[s0],[tau],ts,duration);add_old.add([te],np.array([0]),[.1])
    if any(not np.array_equal(a,b) for a,b in zip(add.at(duration),add_old.at(duration))):
        raise ValueError('Pure ADD changed on CPU')
    zero=Waveform([q0],[s0],[tau],ts,duration);zero.add([te],np.array([0]),[0.],posts=[.7])
    if zero.at(te)[0][0]!=.7 or len(zero.times)!=1:
        raise ValueError('SET with zero auxiliary jump was dropped')
    # Independent synthetic LIF producer: adding post metadata must leave
    # its physical state, event times and effective jumps bitwise unchanged.
    dt=.000125;tq=.024;t_event=.00003125
    base=[np.array([-1.]),np.array([0.]),np.array([0],dtype=np.int64),np.array([.7]),
          np.array([1.]),np.array([math.log(2)/t_event]),np.array([-1.]),np.array([0.]),
          np.array([1/(.5*tq)]),np.array([tq])]
    a=[v.copy() for v in base];b=[v.copy() for v in base]
    prior=old.lif_record(*a,dt);current=lif_record(*b,dt)
    if not all(x.tobytes()==y.tobytes() for x,y in zip(a,b)) or prior[0]!=current[0] or any(not np.array_equal(x,y,equal_nan=True) for x,y in zip(prior[1:],current[1:3])):
        raise ValueError('Producer changed physical state or original events')
    if current[3][0,0]!=1. or prior[0]!=1:
        raise ValueError('Synthetic physical clipped post not recorded')
    result={'schema':'explicit_event_set_cpu_fixture_v1',
            'real_event_legacy_q':float(legacy[0]),'real_event_set_q':float(at[0]),
            'real_event_set_s':float(at[1]),'s_continuity_max_abs':float(abs(at[1]-s_before)),
            'end_q':float(end[0]),'pure_add_bitwise_unchanged':True,
            'zero_jump_set_retained':True,'producer_state_and_legacy_events_bitwise_unchanged':True,
            'synthetic_clipped_post':float(current[3][0,0]),
            'scope':'CPU algebra and one synthetic LIF spike; no organism or CUDA.'}
    (HERE/'CPU_FIXTURE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
