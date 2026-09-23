"""GPU gate: SET uses the physical post; ordinary ADD follows old kernel."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cupy as cp
import numpy as np

from event_waveform import Waveform
from event_ports import FilterPorts

HERE=Path(__file__).resolve().parent
OLD=Path('/home/daroch/AXIOMA_ASTRA/campanas/etapa3_motor_nuevo_20260922/event_ports.py')


def original_ports():
    spec=importlib.util.spec_from_file_location('original_event_ports',OLD)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.FilterPorts


def project(cls,w,t):
    p=cls(np.array([0]),np.array([1]));p.update(w)
    return cp.asnumpy(p.project(cp.zeros(2),cp.asarray([0.,t]),1.))


def main():
    x=json.loads((HERE.parent/'etapa3_evento_20260923_06/full_sham_capture_01/DOMAIN_CAPTURE.json').read_text())
    a=x['port'];q0=a['q0'];s0=a['s0'];tau=a['tau_q_s'];ts=a['tau_s_s'];te=a['times_s'][0];jump=a['jumps'][0];dur=x['owner']['duration_s']
    w=Waveform([q0],[s0],[tau],ts,dur);w.add([te],np.array([0]),[jump],posts=[1.])
    got=project(FilterPorts,w,te);expected=np.array([z[0] for z in w.at(te)])
    if got[0]!=1. or np.max(abs(got-expected))>1e-12:
        raise ValueError('Physical SET failed in actual CUDA kernel')
    add=Waveform([q0],[s0],[tau],ts,dur);add.add([te],np.array([0]),[.1])
    a_new=project(FilterPorts,add,te);a_old=project(original_ports(),add,te)
    if not np.array_equal(a_new,a_old):
        raise ValueError('Ordinary ADD changed in CUDA kernel')
    zero=Waveform([q0],[s0],[tau],ts,dur);zero.add([te],np.array([0]),[0.],posts=[.7])
    z=project(FilterPorts,zero,te)
    if z[0]!=.7:raise ValueError('SET with zero auxiliary jump dropped in CUDA')
    first=Waveform([q0],[s0],[tau],ts,dur)
    first.add([te,te],np.array([0,0]),[0.,.1],posts=[.7,np.nan])
    second=Waveform([q0],[s0],[tau],ts,dur)
    second.add([te,te],np.array([0,0]),[.1,0.],posts=[np.nan,.7])
    order_a,order_b=project(FilterPorts,first,te),project(FilterPorts,second,te)
    if abs(order_a[0]-.8)>1e-15 or order_b[0]!=.7:
        raise ValueError('Same-mark SET/ADD ordering failed')
    result={'schema':'explicit_event_set_gpu_fixture_v1',
            'real_event_cuda_set_q':float(got[0]),'real_event_cuda_set_s':float(got[1]),
            'cpu_cuda_max_abs':float(np.max(abs(got-expected))),
            'ordinary_add_bitwise_unchanged':True,
            'zero_jump_set_q':float(z[0]),
            'same_mark_set_then_add_q':float(order_a[0]),
            'same_mark_add_then_set_q':float(order_b[0]),
            'scope':'One- and two-event CUDA fixtures, no organism.'}
    (HERE/'GPU_FIXTURE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
