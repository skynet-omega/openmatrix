"""Portable readback of selected KC arrays and two accepted-event ledgers."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
NAMES=('native_capture_on_02','reference_capture_on_02')


def accepted(name):
    blocks=json.loads((HERE/name/'EVENT_AUDIT.json').read_text())['blocks']
    start=blocks[0]['start_elapsed_ns']
    events=defaultdict(list)
    for b in blocks:
        if b['duration_ns']==62500:continue
        if b['duration_ns']!=125000:raise ValueError('Unexpected accepted block')
        for e in b['events']:
            key=(e['producer'],e['row'],e['neuron_id'])
            t=b['start_elapsed_ns']+round(e['time_s']*1e9)-start
            events[key].append((t,e['post_q']))
    for v in events.values():v.sort()
    return start,events


def main():
    target=json.loads((HERE/'BOUNDARY.json').read_text())
    for ms in (1,2):
        with np.load(HERE/NAMES[0]/'kc'/f'after_{ms:03d}ms.npz',allow_pickle=False) as a, np.load(
                HERE/NAMES[1]/'kc'/f'after_{ms:03d}ms.npz',allow_pickle=False) as b:
            expected=target['state_max_abs_by_ms'][str(ms)]
            if set(a.files)!=set(b.files) or set(a.files)-{'time_ns'}!=set(expected):
                raise ValueError('KC array schema mismatch')
            if int(a['time_ns'])!=int(b['time_ns']) or int(a['time_ns'])!=target['scientific_time_ns_by_ms'][str(ms)]:
                raise ValueError('KC clock mismatch')
            for key,reference in expected.items():
                x,y=a[key],b[key]
                if x.shape!=y.shape or x.dtype!=y.dtype or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError('KC layout/nonfinite')
                d=np.abs(x-y);idx=np.unravel_index(np.argmax(d),d.shape)
                if list(map(int,idx))!=reference['index'] or float(d[idx])!=reference['max_abs']:
                    raise ValueError('KC maximum changed: '+key)
    origin_a,a=accepted(NAMES[0]);origin_b,b=accepted(NAMES[1])
    if origin_a!=origin_b or origin_a!=target['trial_start_absolute_ns'] or set(a)!=set(b):
        raise ValueError('Event origin/identities differ')
    differences=[]
    for key in a:
        if len(a[key])!=len(b[key]):raise ValueError('Accepted count differs')
        for x,y in zip(a[key],b[key]):
            if x!=y:
                differences.append((min(x[0],y[0]),key,x[0],y[0]))
    first=min(differences)
    frozen=target['first_recorded_accepted_event_difference']
    if list(first[1])!=[frozen['producer'],frozen['row'],frozen['id']] or first[2]!=frozen['native_relative_ns'] or first[3]!=frozen['reference_relative_ns']:
        raise ValueError('First recorded event mismatch')
    print(json.dumps({'selected_state_verified':True,'first_event_id':first[1][2],
                      'first_event_time_ns_native_reference':[first[2],first[3]],
                      'stage3_admission':target['stage3_admission'],
                      'scope':'Two KC endpoint captures and accepted event ledgers; no full-state capture neutrality or full organism rerun.'}))


if __name__=='__main__':main()
