"""Reconstruct the short KC readback and correct the event-clock origin."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_dnb05_native_20260923_12'
NATIVE='native_capture_on_02'
REFERENCE='reference_capture_on_02'
LIMIT=1e-4


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(name):
    r=json.loads((HERE/name/'EVENT_AUDIT.json').read_text())
    blocks=r['blocks'];start=blocks[0]['start_elapsed_ns']
    accepted=[];predictor=0
    for block in blocks:
        if block['duration_ns']==62500:
            predictor+=len(block['events']);continue
        if block['duration_ns']!=125000:raise ValueError('Unexpected physical block')
        for e in block['events']:
            accepted.append({'producer':e['producer'],'id':e['neuron_id'],'row':e['row'],
                             'relative_ns':block['start_elapsed_ns']+round(e['time_s']*1e9)-start,
                             'post_q':e['post_q']})
    return start,accepted,predictor


def main():
    statuses={name:json.loads((HERE/name/'RESULT.json').read_text()) for name in
              ('native_capture_on_01',NATIVE,REFERENCE,'native_capture_off_01')}
    if statuses['native_capture_on_01']['status']!='INCOMPLETE' or any(
            statuses[n]['status']!='COMPLETE' for n in (NATIVE,REFERENCE,'native_capture_off_01')):
        raise ValueError('Unexpected campaign status')
    freezes={name:json.loads((HERE/name/'FROZEN.json').read_text()) for name in (NATIVE,REFERENCE,'native_capture_off_01')}
    if freezes[NATIVE]!=freezes[REFERENCE] or freezes[NATIVE]!=freezes['native_capture_off_01']:
        raise ValueError('Valid arms did not use same source hashes')
    neutrality=json.loads((HERE/'CAPTURE_NEUTRALITY.json').read_text())
    if not neutrality['engines']['native']['exact_scientific_state']:
        raise ValueError('Native capture changes model state')
    times={};states={}
    for ms in (1,2):
        with np.load(HERE/NATIVE/'kc'/f'after_{ms:03d}ms.npz',allow_pickle=False) as a, np.load(
                HERE/REFERENCE/'kc'/f'after_{ms:03d}ms.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('KC state layouts differ')
            if int(a['time_ns'])!=int(b['time_ns']):raise ValueError('KC clocks differ')
            times[str(ms)]=int(a['time_ns'])
            maxima={}
            for key in a.files:
                if key=='time_ns':continue
                x,y=a[key],b[key]
                if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('KC array layout differs')
                if x.dtype.kind in 'fc' and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                    raise ValueError('Nonfinite KC readback')
                d=np.abs(x-y)
                idx=np.unravel_index(np.argmax(d),d.shape)
                maxima[key]={'max_abs':float(d[idx]),'index':list(map(int,idx)),
                             'native_value':float(x[idx]),'reference_value':float(y[idx])}
            states[str(ms)]=maxima
    starts={};events={};predictors={}
    for name in (NATIVE,REFERENCE):
        starts[name],events[name],predictors[name]=load_events(name)
    if starts[NATIVE]!=starts[REFERENCE]:raise ValueError('Different stimulus clock origins')
    selected={name:[e for e in events[name] if e['id']==81004] for name in (NATIVE,REFERENCE)}
    if len(selected[NATIVE])!=1 or len(selected[REFERENCE])!=1:
        raise ValueError('KC81004 accepted-event count changed')
    grouped={}
    for name in (NATIVE,REFERENCE):
        group={}
        for e in events[name]:
            group.setdefault((e['producer'],e['row'],e['id']),[]).append(e)
        for seq in group.values():seq.sort(key=lambda e:e['relative_ns'])
        grouped[name]=group
    if set(grouped[NATIVE])!=set(grouped[REFERENCE]) or any(
            len(grouped[NATIVE][key])!=len(grouped[REFERENCE][key]) for key in grouped[NATIVE]):
        raise ValueError('Accepted event identities/counts differ')
    differences=[]
    for key in grouped[NATIVE]:
        for x,y in zip(grouped[NATIVE][key],grouped[REFERENCE][key]):
            if x['relative_ns']!=y['relative_ns'] or x['post_q']!=y['post_q']:
                differences.append({'producer':key[0],'row':key[1],'id':key[2],
                                    'native_relative_ns':x['relative_ns'],
                                    'reference_relative_ns':y['relative_ns'],
                                    'time_abs_ns':abs(x['relative_ns']-y['relative_ns']),
                                    'post_abs':abs(x['post_q']-y['post_q']) if x['post_q'] is not None and y['post_q'] is not None else None})
    first=min(differences,key=lambda x:min(x['native_relative_ns'],x['reference_relative_ns'])) if differences else None
    ref_prefix={}
    with np.load(HERE/REFERENCE/'traces.npz',allow_pickle=False) as a, np.load(
            PARENT/'reference_sham_20_01/traces.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Reference trace schemas differ')
        for key in a.files:
            x,y=a[key],b[key][:2]
            ref_prefix[key]=bool(np.array_equal(x,y,equal_nan=x.dtype.kind in 'fc'))
    prior=json.loads((PARENT/'reference_sham_20_01/EVENT_AUDIT.json').read_text())
    current=json.loads((HERE/REFERENCE/'EVENT_AUDIT.json').read_text())
    result={
        'schema':'kc_two_ms_boundary_readback_v1',
        'classification':'EXPLORATORY_BOUNDARY_NOT_IDENTIFYING',
        'stage3_admission':False,
        'absolute_state_limit':LIMIT,
        'trial_start_absolute_ns':starts[NATIVE],
        'kc81004_accepted_event':selected,
        'kc81004_event_time_gap_ns':abs(selected[NATIVE][0]['relative_ns']-selected[REFERENCE][0]['relative_ns']),
        'first_recorded_accepted_event_difference':first,
        'state_max_abs_by_ms':states,
        'scientific_time_ns_by_ms':times,
        'native_capture_final_exact':True,
        'reference_capture_final_exact_tested':False,
        'reference_trace_prefix_exact':all(ref_prefix.values()),
        'reference_trace_fields':len(ref_prefix),
        'reference_event_audit_prefix_exact':current['blocks']==prior['blocks'][:len(current['blocks'])],
        'events':{name:{'predictor':predictors[name],'accepted':len(events[name])} for name in (NATIVE,REFERENCE)},
        'runs':{name:{'status':statuses[name]['status'],'completed_ms':statuses[name]['completed_trial_ms'],
                      'wall_s':statuses[name]['wall_total_s']} for name in statuses},
        'source_hashes':freezes[NATIVE],
        'source_files_sha256':{name:sha(HERE/name/'RESULT.json') for name in statuses},
        'interpretation':'The first recorded accepted event difference is KC76431 at 0.098437 versus 0.100000 ms; KC81004 differs at 0.500000 versus 0.503125 ms. A 1-ms state readback is after both events, so it cannot distinguish event timing from pre-event local integrator error. Reference capture has trace/event-prefix, not full-state, neutrality evidence.'}
    (HERE/'BOUNDARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'first_recorded_event':first,'kc_event_gap_ns':result['kc81004_event_time_gap_ns'],
                      'axon_slope_error_1ms':states['1']['axonal_previous_slope']['max_abs'],
                      'axon_trough_error_2ms':states['2']['axonal_trough']['max_abs'],
                      'native_neutral':result['native_capture_final_exact'],
                      'reference_prefix_exact':result['reference_trace_prefix_exact']}))


if __name__=='__main__':main()
