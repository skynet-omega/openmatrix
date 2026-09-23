"""Reconstruct the event-loss finding from stored source operands and receipts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cap=read(HERE/'full_sham_capture_01/DOMAIN_CAPTURE.json')
    audit=read(HERE/'ARITHMETIC_AUDIT.json')
    run=read(HERE/'full_sham_capture_01/RESULT.json')
    chat=read(HERE/'chatgpt_real_case_01/RESULTADO.json')
    prior=read(HERE.parent/'etapa3_evento_20260923_05/COMPARE_PREFIX.json')
    if run['completed_trial_ms']!=79 or run['error']['message']!='accepted event state outside domain':
        raise ValueError('Unexpected organism failure')
    if cap['row']!=28296 or cap['event_slot']!=109 or cap['port_q_row']!=28296 or cap['port']['count']!=1:
        raise ValueError('Unexpected event identity')
    if not audit['owner_port_inputs_exact'] or audit['classification']!='ENCODED_EVENTS_ABOVE_DOMAIN_IN_HIGH_PRECISION':
        raise ValueError('Owner/port discrepancy or arithmetic result changed')
    if not prior['same_failure_receipt'] or not prior['prefix_bitwise_equal']:
        raise ValueError('Tap on/off control failed')
    if chat['status']!='COMPLETE' or chat['result']['add_q_fp64']!=cap['attempted_fine_q']:
        raise ValueError('Independent arithmetic replay differs')
    saved=HERE.parent/'etapa3_flujo_kernel_20260923_04/smoke_on_01/final_state/session'
    descriptor=read(saved.with_suffix('.json'))['hybrid']['kc_spatial_manifest']['inputs']['rows']
    with np.load(saved.with_suffix('.npz'),allow_pickle=False) as arrays:
        gamma_rows=arrays[descriptor['__array__']]
    if 28296 in gamma_rows:
        raise ValueError('Failed row uses spatial commit instead of LIF recorder')
    port=cap['port'];q=port['q0'];tau=port['tau_q_s']
    dt=cap['owner']['duration_s'];t=port['times_s'][0];jump=port['jumps'][0]
    source_value=q*math.exp(-dt/tau)
    source_before=source_value*math.exp((dt-t)/tau)
    source_post=source_before+jump
    if source_post!=1. or jump!=1.-source_before:
        raise ValueError('Cannot derive physical SET(1) from source LIF formula')
    result={
        'schema':'stage3_actual_failed_event_causal_receipt_v1',
        'classification':'EVENT_POST_VALUE_LOST_IN_ADD_ENCODING',
        'stage3_pass':None,
        'organism_run':{'preparation_ms':40,'trial_ms':79,'failure':run['error']['message'],
                        'failed_row':cap['row'],'bodyId':41645,'event_slot':cap['event_slot'],
                        'attempted_q':cap['attempted_fine_q']},
        'owner_port_inputs_exact':True,
        'event_producer':'recorded_lif non-gamma KC; row absent from 1557 spatial gamma rows',
        'source_lif_formula':{'q0':q,'tau_s':tau,'duration_s':dt,'event_time_s':t,
                              'value_at_interval_end_before_event':source_value,
                              'before_event':source_before,'effective_jump':jump,
                              'post_event_from_source':source_post},
        'encoded_history_minus_one_high_precision':audit['high_precision_minus_one'],
        'independent_replays':{'local':audit['classification'],
                               'ChatGPT_code_actual_case':chat['result']['add_q_fp64'],
                               'ChatGPT_code_sha256':sha(HERE/'chatgpt_auditar_evento.py')},
        'budget':{'organism_full_used':1,'arithmetic_replays_used':2},
        'limits':'Proof concerns this source LIF event representation and failed KC port. A production SET/ADD change still needs independent state/event/numerical validation; the failure state is not resumable.',
        'sha256':{name:sha(HERE/name) for name in ('PLAN.json','capture_event.py','run_capture.py',
                    'full_sham_capture_01/DOMAIN_CAPTURE.json','ARITHMETIC_AUDIT.json',
                    'REAL_EVENT_CASE.json','chatgpt_real_case_01/RESULTADO.json')},
    }
    (HERE/'CLOSE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'classification':result['classification'],
                      'source_post':source_post,'attempted_q':cap['attempted_fine_q']}))


if __name__=='__main__':main()
