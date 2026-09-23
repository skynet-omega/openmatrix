"""Recompute endpoint evidence and instrumentation equivalence from saved arrays.

This is an offline check, not an organism run or a whole-body resume test.
"""
from pathlib import Path
import sys,json
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from verify_transport import verify,leaves,read_state
from qualify_transport import qualify

one=verify(H/'smoke_reference_01',H/'smoke_causal_01')
twenty=qualify(H.parent/'causal_runtime_20260922/reference20_01',H.parent/'causal_runtime_20260922/device20_01')
left,right=leaves(H/'smoke_causal_01'),leaves(H/'trace_smoke_02')
changed=[]
for path in sorted(set(left)|set(right)):
    a,b=left.get(path),right.get(path)
    equal=type(a) is type(b)
    if equal:
        equal=(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()) if isinstance(a,np.ndarray) else a==b
    if not equal:changed.append(path)
if changed:raise RuntimeError('Instrumentation changed exported physical state: '+str(changed))
summary={}
records=json.loads((H/'CNS_TRACES.json').read_text())['epochs']
for phase in ('discarded_predictor','accepted_exchange'):
    epochs=[e for e in records if e['phase']==phase]
    rows=np.asarray([r for e in epochs for r in e['trace']])
    bits=rows[:,7].astype(int)
    summary[phase]={'epochs':len(epochs),'trials':len(rows),'accepted':int(sum(rows[:,9]==1)),
        'rejected':int(sum(rows[:,9]==0)),
        **{name:int(sum((bits&mask)!=0)) for mask,name in [(1,'reason_event'),(2,'reason_end'),(4,'reason_proposed_step'),(8,'reason_maximum')]},
        'ties_count_as_multiple_reasons':True,'max_error':float(max(rows[:,8])),
        'proposal_limited_with_error_below_point1':int(sum(((bits&4)!=0)&(rows[:,8]<.1)))}
if summary!=json.loads((H/'CNS_CAUSES.json').read_text())['summary']:raise RuntimeError('Trace summary differs from raw trials')
reference=read_state(H/'smoke_reference_01/brain_final');candidate=read_state(H/'smoke_causal_01/brain_final')
index=int(np.argmax(abs(reference['state']-candidate['state'])))
diagnostic=json.loads((H/'TRANSIENT_DIAGNOSTIC.json').read_text())
if index!=diagnostic['largest_state_index'] or float(abs(reference['state'][index]-candidate['state'][index]))!=diagnostic['absolute_difference']:raise RuntimeError('Transient diagnostic differs')
result={'one_ms':one,'twenty_ms':twenty,'trace_equivalence_exact_all_exported_leaves':True,
    'CNS_causes_from_trials':summary,'transient_index':index,
    'status':'PROMETEDOR_NO_CONFIRMADO','stage3_admission':False,
    'scope':'Reconstructed engineering evidence; no new tolerance, trajectory convergence or biological equivalence.'}
(H/'RECONSTRUCTED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'one_ms_pass':one['screen_pass'],'twenty_ms_partial_pass':twenty['historical_partial_screen']['screen_pass'],'unbounded_changed_fields':len(twenty['uncovered_changed_numeric_fields']),'instrumentation_exact':True,'status':result['status']}))
