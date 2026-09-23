from pathlib import Path
import sys,json
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'causal_runtime_20260922'))
# This import executes the existing grouping/retry fixture once before extension.
from check_device import fixture
b,engine,ledger=fixture([0,500]);b.q.fill(0);b.last_siz.fill(0);b.previous_slope.fill(1);b.trough.fill(-65)
received=[]
def fail_after_enqueue(t,r,j):
 received.extend(t.tolist());raise RuntimeError('INJECT_EVENT_PUBLICATION_FAILURE')
engine.events.active.add=fail_after_enqueue
try:engine.advance(b,125000,np.zeros((2,4)),np.zeros((2,4)))
except RuntimeError as e:
 if str(e)!='INJECT_EVENT_PUBLICATION_FAILURE':raise
else:raise RuntimeError('Publication fixture did not trigger')
if not received or not engine.publication_failed or not b._native_publication_invalid:raise RuntimeError('Failed publication not invalidated')
try:engine.advance(b,125000,np.zeros((2,4)),np.zeros((2,4)))
except RuntimeError as e:
 if 'invalidated' not in str(e):raise
else:raise RuntimeError('Partly published owner reused')
result={'status':'PASS','kernel_succeeded_before_injected_failure':True,'receiver_partially_mutated':True,'owner_invalidated':True,'retry_rejected':True,'claim':'Fail closed after publication error, not automatic rollback/recovery.'}
(H/'PUBLICATION_FAILURE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
