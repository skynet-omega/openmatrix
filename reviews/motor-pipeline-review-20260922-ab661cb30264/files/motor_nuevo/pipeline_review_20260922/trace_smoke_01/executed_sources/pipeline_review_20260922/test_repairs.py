from pathlib import Path
from types import SimpleNamespace as NS
import sys,tempfile,copy,json,unittest.mock as mock
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'causal_runtime_20260922'))
import operator_state as op
from run_storage import RunStorage,verify_snapshot
checks=[]
bindings={'a':('a',),'b':('b',)};owner=NS(a=np.array([1.,2.]),b=np.array([3.,4.]));registry=op.OperatorState(owner,bindings);saved=registry.state_dict()
bad=copy.deepcopy(saved);bad['bindings']['a'],bad['bindings']['b']=bad['bindings']['b'],bad['bindings']['a']
try:op.OperatorState.from_state(bad,expected_bindings=bindings)
except ValueError:checks.append('swapped_routes_rejected')
else:raise RuntimeError('Changed routes accepted')
restored=op.OperatorState.from_state(saved,expected_bindings=bindings);owner.a[:]=8.;owner.b[:]=9.;owner.b.flags.writeable=False
try:restored.restore(owner)
except ValueError:pass
else:raise RuntimeError('Read-only destination accepted')
if not np.array_equal(owner.a,[8,8]):raise RuntimeError('Prevalidation changed earlier field')
checks.append('readonly_no_partial_write');owner.b.flags.writeable=True
real_copy=op._copy;calls=0
def fail_once(dst,src):
 global calls
 calls+=1
 if calls==2:raise RuntimeError('Injected transfer failure')
 real_copy(dst,src)
with mock.patch.object(op,'_copy',fail_once):
 try:restored.restore(owner)
 except RuntimeError:pass
 else:raise RuntimeError('Transfer failure not observed')
if not np.array_equal(owner.a,[8,8]) or not np.array_equal(owner.b,[9,9]):raise RuntimeError('Transfer rollback failed')
checks.append('transfer_failure_rollback_exact');calls=0
def fail_persistently(dst,src):
 global calls
 calls+=1
 if calls>=2:raise RuntimeError('Persistent transport failure')
 real_copy(dst,src)
with mock.patch.object(op,'_copy',fail_persistently):
 try:restored.restore(owner)
 except RuntimeError:pass
 else:raise RuntimeError('Persistent failure not observed')
if not owner._operator_restore_invalid:raise RuntimeError('Partial owner not invalidated')
checks.append('failed_rollback_invalidates_owner')
with tempfile.TemporaryDirectory() as td:
 root=Path(td);out=root/'run';storage=RunStorage(out);(out/'sentinel').write_text('unchanged')
 try:RunStorage(out)
 except FileExistsError:pass
 else:raise RuntimeError('Existing run reused')
 if (out/'sentinel').read_text()!='unchanged':raise RuntimeError('Existing run modified')
 checks.append('existing_run_untouched')
 def failed_writer(folder):
  (folder/'first').write_text('partial');raise KeyboardInterrupt()
 try:storage.snapshot('final_state',failed_writer)
 except KeyboardInterrupt:pass
 else:raise RuntimeError('Checkpoint failure not propagated')
 if (out/'final_state').exists():raise RuntimeError('Failed snapshot published')
 checks.append('interrupted_snapshot_unpublished')
 storage.snapshot('final_state',lambda folder:(folder/'data').write_text('complete'))
 verify_snapshot(out/'final_state');(out/'final_state/data').write_text('corrupt')
 try:verify_snapshot(out/'final_state')
 except ValueError:checks.append('snapshot_corruption_rejected')
 else:raise RuntimeError('Bad snapshot hash accepted')
result={'status':'PASS','checks':checks};(H/'REPAIR_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
