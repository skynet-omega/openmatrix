from types import SimpleNamespace as NS
from pathlib import Path
import copy,json
import numpy as np
from operator_state import OperatorState
owner=NS(a=np.arange(6,dtype=float),nested={'b':np.array([.2,.4])})
registry=OperatorState(owner,{'a':('a',),'b':('nested','b')});saved=registry.state_dict()
owner.a[:]=9;owner.nested['b'][:]=-2
restored=OperatorState.from_state(saved,expected_bindings={'a':('a',),'b':('nested','b')});restored.restore(owner)
if restored.differences(owner):raise RuntimeError('Registry roundtrip differs')
bad=copy.deepcopy(saved);bad['values']['a'][0]=3
try:OperatorState.from_state(bad,expected_bindings={'a':('a',),'b':('nested','b')})
except ValueError:pass
else:raise RuntimeError('Corruption accepted')
# A later invalid destination must not permit an earlier partial write.
owner.a[:]=5;owner.nested['b']=np.ones(3)
try:restored.restore(owner)
except ValueError:pass
else:raise RuntimeError('Changed destination shape accepted')
if not np.array_equal(owner.a,np.full(6,5.)):raise RuntimeError('Partial operator publication')
report={'status':'PASS','roundtrip_exact':True,'corruption_rejected':True,'layout_checked_before_writes':True}
(Path(__file__).parent/'OPERATOR_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
