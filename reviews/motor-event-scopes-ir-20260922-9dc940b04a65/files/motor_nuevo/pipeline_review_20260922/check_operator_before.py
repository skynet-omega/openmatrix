from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('before',H/'operator_state_before.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=NS(tau=np.array([1.,2.]),theta=np.array([3.,4.]));snap=m.OperatorState(a,{'tau':('tau',),'theta':('theta',)}).state_dict();snap['bindings']['tau'],snap['bindings']['theta']=snap['bindings']['theta'],snap['bindings']['tau'];r=m.OperatorState.from_state(snap);r.restore(a)
b=NS(a=np.array([1.,2.]),b=np.array([3.,4.]));r=m.OperatorState(b,{'a':('a',),'b':('b',)});b.a[:]=8;b.b[:]=9;b.b.flags.writeable=False
try:r.restore(b)
except ValueError:pass
result={'changed_bindings_accepted':not r.bindings=={},'swapped_tau':a.tau.tolist(),'swapped_theta':a.theta.tolist(),'readonly_second_left_partial_write':bool(np.array_equal(b.a,[1,2]))}
(H/'OPERATOR_BEFORE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
