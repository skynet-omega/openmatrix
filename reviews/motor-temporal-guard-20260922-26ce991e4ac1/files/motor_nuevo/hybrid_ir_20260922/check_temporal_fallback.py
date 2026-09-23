"""Exercise the actual wrapper's advance with a CPU spy; no body or CUDA mock claim."""
from pathlib import Path
import sys,types,importlib.util,json,numpy as np
R=Path(__file__).resolve().parent
class Spy:
 def advance(self,*args,boundaries=None,**kwargs):return boundaries
saved={k:sys.modules.get(k) for k in ('graph_core','cupy','organism_adapter')}
adapter=types.ModuleType('organism_adapter');adapter.NativeGraph=Spy
sys.modules['organism_adapter']=adapter;sys.modules['cupy']=np
fake=types.ModuleType('graph_core');fake.NativeGraph=Spy;sys.modules['graph_core']=fake
try:
 spec=importlib.util.spec_from_file_location('tested_scope',R/'scoped_graph.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 tests=[]
 for mode in ('baseline','scoped'):
  for continuous in (False,True):
   undo=module.install(None,None,mode,None)
   try:
    g=object.__new__(adapter.NativeGraph);g.event_contract={'continuous_free_rhs':continuous,'temporal_coverage':'UNTRUSTED_BOOLEAN_CANNOT_AUTHORIZE'};g.global_cuts=0;g.scoped_epochs=0
    cuts=np.asarray([.0001125]);got=g.advance(125000,125000,100,125000,boundaries=cuts)
    if got is not cuts or g.global_cuts!=1 or g.scoped_epochs!=0:raise ValueError('Boundary lost without temporal certificate')
    tests.append({'mode':mode,'continuous':continuous,'cuts_retained':True})
   finally:undo()
 result={'status':'PASS','cases':tests,'scope':'Actual advance method, CPU spy parent. Numerical CUDA counterexample has a separate receipt.'}
 (R/'TEMPORAL_FALLBACK_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:
 for k,v in saved.items():
  if v is None:sys.modules.pop(k,None)
  else:sys.modules[k]=v
