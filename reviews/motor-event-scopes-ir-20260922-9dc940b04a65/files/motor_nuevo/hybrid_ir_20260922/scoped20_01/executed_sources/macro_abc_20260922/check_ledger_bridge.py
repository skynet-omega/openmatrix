"""Regression of the real Ledger→GPU port boundary, including rejected publication."""
from pathlib import Path
import sys,json,numpy as np,cupy as cp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'verification_vendor'))
from event_waveform import Waveform,kernel
from reset_ledger import make_ledger
from reset_ports import ResetFilterPorts
Ledger=make_ledger(Waveform,kernel)
# Reproduce the old published selection exactly, then run the repaired boundary.
import types
from reset_adapter_before_review import prepare as old_prepare
fake_events=types.ModuleType('event_coupling');fake_events.Waveform=Waveform
fake_adapter=types.ModuleType('organism_adapter');fake_adapter.FilterPorts=ResetFilterPorts
saved={k:sys.modules.get(k) for k in ('event_coupling','organism_adapter')}
sys.modules['event_coupling']=fake_events;sys.modules['organism_adapter']=fake_adapter
undo=old_prepare()
try:
 old=fake_events.Waveform([.8],[.2],[.003],.005,.000125);old.add([20e-6],np.array([0]),[0.],post_values=[.7])
 if len(old.times)!=0 or old.post_values!=[]:raise RuntimeError('Old counterexample did not reproduce')
finally:
 undo()
 for k,v in saved.items():
  if v is None:sys.modules.pop(k,None)
  else:sys.modules[k]=v
w=Ledger([.8],[.2],[.003],.005,.000125)
w.add([20e-6],np.array([0]),[0.],post_values=[.7])
if w.post_values!=[.7] or len(w.times)!=1:raise RuntimeError('SET lost by actual ledger')
p=ResetFilterPorts([0],[1]);p.update(w);out=p.project(cp.zeros(2),cp.array([0.,20e-6]),1.).get()
if out[0]!=.7:raise RuntimeError('Zero-delta SET lost before GPU')
errors=[]
for t in (.000125,.00001,.00002,.000125):
 q,s=w.at(t);y=p.project(cp.zeros(2),cp.asarray([0.,t]),1.).get();errors.append(float(np.max(abs(np.array([q[0],s[0]])-y))))
 if errors[-1]>1e-12:raise RuntimeError('Ledger.at and port disagree')
w.add([.00003,.00003],np.array([0,0]),[.1,0.],post_values=[None,.6]);p.update(w)
if p.project(cp.zeros(2),cp.array([0.,.00003]),1.).get()[0]!=.6:raise RuntimeError('Simultaneous ADD/SET order lost')
count=len(w.times)
try:w.add([.00004,.00005],np.array([0,0]),[.1,0.],post_values=[None,float('nan')])
except ValueError:pass
else:raise RuntimeError('Invalid SET accepted')
if len(w.times)!=count or len(w.post_values)!=count:raise RuntimeError('Invalid operation partially published')
# Positive ADD-only behavior remains bit-identical to the legacy implementation.
a=Waveform([.2],[.1],[.003],.005,.000125);b=Ledger([.2],[.1],[.003],.005,.000125)
for w in (a,b):w.add([.00002],np.array([0]),[.1])
for x,y in zip(a.at(.0001),b.at(.0001)):
 if x.tobytes()!=y.tobytes():raise RuntimeError('Changed ADD-only path')
r={'old_zero_SET_loss_reproduced':True,'zero_SET_reaches_GPU':True,'maximum_CPU_GPU_difference':max(errors),'simultaneous_order':True,'failed_publication_atomic':True,'ADD_only_bit_identical':True,'whole_organism_rerun':False}
(R/'LEDGER_BRIDGE_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
