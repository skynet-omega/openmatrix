"""Regression for a publisher that is created at first physical advance."""
from types import SimpleNamespace
import event_guard
class Base:
 def advance(self,*args,**kw):raise RuntimeError('original')
class FakeCore:
 def __init__(self,b,events):
  if not hasattr(b,'_motor_axonal_callback'):raise RuntimeError('eager initialization')
  self.report={'initialized':True};self.calls=0
 def advance(self,b,*args,**kw):self.calls+=1;return args,kw
b=Base();s=SimpleNamespace(profile='causal_cuda',brain=SimpleNamespace(_spatial_batch=b),cell=SimpleNamespace(core=None,report={}),events=object(),implementations={})
saved=event_guard.EventGuardCell
try:
 event_guard.EventGuardCell=FakeCore;event_guard.attach(s)
 if s.cell.core is not None:raise RuntimeError('must remain deferred')
 b._motor_axonal_callback=object()
 if b.advance(125000,mode='test')!=((125000,),{'mode':'test'}):raise RuntimeError('argument forwarding')
 b.advance(125000)
 if s.cell.core.calls!=2 or s.implementations['membrane']['module']!='event_guard':raise RuntimeError('ownership/provenance')
finally:event_guard.EventGuardCell=saved
print('PASS: deferred publisher, single owner, arguments and actual source identity')
