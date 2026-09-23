"""Structural event scopes; retain cuts until temporal coverage is established.

The published C0-only crossing rule failed an adversarial native CUDA fixture.
No temporal-bound backend is implemented yet. Structural continuity alone never
authorizes discarding a supplied boundary in this version.
"""
import sys,numpy as np,cupy as cp
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph
from model_event_contract import declare

def install(brain,events,mode,output):
 import organism_adapter,json
 parent=organism_adapter.NativeGraph
 class ScopedGraph(NativeGraph):
  def __init__(self,*args,**kwargs):
   self.program,self.event_contract=declare(brain,events.rows)
   super().__init__(*args,**kwargs)
   # Independent runtime falsifier against the actual full coefficient callback.
   # Pinned code proof is required too: zero probes alone cannot certify dependence.
   errors=[]
   with self.stream:
    z=self.x.copy();a,r=self.coefficient(z);ref=(a-z)*r
    for value in (0.,.37,1.):
     pert=z.copy();pert[cp.asarray(events.rows)]=value
     a,r=self.coefficient(pert);delta=(a-pert)*r-ref
     errors.append(float(cp.max(cp.abs(delta)).get()))
   self.stream.synchronize();self.event_contract['actual_full_RHS_probe_errors']=errors
   if self.event_contract['continuous_free_rhs'] and any(e!=0 for e in errors):raise ValueError('Dependency declaration contradicted by full RHS')
   self.event_contract['temporal_coverage']='UNPROVEN_RETAIN_EVENT_CUTS'
   output.write_text(json.dumps(self.event_contract,indent=2)+'\n');self.scoped_epochs=0;self.global_cuts=0
  def advance(self,*args,boundaries=None,**kwargs):
   # C0 at an event does not prevent a post-event contribution falling after all
   # stages of the error estimator. Keep the actual boundary, even with C0.
   if mode not in ('baseline','scoped'):raise ValueError('Unknown event policy')
   self.global_cuts+=0 if boundaries is None else len(boundaries)
   return super().advance(*args,boundaries=boundaries,**kwargs)
 organism_adapter.NativeGraph=ScopedGraph
 def restore():organism_adapter.NativeGraph=parent
 return restore
