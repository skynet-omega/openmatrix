"""Execution adapter for the unchanged full-mass membrane equations.

Retains the original PN object, identity, calcium chemistry and serialization.
The GPU proposal is imported/exported at each declared electrical boundary so
outer predictor rollback and incoming histories remain authoritative.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,time
import numpy as np
import cupy as cp

def install(brain):
 folder=Path('/home/daroch/AXIOMA_ASTRA/motor_nuevo/resident_pn_20260922')
 sys.path.insert(0,str(folder))
 from resident import Backend,Calcium
 from graph_step import advance_graph_resident
 p=brain._online_source.pn;original=p.advance
 start=time.perf_counter()
 view=SimpleNamespace(cp=cp,backend=Backend(p.backend),active_nodes=p.active_nodes,_nodes_gpu=cp.asarray(p.active_nodes),gbar_nS=cp.asarray(p.gbar_nS),reversal_mV=cp.asarray(p.reversal_mV),leak_reversal_mV=p.leak_reversal_mV,calcium_port=Calcium(p.calcium_port),_assert_model=p._assert_model)
 report={'calls':0,'accepted':0,'rejected':0,'build_s':time.perf_counter()-start,'wall_s':0.,'model_identity':p.identity,'state_owner_preserved':True}
 def advance(ns,current,**options):
  if '_local_channel' in options:raise ValueError('Physical session owns calcium')
  start=time.perf_counter()
  view.voltage=cp.asarray(p.cp.asnumpy(p.voltage));view.gates=cp.asarray(p.gates);view.ionic_charge_pC=cp.asarray(p.ionic_charge_pC);view.time_ns=p.time_ns
  view.calcium_port.gates=cp.asarray(p.calcium_port.gates)
  result=advance_graph_resident(view,ns,current,_local_channel=view.calcium_port,**options)
  if result['accepted']:
   p.voltage=p.cp.asarray(cp.asnumpy(view.voltage));p.gates=cp.asnumpy(view.gates);p.ionic_charge_pC=cp.asnumpy(view.ionic_charge_pC);p.time_ns=view.time_ns
   report['accepted']+=1
  else:report['rejected']+=1
  report['calls']+=1;report['wall_s']+=time.perf_counter()-start
  report['graph_statistics']=getattr(view,'graph_statistics',{}).copy();report['linear_fallbacks']=view.backend.fallbacks
  return result
 p.advance=advance
 return report,lambda:setattr(p,'advance',original)
