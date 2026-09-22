"""External review's downstream-event falsifier, executed on actual CUDA core."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
import numpy as np,cupy as cp
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph
from event_ports import FilterPorts
tau=125e-6;results=[]
for fraction in (.9,.900003):
 event=fraction*tau;exact=.25*((tau-event)/tau)**2*np.exp(-(tau-event)/tau)
 for aligned in (False,True):
  port=FilterPorts([0],[1]);port.update(SimpleNamespace(rows=[0],q=np.array([0.]),s=np.array([0.]),tau=np.array([tau]),ts=tau,times=[event],jumps=[.5]))
  rate=cp.asarray([0.,0.,1/tau])
  def coefficients(z):return cp.stack((z[0],z[1],z[1])),rate
  core=NativeGraph(np.zeros(3),coefficients,rtol=1e-5,atol=1e-7,norm_size=3,project=port.project,native_library=HERE/'libgraph_control_v2.so')
  _,count,error=core.advance(125000,125000,100,125000,boundaries=[event] if aligned else None)
  y=core.x.get(stream=core.stream);core.close();err=abs(y[2]-exact)
  if aligned and err>1e-6:raise RuntimeError('Downstream event still missed')
  if not aligned and y[2]!=0.:raise RuntimeError('Legacy counterexample no longer reproduced')
  results.append(dict(aligned=aligned,event_s=event,z=float(y[2]),analytic_z=exact,absolute_error=err,steps=count,max_estimated_error=error))
(HERE/'EVENT_BOUNDARY_CHECK.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
