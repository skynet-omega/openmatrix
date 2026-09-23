"""Candidate controller: real event boundaries and independent scalar ODE."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,json,ctypes as ct
import numpy as np,cupy as cp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
T=Path(__file__).resolve().parent;sys.path.insert(0,str(T.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph
from event_ports import FilterPorts
def reference(initial,tq,ts,events,t):
 matrix=np.array([[-1/tq,0.],[1/ts,-1/ts]]);state=np.asarray(initial,dtype=float);last=0.
 for at,jump in sorted(events):
  if at>t:break
  state=expm(matrix*(at-last))@state;state[0]+=jump;last=at
 return expm(matrix*(t-last))@state

events=np.array([4.99e-6,9.61e-6,62.04e-6,75.31e-6,88.85e-6,100e-6,121.875e-6]);jumps=np.full(len(events),.02);end=125e-6
def exact(t):return reference([.1,.2],.003,.005,list(zip(events,jumps)),t)
def rhs(t,y):return [2000*(.2+.5*np.tanh(3*exact(t)[1])-y[0])]
solution=solve_ivp(rhs,(0,end),[.1],rtol=1e-11,atol=1e-13,max_step=5e-7)
expected=np.r_[exact(end),solution.y[0,-1]];reports={}
for mode,library in [('parent',T.parent/'pipeline_review_20260922/libgraph_control_trace.so'),('nominal',T/'libgraph_control_nominal.so')]:
 holder={}
 def freeze(g):
  port=FilterPorts([0],[1]);port.update(NS(q=np.array([.1]),s=np.array([.2]),tau=np.array([.003]),ts=.005,times=events,rows=np.zeros(len(events),dtype=int),jumps=jumps));holder['port']=port
 def project(y,clock,f):return holder['port'].project(y,clock,f)
 def coefficient(y):
  a=y.copy();rate=cp.zeros(3);a[2]=.2+.5*cp.tanh(3*y[1]);rate[2]=2000.;return a,rate
 graph=NativeGraph([.1,.2,.1],coefficient,rtol=1e-5,atol=1e-7,norm_size=3,project=project,freeze=freeze,native_library=library)
 nxt,counts,error=graph.advance(125000,31250,100,1000000,boundaries=events)
 graph.lib.engine_trace_count.argtypes=[ct.c_void_p];graph.lib.engine_trace_count.restype=ct.c_long;graph.lib.engine_trace_copy.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double),ct.c_long]
 trace=np.zeros((graph.lib.engine_trace_count(graph.handle),10));graph.lib.engine_trace_copy(graph.handle,trace.ctypes.data_as(ct.POINTER(ct.c_double)),trace.size)
 for at,h,*_ in trace:
  if np.any((events>at+1e-15)&(events<at+h-1e-15)):raise ValueError('Trial skipped forcing boundary')
 value=graph.x.get();difference=float(max(abs(value-expected)))
 if not np.isfinite(value).all() or difference>1e-5:raise ValueError('Controller differs from scalar ODE reference')
 reports[mode]={'accepted':counts[0],'rejected':counts[1],'maximum_error_estimate':error,'reference_max_difference':difference,'next_proposal_ns':nxt,'events_preserved':True}
 graph.close()
result={'status':'PASS','runs':reports,'scope':'Small nonlinear receiver with exact two-state source; prospective method control, not whole-organism admission.'}
(T/'NOMINAL_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
