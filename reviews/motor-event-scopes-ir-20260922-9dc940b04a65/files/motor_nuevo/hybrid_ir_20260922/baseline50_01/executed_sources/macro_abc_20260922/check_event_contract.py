from pathlib import Path
from types import SimpleNamespace as NS
import sys,json,math,numpy as np,cupy as cp
from scipy.linalg import expm
from reset_ports import ResetFilterPorts
from diagnostic_graph import NativeGraph
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'verification_vendor'))
from event_waveform import lif_record
from lif_event_metadata import lif_record_with_post
# This source-adapter comparison uses the unmodified historical function.
inputs=[np.full(4,-60.),np.zeros(4),np.zeros(4,dtype=np.int64),np.array([.2,.8,.99,1.]),np.full(4,20.),np.full(4,100000.),np.full(4,-60.),np.full(4,-40.),np.full(4,10.),np.full(4,.024),.000125]
a=[x.copy() if isinstance(x,np.ndarray) else x for x in inputs];b=[x.copy() if isinstance(x,np.ndarray) else x for x in inputs]
x=lif_record(*a);y=lif_record_with_post(*b)
for l,r in zip(a,b):
 if isinstance(l,np.ndarray) and l.tobytes()!=r.tobytes():raise RuntimeError('Changed physical source')
for l,r in zip(x,y[:3]):
 if isinstance(l,np.ndarray):same=l.tobytes()==r.tobytes()
 else:same=l==r
 if not same:raise RuntimeError('Changed events')
# Same primitive, three mathematical domains. No bounds/clamp in the port.
errors=[]
for q0,s0,post in ((.2,.1,.8),(-60.,-70.,-55.),(3.,2.,6.)):
 w=NS(q=np.array([q0]),s=np.array([s0]),tau=np.array([.003]),ts=.005,times=[.00002,.00004,.00004],rows=[0,0,0],jumps=[.1,0.,.03],post_values=[None,post,None])
 port=ResetFilterPorts([0],[1]);port.update(w);matrix=np.array([[-1/.003,0.],[1/.005,-1/.005]])
 for t in (0.,.000019,.00002,.000039,.00004,.00005,.000125):
  v=np.array([q0,s0]);last=0.
  for at,j,p in zip(w.times,w.jumps,w.post_values):
   if at>t:break
   v=expm(matrix*(at-last))@v;v[0]=v[0]+j if p is None else p;last=at
  expected=expm(matrix*(t-last))@v;actual=port.project(cp.zeros(2),cp.asarray([0.,t]),1.).get();error=float(max(abs(expected-actual)));errors.append(error)
  if error>1e-12:raise RuntimeError('Port differs from independent jump reference')
# Same existing executor, model-supplied domains; no core change per model.
initial=np.array([.2,-60.,3.]);target=np.array([.8,-55.,6.]);rates=np.array([50.,100.,25.])
target_gpu,rates_gpu=cp.asarray(target),cp.asarray(rates)
def coeff(x):return target_gpu,rates_gpu
g=NativeGraph(initial,coeff,rtol=1e-5,atol=1e-8,norm_size=3,state_bounds=([0.,-100.,0.],[1.,50.,np.inf]))
try:
 g.advance(100000,100000,100,100000);expected=initial+(-np.expm1(-.0001*rates))*(target-initial)
 if max(abs(g.x.get()-expected))>1e-12:raise RuntimeError('Heterogeneous-domain executor')
finally:g.close()
result={'physical_source_and_events_bit_exact':True,'port_domains':3,'mixed_ADD_SET_queries':len(errors),'maximum_reference_difference':max(errors),'heterogeneous_domains_same_executor':True,'scope':'Declared affine cascade and target/rate ODE only; not arbitrary mass, delay, chemical conservation or species performance.'}
# Queries are pure even when time goes backwards (rejected-stage replay).
w=NS(q=np.array([.8]),s=np.array([.2]),tau=np.array([.003]),ts=.005,times=[.00004,.00004],rows=[0,0],jumps=[0.,.1],post_values=[.7,None])
port=ResetFilterPorts([0],[1]);port.update(w)
v1=port.project(cp.zeros(2),cp.asarray([0.,.0001]),1.).get()
port.project(cp.zeros(2),cp.asarray([0.,.00001]),1.).get()
v2=port.project(cp.zeros(2),cp.asarray([0.,.0001]),1.).get()
if v1.tobytes()!=v2.tobytes():raise RuntimeError('Query-order dependent port')
w.post_values=[None,.7];w.jumps=[.1,0.];port.update(w)
v3=port.project(cp.zeros(2),cp.asarray([0.,.0001]),1.).get()
if abs(v3[0]-v1[0])<.09:raise RuntimeError('Lost noncommuting event order')
# Violation of a declared voltage bound rejects and restores the prior state.
target_gpu=cp.asarray([100.]);rates_gpu=cp.asarray([100000.])
def outside(x):return target_gpu,rates_gpu
g=NativeGraph(np.array([-60.]),outside,rtol=1e-5,atol=1e-8,norm_size=1,state_bounds=(-100.,50.))
try:
 try:g.advance(100000,100000,100,100000)
 except RuntimeError as exc:
  if not hasattr(exc,'numerical_diagnostic'):raise
 else:raise RuntimeError('Out-of-domain state accepted')
 if g.x.get()[0]!=-60.:raise RuntimeError('Failed trial not rolled back')
finally:g.close()
result.update(pure_queries_after_backward_replay=True,simultaneous_event_order_preserved=True,model_domain_rejection_rollback=True)
(R/'EVENT_CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
