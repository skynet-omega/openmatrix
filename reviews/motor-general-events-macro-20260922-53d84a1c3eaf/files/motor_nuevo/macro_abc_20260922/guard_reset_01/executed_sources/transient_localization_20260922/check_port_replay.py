"""Independent two-state matrix-exponential replay of recorded event histories."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,json
import numpy as np
from scipy.linalg import expm
T=Path(__file__).resolve().parent;sys.path.insert(0,str(T.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
import cupy as cp
from event_ports import FilterPorts
from analyze_boundaries import load

def reference(initial,tq,ts,events,t):
 matrix=np.array([[-1/tq,0.],[1/ts,-1/ts]])
 state=np.asarray(initial,dtype=float).copy();last=0.
 for at,jump in sorted(events):
  if at>t:break
  state=expm(matrix*(at-last))@state;state[0]+=jump;last=at
 return expm(matrix*(t-last))@state

mapping=json.loads((T/'reference_01/MAPPING.json').read_text());rows=np.asarray(mapping['event_rows']);slot=int(np.flatnonzero(rows==57073)[0]);export={'row':57073,'node_id':75907,'state_coordinate':234831,'transmission_start':177758,'event_port_slot':slot,'spatial_cell_slot':mapping['target_cell_slot'],'owner':'spatial SIZ sampled-peak source; FilterPorts owns projected q/s in CNS','histories':{}};maximum=0.;queries=0
for label in ('reference_01','causal_01'):
 epochs=json.loads((T/label/'BOUNDARIES.json').read_text());out=[]
 for e in epochs:
  z=load(T/label/f"boundary_{e['index']:02d}.npz");events=list(zip(z['times'][z['rows']==slot].tolist(),z['jumps'][z['rows']==slot].tolist()))
  duration=e['ns']*1e-9;w=NS(q=z['q0'][[slot]],s=z['s0'][[slot]],tau=z['tau'][[slot]],ts=float(z['ts']),times=[x[0] for x in events],rows=[0]*len(events),jumps=[x[1] for x in events])
  port=FilterPorts([0],[1]);port.update(w);clock=cp.zeros(2);initial=(float(w.q[0]),float(w.s[0]));tq=float(w.tau[0]);ts=w.ts
  points=sorted({0.,duration,duration*.25,duration*.5,duration*.75}|{p for at,_ in events for p in (max(0.,at-1e-12),at,min(duration,at+1e-12))})
  for t in points:
   clock.set(np.array([0.,t]));actual=port.project(cp.zeros(2),clock,1.).get();expected=reference(initial,tq,ts,events,t);error=float(max(abs(expected-actual)))
   maximum=max(maximum,error);queries+=1
   if not np.isfinite(actual).all() or error>1e-12:raise ValueError('Recorded port differs from matrix exponential')
  final=reference(initial,tq,ts,events,duration);saved=z['after'][[57073,234831]]
  if float(max(abs(final-saved)))>1e-12:raise ValueError('Published boundary differs from independent port')
  out.append({**e,'q0':initial[0],'s0':initial[1],'tau_q':tq,'tau_s':ts,'events':[{'relative_s':at,'jump_after_clipping':jump} for at,jump in events],'query_relative_s':duration,'projected_q':float(saved[0]),'projected_s':float(saved[1]),'independent_q':float(final[0]),'independent_s':float(final[1])})
 export['histories'][label]=out
(T/'PORT_75907.json').write_text(json.dumps(export,indent=2)+'\n')
result={'status':'PASS','queries':queries,'maximum_absolute_residual':maximum,'tolerance':1e-12,'both_histories_replayed_in_same_CUDA_port':True,'independent_reference':'SciPy expm of triangular2x2 dynamics between recorded post-clipping jumps','claim':'Same histories produce the specified filter response; does not validate source event-time accuracy or biological equivalence.'}
(T/'PORT_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
