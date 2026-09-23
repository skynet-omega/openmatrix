"""Actual first-accepted inputs; common complete1557-cell initial fixture."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,json,time,gc
import numpy as np,cupy as cp
T=Path(__file__).resolve().parent;sys.path[:0]=[str(T.parent/'causal_runtime_20260922'),str(T.parent/'native_hybrid_20260922'),str(T.parent/'native_hybrid_20260922/vendor')]
from device_cell import DeviceCell
from native_cell import Cell
from event_guard import EventGuardCell
from analyze_boundaries import load,filtered

fixture=load(T/'reference_01/first_cell_fixture.npz');meta=json.loads((T/'reference_01/first_cell_fixture.json').read_text());inputs=load(T/'reference_01/cell_01.npz')
def run(kind,maximum):
 b=NS(n=meta['n'],ports=17,backend='cuda',rest=meta['rest'],elapsed_ns=0,host=cp.asnumpy)
 for k,v in fixture.items():
  if not k.startswith('ax_') and k not in ('ge','gi'):setattr(b,k,cp.asarray(v))
 fields={k[3:]:cp.asarray(v) for k,v in fixture.items() if k.startswith('ax_') and k!='ax_gain'}
 pub=NS(fields=fields,gain=cp.asarray(fixture['ax_gain']),elapsed=0,wrapper=NS(publisher=NS(synaptic_tau=meta['ts'])))
 b._motor_axonal_callback=pub;ledger=[];events=NS(start_elapsed=0,gamma=np.arange(b.n),active=NS(add=lambda t,r,j:ledger.extend(zip(t.tolist(),r.tolist(),j.tolist()))))
 cls={'reference':Cell,'causal':DeviceCell,'guard':EventGuardCell}[kind]
 engine=cls(b,events,**({} if kind=='reference' else {'validate_publisher':False}));start=time.perf_counter()
 engine.advance(b,125000,inputs['ge'],inputs['gi'],inner_step_ns=maximum);wall=time.perf_counter()-start
 arrays={k:b.host(getattr(b,k)).copy() for k in ('delta','gates','q','counts','last_siz','previous_slope','trough','clipped')};arrays['events']=np.asarray(sorted(ledger,key=lambda x:(x[1],x[0])),dtype=float).reshape(-1,3)
 result={'kind':kind,'maximum_ns':maximum,'wall_s':wall,'native_wall_s':engine.report['native_wall_s'],'events':len(ledger),'accepted':engine.report['accepted'],'rejected':engine.report['rejected']}
 name=f'{kind}_{maximum}';np.savez_compressed(T/f'replay_{name}.npz',**arrays)
 if kind=='reference':engine.lib.cell_destroy(engine.handle);engine.handle=None
 del engine,b,pub,fields;gc.collect();return name,result

reports={}
for kind,maximum in [('reference',25000),('causal',25000),('causal',6250),('causal',1562),('guard',25000)]:
 name,result=run(kind,maximum);reports[name]=result;print(json.dumps(result),flush=True)
fine=load(T/'replay_causal_1562.npz');comparison={}
for name in reports:
 a=load(T/f'replay_{name}.npz');events=a['events'];ef=fine['events'];same=events.shape==ef.shape and np.array_equal(events[:,1],ef[:,1])
 comparison[name]={'event_rows_equal_to_refined':bool(same),'max_mark_difference_s':float(np.max(abs(events[:,0]-ef[:,0]),initial=0)) if same else None,'voltage_max_difference':float(np.max(abs(a['delta']-fine['delta']))),'q_max_difference':float(np.max(abs(a['q']-fine['q']))),'counts_equal':np.array_equal(a['counts'],fine['counts'])}
result={'runs':reports,'comparison_to_predeclared1562ns_refinement':comparison,'scope':'One held-input125us actual membrane fixture; no whole-organism speed or convergence certificate.'}
(T/'MEMBRANE_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparison))
