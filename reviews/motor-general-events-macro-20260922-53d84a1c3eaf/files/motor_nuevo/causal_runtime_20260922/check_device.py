"""Saved actual membrane states; independent grouping and failed-epoch controls."""
from pathlib import Path
import sys,json,copy
from types import SimpleNamespace as NS
import numpy as np,cupy as cp
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'),str(HERE.parent/'native_hybrid_20260922/vendor')]
from verify_transport import read_state
from device_cell import DeviceCell
s=read_state(HERE.parent/'native_hybrid_20260922/baseline_01/brain_final');c=s['kc_spatial_manifest']['cell']

def fixture(ids):
 n=len(ids);b=NS(n=n,ports=17,backend='cuda',rest=c['rest_mV'],elapsed_ns=0,host=cp.asnumpy)
 for dst,src in [('C','C_nF'),('G','G_nS'),('chanG','channel_G_nS'),('chanb','channel_b_nS'),('shuntG','shunt_G'),('shuntb','shunt_b'),('obs','observation_basis')]:setattr(b,dst,cp.asarray(c[src]))
 b.ena=cp.asarray(np.tile(np.array([60.,60.,-80.])-b.rest,17));b.caps=cp.full(n,100.);b.tau=cp.full(n,.03)
 for k,v in s['kc_spatial_state'].items():
  if isinstance(v,np.ndarray):setattr(b,k,cp.asarray(v[ids]))
 fields={k:cp.asarray(v[ids]) for k,v in s['kc_axonal_state'].items() if isinstance(v,np.ndarray)}
 publisher=NS(fields=fields,gain=cp.ones(n),elapsed=0,wrapper=NS(publisher=NS(synaptic_tau=.003)))
 b._motor_axonal_callback=publisher;ledger=[]
 events=NS(start_elapsed=0,gamma=np.arange(n),active=NS(add=lambda t,r,j:ledger.extend(zip(t.tolist(),r.tolist(),j.tolist()))))
 engine=DeviceCell(b,events,validate_publisher=False)
 return b,engine,ledger

def run(ids,fail=False):
 b,e,ledger=fixture(ids);ge=np.array([[.01*(1+int(i)%7)]*4 for i in ids]);gi=ge*.37
 before={k:v.get().copy() for k,v in e.state.items()}
 if fail:
  current=np.zeros((b.n,17));current[0]=1e300
  try:e.advance(b,125000,ge,gi,current_pA=current)
  except RuntimeError:pass
  else:raise RuntimeError('Injected overflowing finite input was not rejected')
  if ledger or b.elapsed_ns or b._motor_axonal_callback.elapsed:raise RuntimeError('Failed epoch published events/clock')
  for k,v in before.items():
   if not np.array_equal(getattr(b,k).get(),v):raise RuntimeError('Failed epoch mutated owner '+k)
 e.advance(b,125000,ge,gi)
 state={k:getattr(b,k).get() for k in before}
 state.update({'axon_'+k:v.get() for k,v in b._motor_axonal_callback.fields.items()})
 return {int(i):{k:v[n].copy() for k,v in state.items()} for n,i in enumerate(ids)},sorted((float(t),int(ids[int(r)]),float(j)) for t,r,j in ledger),e.counts.get()

ids=[0,500,1000,1556];base,events,counts=run(ids);permuted,permuted_events,_=run(ids[::-1]);single,single_events,_=run([500]);retry,retry_events,_=run(ids,True)
if events!=permuted_events or events!=retry_events or [e for e in events if e[1]==500]!=single_events:raise RuntimeError('Event timing depends on batch/retry')
for label,other in [('permutation',permuted),('single',single),('retry',retry)]:
 for i,fields in other.items():
  for k,v in fields.items():
   if not np.array_equal(v,base[i][k]):raise RuntimeError(f'{label} changed cell {i}/{k}')
# A detector counterexample: prescribed waveform, not a simulated membrane.
def peaks(sequence):
 last=-65.;slope=0.;trough=-65.;events=[]
 for t,v in sequence:
  delta=v-last;peak=slope>0 and delta<=0
  if peak and last>-40 and last-trough>=20:events.append(t)
  trough=v if peak else min(trough,v);last=v;slope=delta
 return events
fine=peaks([(6.25,-30),(12.5,-65),(25.,-65)])
coarse=peaks([(12.5,-65),(25.,-65)])
if fine!=[12.5] or coarse!=[]:raise RuntimeError('Peak counterexample invalid')
report={'status':'PASS','event_sequences_exact_under_grouping_and_retry':True,'permutation_exact':True,'single_vs_batch_exact':True,'failure_no_publication':True,'retry_exact':True,'actual_geometry_and_initial_states':True,'held_inputs_and_filter_parameters':'declared fixture, not full-organism reproduction','counts':counts.tolist(),'detector_counterexample':{'fine_event_us':fine,'coarse_event_us':coarse,'implication':'Local voltage tolerances do not certify spike-time topology; keep exact count controls and do not claim universal event fidelity.'}}
(HERE/'DEVICE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
