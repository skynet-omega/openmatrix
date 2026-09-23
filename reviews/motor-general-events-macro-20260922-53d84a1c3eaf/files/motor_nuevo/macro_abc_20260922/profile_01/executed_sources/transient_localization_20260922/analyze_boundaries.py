"""Reconstruct filter residuals and compare physical boundary histories."""
from pathlib import Path
import sys,json
import numpy as np
T=Path(__file__).resolve().parent;ROOT=T.parents[1]
sys.path.insert(0,str(ROOT/'campanas/etapa3_motor_nuevo_20260922'))
from verify_transport import leaves,read_state,flatten

def changed(a,b):
 out=[]
 for p in sorted(set(a)|set(b)):
  x,y=a.get(p),b.get(p);eq=type(x) is type(y)
  if eq:eq=(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()) if isinstance(x,np.ndarray) else x==y
  if not eq:out.append(p)
 return out
def conv(t,tq,ts):
 z=t*(1/ts-1/tq);den=1-ts/tq
 # t>=0; masked division retains the equal-constant limit.
 out=np.array(t/ts*np.exp(-t/ts)*(1+z/2+z*z/6+z*z*z/24),dtype=float)
 mask=abs(z)>=1e-5
 np.divide(np.where(z>=0,np.exp(-t/tq)*(-np.expm1(-np.maximum(z,0))),np.exp(-t/ts)*np.expm1(np.minimum(z,0))),den,out=out,where=mask)
 return out
def filtered(z,ns,times=None,jumps=None):
 t=ns*1e-9;tq=z['tau'];ts=float(z['ts']);q=z['q0']*np.exp(-t/tq);s=z['s0']*np.exp(-t/ts)+z['q0']*conv(t,tq,ts)
 times=z['times'] if times is None else times;jumps=z['jumps'] if jumps is None else jumps
 for time,row,jump in zip(times,z['rows'],jumps):
  u=t-time
  if u>=0:
   q[row]+=jump*np.exp(-u/tq[row]);s[row]+=jump*conv(np.asarray(u),np.asarray(tq[row]),ts)
 return q,s
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
 a,b=T/'reference_01',T/'causal_01';ma=json.loads((a/'MAPPING.json').read_text());mb=json.loads((b/'MAPPING.json').read_text())
 if ma!=mb:raise ValueError('Different row mapping')
 rows=np.asarray(ma['event_rows']);sr=177758+rows;focus=np.flatnonzero(rows==57073).item()
 initial=changed(dict(flatten(read_state(a/'initial_brain'))),dict(flatten(read_state(b/'initial_brain'))))
 instrumentation={}
 for name,older in [('reference_01','smoke_reference_01'),('causal_01','smoke_causal_01')]:
  diffs=changed(leaves(T/name),leaves(T.parent/'pipeline_review_20260922'/older))
  instrumentation[name]=diffs
  if diffs:raise ValueError('Instrumentation changed endpoint: '+str(diffs))
 epochs=json.loads((a/'BOUNDARIES.json').read_text());other=json.loads((b/'BOUNDARIES.json').read_text());records=[]
 for meta in epochs:
  i=meta['index'];x=load(a/f'boundary_{i:02d}.npz');y=load(b/f'boundary_{i:02d}.npz')
  if meta['ns']!=other[i]['ns'] or meta['clock_start_ns']!=other[i]['clock_start_ns']:raise ValueError('Clocks differ')
  def ordered(z):
   order=np.lexsort((z['times'],z['rows']));return z['rows'][order],z['times'][order],z['jumps'][order]
  xr,xt,xj=ordered(x);yr,yt,yj=ordered(y);same=np.array_equal(xr,yr)
  qx,sx=filtered(x,meta['ns']);qy,sy=filtered(y,meta['ns'])
  cx=load(a/f'cell_{i:02d}.npz');cy=load(b/f'cell_{i:02d}.npz')
  # Counterfactual swaps only event times, keeping the candidate inputs/jumps.
  swapped=None
  if same:
   order=np.lexsort((y['times'],y['rows']));times=y['times'].copy();times[order]=xt
   _,st=filtered(y,meta['ns'],times=times)
   swapped=float(np.max(abs(sx-st),initial=0))
  records.append({**meta,'before_state_max':float(np.max(abs(x['before']-y['before']))),
   'after_state_max':float(np.max(abs(x['after']-y['after']))),
   'ge_max':float(np.max(abs(cx['ge']-cy['ge']))),'gi_max':float(np.max(abs(cx['gi']-cy['gi']))),
   'event_rows_and_counts_equal':same,'event_time_max_s':float(np.max(abs(xt-yt),initial=0)) if same else None,'event_jump_max':float(np.max(abs(xj-yj),initial=0)) if same else None,
   'q0_max':float(np.max(abs(x['q0']-y['q0']))),'s0_max':float(np.max(abs(x['s0']-y['s0']))),
   'ref_filter_q_residual':float(np.max(abs(qx-x['after'][rows]))),'ref_filter_s_residual':float(np.max(abs(sx-x['after'][sr]))),
   'causal_filter_q_residual':float(np.max(abs(qy-y['after'][rows]))),'causal_filter_s_residual':float(np.max(abs(sy-y['after'][sr]))),
   'filter_s_difference':float(np.max(abs(sx-sy))), 'only_times_swapped_filter_s_difference':swapped,
   'focus_s_difference':float(sx[focus]-sy[focus]),'focus_event_times_reference':x['times'][x['rows']==focus].tolist(),'focus_event_times_causal':y['times'][y['rows']==focus].tolist()})
 result={'initial_changed_leaves':initial,'operators_equal':json.loads((a/'OPERATOR.json').read_text())==json.loads((b/'OPERATOR.json').read_text()),'instrumentation_changed_leaves':instrumentation,'epochs':records,'scope':'Exact endpoint controls, actual marks and analytical filter reconstruction. Time-swap is an offline counterfactual, not an organism run or proof of all later causal mediation.'}
 (T/'LOCALIZATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'initial_changed':len(initial),'instrumentation_exact':True,'epochs':[{k:r[k] for k in ('index','phase','after_state_max','event_time_max_s','filter_s_difference','only_times_swapped_filter_s_difference','focus_s_difference')} for r in records]},indent=2))

if __name__=='__main__':main()
