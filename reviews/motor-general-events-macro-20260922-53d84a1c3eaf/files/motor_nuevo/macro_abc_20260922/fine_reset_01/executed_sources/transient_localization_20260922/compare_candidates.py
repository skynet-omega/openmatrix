"""Reconstruct candidate comparisons without trusting emitted PASS fields."""
from pathlib import Path
import sys,json
import numpy as np
T=Path(__file__).resolve().parent;H=T.parent/'pipeline_review_20260922'
sys.path.insert(0,str(H));from qualify_transport import qualify
from verify_transport import read_state,flatten,histories,history_difference

def brain_comparison(a,b):
 x,y=read_state(a),read_state(b);xx,yy=dict(flatten(x)),dict(flatten(y));errors={};discrete=[];layout=[];nonfinite=[]
 for p in sorted(set(xx)|set(yy)):
  if '/history/' in p:continue
  if p not in xx or p not in yy:layout.append(p);continue
  v,w=xx[p],yy[p]
  if type(v)!=type(w):layout.append(p);continue
  if isinstance(v,np.ndarray):
   if v.shape!=w.shape or v.dtype!=w.dtype:layout.append(p);continue
   if v.dtype.kind in 'fc':
    if not np.isfinite(v).all() or not np.isfinite(w).all():nonfinite.append(p)
    errors[p]=float(np.max(abs(v-w),initial=0))
   elif v.tobytes()!=w.tobytes():discrete.append(p)
  elif type(v)is float:
   if not np.isfinite([v,w]).all():nonfinite.append(p)
   errors[p]=abs(v-w)
  elif v!=w and '/statistics/' not in p and p not in ('/next_step_ns','/kc_apl_dynamic_state/coupling_steps'):discrete.append(p)
 hx,hy=dict(histories(x)),dict(histories(y))
 if set(hx)!=set(hy):raise ValueError('History owner mismatch')
 hd={p:history_difference(hx[p],hy[p]) for p in hx}
 return {'normalized_state_max_abs':errors['/state'],'gate_max_abs':max(v for k,v in errors.items() if k.endswith('/gates')),'voltage_max_abs_mV':max(v for k,v in errors.items() if k.endswith(('/delta','/voltage_delta_mV'))),'changed_discrete_fields':discrete,'layout_differences':layout,'nonfinite':nonfinite,'histories':hd,'all_numeric_errors':errors,'scope':'Brain snapshots only; no body at1/5ms and no claim of intermediate event-count agreement. All numeric fields reported, no new blanket tolerance.'}

def main():
 a,b=T/'fine1562_20',T/'guard20_02'
 if json.loads((b/'RESULT.json').read_text())['status']!='COMPLETE':raise ValueError('Incomplete candidate')
 fine_receipt=json.loads((a/'RESULT.json').read_text())
 snapshots={str(i):brain_comparison(a/f'brain_{i:02d}ms',b/f'brain_{i:02d}ms') for i in (1,5,20) if (a/f'brain_{i:02d}ms.npz').exists()}
 endpoint=qualify(a,b) if fine_receipt['status']=='COMPLETE' else None
 old_endpoint=qualify(T.parent/'causal_runtime_20260922/reference20_01',b)
 perf={}
 for d in (a,b):
  r=json.loads((d/'RESULT.json').read_text());times=r['times'];runtime=r['runtime']
  perf[d.name]={'ms':len(times),'wall_s':r['wall_s'],'advance_s':sum(v['advance_s'] for v in times),'observation_s':sum(v['observation_s'] for v in times),'mean_advance_s_per_simulated_ms':sum(v['advance_s'] for v in times)/len(times),'membrane_native_wall_s':runtime['cell']['native_wall_s'],'CNS':runtime['CNS'],'PN':runtime['PN']}
 result={'brain_snapshots':snapshots,'twenty_ms_against_fine':endpoint,'fine_receipt_status':fine_receipt['status'],'fine_failure':fine_receipt['error'],'twenty_ms_against_old_reference':old_endpoint,'performance':perf,'fine_one_ms_repeat':brain_comparison(T/'fine1562_01/brain_final',a/'brain_01ms'),'stage3_admission':False,'scope':'Bounded sham comparison. Missing refined20ms endpoint remains unavailable; old reference is not ground truth. Brain-only checkpoints1/5ms do not validate the entire trajectory. No stage3 behavioral experiment.'}
 (T/'CANDIDATES.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'snapshots':{k:{f:v[f] for f in ('normalized_state_max_abs','gate_max_abs','voltage_max_abs_mV','changed_discrete_fields')} for k,v in snapshots.items()},'refined_endpoint_available':endpoint is not None,'old_reference_endpoint_screen':old_endpoint['historical_partial_screen']['screen_pass'],'uncovered_fields':len(old_endpoint['uncovered_changed_numeric_fields']),'performance':perf},indent=2))
if __name__=='__main__':main()
