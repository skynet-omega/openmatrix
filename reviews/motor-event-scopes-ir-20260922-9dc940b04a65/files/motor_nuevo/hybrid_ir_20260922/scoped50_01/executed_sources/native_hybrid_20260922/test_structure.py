"""Reproduce external reviewer counterexamples using real saved state."""
from pathlib import Path
import sys,copy,shutil,tempfile,json
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'
sys.path[:0]=[str(P),str(HERE/'vendor') if (HERE/'vendor').exists() else '/home/daroch/AXIOMA_FLYWIRE/matrix/src']
from verify_transport import verify,read_state,flatten,history_difference
from session_io import write_state
ref=P/'transport_resident_left_01';state=read_state(ref/'brain_final');checks=[]
def setpath(obj,path,value):
 keys=path.strip('/').split('/')
 for key in keys[:-1]:obj=obj[key]
 obj[keys[-1]]=value
with tempfile.TemporaryDirectory(dir=HERE) as td:
 out=Path(td)
 for name in ('traces.npz','body_final.npz'):shutil.copy2(ref/name,out/name)
 cases=[]
 for kind in ('shape','dtype'):
  x=copy.deepcopy(state);h=x['pn_online_state']['receptors']['GABA']['history']
  for seg in h:
   for k in ('left','right'):
    seg[k]=np.repeat(seg[k][...,None],2,axis=-1) if kind=='shape' else seg[k].astype(np.float32)
  cases.append(('history_'+kind,x))
 for label,predicate in [('nested_clock',lambda p,v:p.startswith('/pn_online_state/') and p.endswith('/time_ns')),('nested_identity',lambda p,v:p.startswith('/pn_online_state/') and p.endswith('/identity'))]:
  path,value=next((p,v) for p,v in flatten(state) if predicate(p,v));x=copy.deepcopy(state);setpath(x,path,value+1 if type(value) is int else value+'_corrupt');cases.append((label,x))
 cases.append(('trace_clock',copy.deepcopy(state)))
 for name,x in cases:
  shutil.copy2(ref/'traces.npz',out/'traces.npz')
  if name=='trace_clock':
   with np.load(out/'traces.npz') as z:trace={k:z[k].copy() for k in z.files}
   trace['PN_time_ns'][0]+=1;np.savez(out/'traces.npz',**trace)
  write_state(out/'brain_final',x)
  try:detected=not verify(ref,out)['screen_pass']
  except (ValueError,TypeError,KeyError):detected=True
  if not detected:raise RuntimeError('Undetected '+name)
  checks.append(dict(case=name,detected=True))
 if not verify(ref,ref)['screen_pass']:raise RuntimeError('Identical state rejected')
# Legitimate repartition is preserved.
a={'time_ns':0,'identity':'channels','history':[dict(start_ns=0,end_ns=10,left=np.array([0.]),right=np.array([1.]))]}
b=copy.deepcopy(a);b['history']=[dict(start_ns=0,end_ns=5,left=np.array([0.]),right=np.array([.5])),dict(start_ns=5,end_ns=10,left=np.array([.5]),right=np.array([1.]))]
if history_difference(a,b)['max_abs']!=0:raise RuntimeError('Valid repartition rejected')
result=dict(checks=checks,identical_pass=True,repartition_pass=True,python_optimized=not __debug__)
(HERE/'STRUCTURE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
