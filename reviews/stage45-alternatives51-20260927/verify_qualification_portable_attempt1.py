"""CPU replay of qualification identities and equations from the compact capsule."""
from pathlib import Path
import json
import numpy as np
import verify_qualification as original

def need(ok,message):
 if not ok:raise ValueError(message)

def verify(root):
 original.H=root
 with np.load(root/'qualification_projection.npz') as z:
  initial=z['initial_native_q'].copy();reference={k[4:]:z[k].copy() for k in z.files if k.startswith('ref_')}
 with np.load(root/'qual_identity/traces.npz') as z:
  need(set(z.files)==set(reference),'Identity trace schema')
  need(all(np.array_equal(z[k],v) for k,v in reference.items()),'All33 qualification fields versus49')
 reports={};total=0
 for arm in ['identity','air','conductance','current']:
  d=root/('qual_'+arm);r=json.loads((d/'RESULT.json').read_text())
  need(r['status']=='COMPLETE' and r['initial_exact'] and r['committed_ms']==4,'Incomplete qualification')
  total+=r['attempted_ms']
  with np.load(d/'neural_and_inputs.npz') as z:
   need(np.array_equal(initial,z['initial_q']),'Initial qualification native vector')
   need(all(np.isfinite(z[k]).all() for k in z.files),'Nonfinite qualification')
   need(np.array_equal(z['JO_drive'][:2],np.zeros((2,335))),'Qualification air prefix')
   if arm=='identity':need(np.array_equal(z['JO_drive'],np.zeros((4,335))),'Identity extra air drive')
   if arm=='air':need(z['JO_drive'][2:].max()>0,'Air positive control')
  if arm in ['conductance','current']:reports[arm]=original.candidate_check(arm,initial)
 with np.load(root/'qual_conductance/candidate_owner.npz') as g,np.load(root/'qual_current/candidate_owner.npz') as i:
  need(all(np.array_equal(g[k],i[k]) for k in ['q0','S','gE0']),'Qualification G/I initial ownership')
 need(total==16,'Qualification interaction budget')
 return dict(status='PASS',all33_identity_fields_exact=True,initial_native_q_all_rows_exact=True,G_I_adoption_all_rows_exact=True,neural_ms=total,coefficient_checks=reports,scope='CPU verification of packaged original qualification recordings; not a new GPU restart')

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
