"""Preserve the historical partial screen; refuse an unsupported full-state PASS.

No new error limits are invented here. An unbounded changed physical field needs
a prospective criterion or an exact same-method control, not a silent default.
"""
from pathlib import Path
import sys,json
import numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1]/'campanas/etapa3_motor_nuevo_20260922';sys.path.insert(0,str(P))
from verify_transport import verify
from verification_vendor.compare import compare,leaves

def qualify(reference,candidate):
 historical=verify(reference,candidate);raw=compare(reference,candidate);a,b=leaves(reference),leaves(candidate);uncovered={}
 for path,error in raw['all_numeric_errors'].items():
  if error==0:continue
  if '/statistics/' in path or path in ('/brain/next_step_ns','/brain/kc_apl_dynamic_state/coupling_steps'):continue
  if '/history/' in path:continue  # validated as functions by the historical wrapper
  covered=path in ('/brain/state','/traces/yaw_delta_deg') or path.endswith(('/gates','/voltage_delta_mV'))
  if path=='/body_final/qpos':
   error=float(np.max(abs(a[path][3:]-b[path][3:]),initial=0))
   if error==0:continue
  if not covered:uncovered[path]={'max_abs':float(error),'criterion':'UNDECLARED'}
 qualified=historical['screen_pass'] and not uncovered
 return {'historical_partial_screen':historical,'uncovered_changed_numeric_fields':uncovered,
         'full_state_qualified':qualified,'status':'QUALIFIED_FOR_DECLARED_SCOPE' if qualified else 'NOT_QUALIFIED',
         'scope':'No full-state scientific admission while changed fields lack declared bounds. Historical short-screen limits and result are preserved; no biological equivalence or long-time guarantee.'}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--out',required=True);a=p.parse_args()
 result=qualify(a.reference,a.candidate);Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'historical_partial_pass':result['historical_partial_screen']['screen_pass'],'full_state_qualified':result['full_state_qualified'],'uncovered_fields':len(result['uncovered_changed_numeric_fields'])}))
