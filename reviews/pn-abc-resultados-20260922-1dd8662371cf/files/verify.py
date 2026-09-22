"""Reconstruct PN state errors and linear residuals; reject criterion/result tampering."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
from scipy.sparse import load_npz
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from portable import read,errors
CONTRACT_SHA='09c118a5b3757f4ebd121377991d5c8858472e736f09a83a213034bddd751d85'

def require(ok,message):
 if not ok:raise ValueError(message)
def verify(root=H,overrides=None):
 overrides={} if overrides is None else overrides
 raw=overrides.get('contract_bytes',(root/'CONTRACT.json').read_bytes());require(hashlib.sha256(raw).hexdigest()==CONTRACT_SHA,'Frozen contract changed');contract=json.loads(raw)
 rows=[]
 for folder in ('compact_01','compiled_01','backend_integration_01'):
  d=root/folder;r=overrides.get(folder,json.loads((d/'RESULT.json').read_text()));state_a=read(d/'replay_parent');state_b=read(d/'replay_candidate')
  if overrides.get('corrupt_state'):state_b['voltage'][0]+=.001
  e=errors(state_a,state_b);passed=True
  for key,value in e.items():
   limit=contract['voltage_abs_mV'] if key=='/voltage' else contract['charge_abs_pC'] if 'charge' in key else contract['probability_abs'] if 'gate' in key else contract['other_numeric_abs']
   passed=passed and value<=limit
  require(e==r['replay']['max_state_errors'],'Recorded state errors differ from arrays');require(passed==r['replay']['within_contract'],'State acceptance flag differs')
  wall=r['replay']['wall_s'];require(all(np.isfinite(t) and t>0 for t in wall.values()),'Invalid timing');speed=wall['parent']/wall['candidate'];require(speed==r['replay']['speedup'],'Speed ratio changed')
  require((speed>=contract['minimum_material_speedup'])==r['material_speedup'],'Speed acceptance flag differs')
  initial=read(root/'capture_01/initial');require(state_a['time_ns']-initial['time_ns']==1000000,'Replay duration changed')
  rows.append({'route':r['route'],'PN_full_step_speedup':speed,'voltage_max_abs_mV':e['/voltage'],'all_state_limits_passed':passed,'classification':'PROMETEDOR_NO_CONFIRMADO' if speed>=2 and passed else 'DESCARTADO_PARA_PROMOCION_PN'})
 G=load_npz(root/'capture_01/G.npz');M=load_npz(root/'capture_01/M.npz');linear=[]
 for i in range(2):
  s=read(root/f'validation_01/linear_{i}');r=s['rhs']-((G+s['shift']*M)@s['candidate']+s['diagonal']*s['candidate']);norm=float(np.linalg.norm(r));error=float(np.max(abs(s['candidate']-s['independent_LU'])))
  require(norm<=s['limit'] and error<=contract['voltage_abs_mV'],'Independent linear accuracy failed');linear.append({'case':i,'full_residual_l2':norm,'max_error_vs_LU':error})
 return {'scope':'PN prescribed1ms; state arrays and two independent linear systems. No complete organism.','rows':rows,'linear_reconstructed':linear,'base_one_second_reference':'Separate result and data, not part of this PN verifier.'}

def main():
 result=verify();corruptions=[]
 for name,over in [('criterion',{'contract_bytes':b'{}'}),('state',{'corrupt_state':True})]:
  try:verify(overrides=over)
  except ValueError:corruptions.append(name)
  else:raise ValueError('Corruption undetected')
 r=json.loads((H/'backend_integration_01/RESULT.json').read_text());r['material_speedup']=not r['material_speedup']
 try:verify(overrides={'backend_integration_01':r})
 except ValueError:corruptions.append('acceptance_flag')
 else:raise ValueError('Corrupted flag accepted')
 result['corruptions_rejected']=corruptions;print(json.dumps(result,indent=2))
if __name__=='__main__':main()
