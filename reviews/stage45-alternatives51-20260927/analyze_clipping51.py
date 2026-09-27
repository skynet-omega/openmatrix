"""Supplement: actual recorded DNg RHS fractions, not inferred whole-brain fractions."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent

def compute(root):
 out={}
 for arm in ['G_odor0','G_odor1','I_odor0','I_odor1']:
  with np.load(root/arm/'candidate_owner.npz') as z,np.load(root/arm/'dng100_observed.npz') as a:
   rows=a['rows'];r=a['records'][:,:128].reshape(-1,4,2,16)
   S=z['S'][rows];q0=z['q0'][rows];g0=z['gE0'][rows]
   X=r[:,:,:,2]+np.maximum(r[:,:,:,4],0);Y=-r[:,:,:,3]+r[:,:,:,5]+np.maximum(-r[:,:,:,4],0)
   E=X/S;I=Y/S;EL=2*q0-g0;total=1+E+I
   raw=(EL+E)/total if arm.startswith('G') else (EL+E*(1-q0)-I*q0+q0)/2
   rate=total/(2*r[:,:,:,9]) if arm.startswith('G') else 1/r[:,:,:,9]
   if not np.isfinite(raw).all() or not np.array_equal(np.clip(raw,0,1),r[:,:,:,7]) or not np.array_equal(rate,r[:,:,:,8]):raise ValueError('Recorded G/I coefficients differ '+arm)
   out[arm]=dict(recorded_DNg_RHS_cells=int(raw.size),fraction_raw_below0=float((raw<0).mean()),fraction_raw_above1=float((raw>1).mean()),max_underflow=float(np.maximum(-raw,0).max()),max_overflow=float(np.maximum(raw-1,0).max()),final_DNg_target_equals_base=bool(np.array_equal(r[:,:,:,7],r[:,:,:,11])),CPU_coefficients_exact=True)
 return dict(DNg100=out,scope='Includes all recorded DNg RHS evaluations, including integration trials. Other rows have only min/max BASE targets; whole-brain per-evaluation clipping frequency was not recorded. Specialized overrides may supersede BASE targets.',interpretation='Any clipping contributes to the bounded engineering dynamics; do not identify an isolated biological conductance effect.')

if __name__=='__main__':
 r=compute(H);(H/'CLIPPING_SUPPLEMENT.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r))
