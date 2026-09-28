"""Reconstruct the known FP64-to-FP32 cast; no new RHS/CNS measurement."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time
import numpy as np
H=Path(__file__).resolve().parent

def compute(root):
 p=json.loads((root/'PILOT_PLAN.json').read_text());out={};totals={}
 with np.load(root/'parent51_projection.npz') as old:
  for name in p['arms']:
   with np.load(root/name/'neural_and_inputs.npz') as a:
    u=a['JO_drive'];prev=old[name+'__JO_drive'];u32=u.astype(np.float32);prev32=prev.astype(np.float32)
    total=u32.astype(np.float64).sum(axis=1);totals[name]=total
    out[name]={'max_body_mm_s':float(np.abs(a['air_kinematics'][:,3:6]).max()),
     'max_FP64_JO_delta':float(np.abs(u-prev).max()),'changed_FP64_values':int(np.count_nonzero(u!=prev)),
     'changed_FP32_values':int(np.count_nonzero(u32!=prev32)),
     'max_FP32_JO_delta':float(np.max(np.abs(u32.astype(np.float64)-prev32.astype(np.float64)))),
     'mean_JO_sum_FP32_values_accumulated_in_FP64':float(total[50:90].mean()),
     'final_release_all_exact_vs51':bool(np.array_equal(a['final_q'],old[name+'__final_q']))}
  L=old['airL_odor0__JO_drive'][10:];R=old['airR_odor0__JO_drive'][10:]
  Q=min(L[0].sum(),R[0].sum());a=[(v*(Q/v.sum(axis=1))[:,None]).astype(np.float32).astype(np.float64) for v in [L,R]]
 ratios={str(o):float(totals[f'airR_odor{o}'][50:90].mean()/totals[f'airL_odor{o}'][50:90].mean()-1) for o in [0,1]}
 return {'schema':'repair52_reconstructed_fp32_boundary_v1','arms':out,'postcast_R_over_L_minus1':ratios,
  'L1_screen51_after_cast':{'common_precast_total':float(Q),'max_pair_sum_difference':float(np.max(np.abs(a[0].sum(axis=1)-a[1].sum(axis=1)))),
    'max_abs_target_residual':float(max(np.max(np.abs(v.sum(axis=1)-Q)) for v in a))},
  'scope':'Deterministic CPU reconstruction of the added JO component after the frozen FastCSR cast; not a new live JO-RHS probe. Actual legacy native JO drive is zero in this preparation; the existing witness audits addition before cast. Future designs must recheck that owner and capture/check postcast values.',
  'sum_definition':'Promote individual FP32 components toFP64 before summing; do not conflate FP32 reduction error with the operator input.'}

if __name__=='__main__':
 start=time.process_time();r=compute(H);r['CPU_s']=time.process_time()-start
 (H/'FP32_BOUNDARY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
