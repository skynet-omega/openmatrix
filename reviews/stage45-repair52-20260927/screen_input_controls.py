"""One finite recorded-input screen; no CNS or neural-response fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,resource
import numpy as np
H=Path(__file__).resolve().parent

def need(ok,m):
 if not ok:raise ValueError(m)

def main():
 cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(5,6))
 with np.load(H/'donors/JO_anatomy_arrays.npz') as z:types=z['source_type'];sides=z['source_side']
 with np.load(H/'parent51_projection.npz') as z:L=z['airL_odor0__JO_drive'][10:];R=z['airR_odor0__JO_drive'][10:]
 permutation=np.arange(335)
 for typ in np.unique(types):
  for side in ['L','R']:
   rows=np.flatnonzero((types==typ)&(sides==side));permutation[rows]=rows[::-1]
 need(np.unique(permutation).size==335 and np.any(permutation!=np.arange(335)),'Nontrivial identity permutation')
 q=min(L[0].sum(),R[0].sum());q2=min(np.linalg.norm(L[0]),np.linalg.norm(R[0]))
 a=[u*(q/u.sum(axis=1))[:,None] for u in [L,R]]
 b=[u*(q2/np.linalg.norm(u,axis=1))[:,None] for u in [L,R]]
 for original,scaled in zip([L,R],a):need(np.array_equal(original>0,scaled>0),'L1 control support change')
 out={'scope':'CPU transform of exposed51 recorded JO drives, not corrected52 or a CNS experiment',
  'A':{'common_L1':float(q),'L1_max_residual':float(max(np.abs(v.sum(axis=1)-q).max() for v in a)),
       'L2_ratio_R_over_L_mean':float(np.mean(np.linalg.norm(a[1],axis=1)/np.linalg.norm(a[0],axis=1))),
       'support_unchanged':True,'active_L':int((L[0]>0).sum()),'active_R':int((R[0]>0).sum()),
       'scaling_min':float(min((q/v.sum(axis=1)).min() for v in [L,R])),
       'scaling_max':float(max((q/v.sum(axis=1)).max() for v in [L,R]))},
  'B':{'common_L2':float(q2),'L1_ratio_R_over_L_mean':float(np.mean(b[1].sum(axis=1)/b[0].sum(axis=1)))},
  'C':{'permuted_id_positions':int((permutation!=np.arange(335)).sum()),
       'changed_input_values':int(sum(np.count_nonzero(v[:,permutation]!=v) for v in [L,R])),
       'max_abs_input_change':float(max(np.abs(v[:,permutation]-v).max() for v in [L,R])),
       'all160_vectors_identical':bool(all(np.array_equal(v[:,permutation],v) for v in [L,R]))},
  'new_CNS_ms':0,'CPU_s':time.process_time()-cpu,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 (H/'INPUT_CONTROL_SCREEN.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
