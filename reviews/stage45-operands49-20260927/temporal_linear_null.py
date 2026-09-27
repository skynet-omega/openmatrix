"""Counterexample: order-odd does not alone demonstrate motion computation."""
import json,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def response(x):
 # Two independent fixed causal filters. No cross-side terms or learning.
 y=np.zeros_like(x);a=np.exp(-1/np.array([18.,25.]))
 for t in range(1,len(x)):y[t]=a*y[t-1]+(1-a)*x[t]
 return y
left_first=np.zeros((200,2));left_first[10:30,0]=1;left_first[50:70,1]=1
right_first=left_first[:,::-1].copy()
a=response(left_first);b=response(right_first)
odd=.5*((a[:,0]-a[:,1])-(b[:,0]-b[:,1]))
if not(np.max(np.abs(odd))>.5):raise ValueError('Counterexample failed')
# Superposition residual explicitly remains zero for independent linear filters.
x=left_first.copy();x[:,1]=0;y=left_first.copy();y[:,0]=0
residual=a-response(x)-response(y)
if np.any(residual):raise ValueError('Filter is not the declared null')
out=dict(schema='temporal49_null_v1',max_abs_order_odd=float(np.max(np.abs(odd))),max_abs_superposition_residual=float(np.max(np.abs(residual))),integrated_input_by_side=left_first.sum(axis=0).tolist(),interpretation='Nonzero O_X is insufficient: the independent linear null produces it. A prospective interaction test needs isolated-side responses or another valid null, not just mirrored sequences.',neural_ms=0)
p=H/'TEMPORAL_LINEAR_NULL.json'
if p.exists():raise ValueError('Preserve result')
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
