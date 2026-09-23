"""Deliberate domain violation: preserve offending endpoint and rollback."""
from pathlib import Path
import sys,numpy as np,cupy as cp
T=Path(__file__).resolve().parent;P=T.parents[1]/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(P));from graph_core import NativeGraph
# Exact constant-coefficient flow makes the embedded error small while the
# intended target exceeds the declared physical domain.
def coeff(x):return cp.full_like(x,1.5),cp.full_like(x,1e5)
g=NativeGraph(np.zeros(2),coeff,rtol=1e-5,atol=1e-8,norm_size=2)
try:
 try:g.advance(100000,100000,100,100000)
 except RuntimeError as exc:
  d=exc.numerical_diagnostic
  if d['out_of_domain_count']!=2 or d['sample_indices']!=[0,1] or not all(v>1 for v in d['sample_values']):raise RuntimeError('Missing attempted state')
  if not np.array_equal(g.x.get(),np.zeros(2)):raise RuntimeError('Rollback changed')
 else:raise RuntimeError('Deliberate violation accepted')
finally:g.close()
print('PASS: failing coordinates retained, original rejection and rollback preserved')
