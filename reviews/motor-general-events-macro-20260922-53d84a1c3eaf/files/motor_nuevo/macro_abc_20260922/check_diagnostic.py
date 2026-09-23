import numpy as np,cupy as cp
from diagnostic_graph import DiagnosticGraph,NativeGraph
for target in (.3,1.5):
 out=[]
 def coeff(x):return cp.full_like(x,target),cp.full_like(x,1e5)
 for cls in (NativeGraph,DiagnosticGraph):
  g=cls(np.zeros(2),coeff,rtol=1e-5,atol=1e-8,norm_size=2)
  try:
   try:g.advance(100000,100000,100,100000)
   except RuntimeError:
    if target<1:raise
    if cls is DiagnosticGraph:
     d=g.failure_arrays()
     if len(g.diagnostic_stages)!=6 or len(g.diagnostic_outputs)!=3 or not np.all(d['out_2_post']>1):raise RuntimeError('Missing stages')
     if not np.array_equal(d['out_2_pre'],d['out_2_post']):raise RuntimeError('Unexpected projection')
   else:
    if target>1:raise RuntimeError('Bad trial accepted')
   out.append(g.x.get())
  finally:g.close()
 if out[0].tobytes()!=out[1].tobytes():raise RuntimeError('Instrumentation changed output/rollback')
print('PASS: successful state and failed rollback bit-identical; all6stages captured')
