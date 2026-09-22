"""Captured/uncaptured cache state and nonmonotone communicated signals."""
from pathlib import Path
import runpy,json,numpy as np
import cupy as cp
H=Path(__file__).resolve().parent
# Reuse the independent synthetic fixture and its successful equation checks.
v=runpy.run_path(str(H/'test_numerics.py'));e=v['e'];n=e.n;rng=np.random.default_rng(711);spec={'method':'incremental_rk4','h_s':.000125,'quantum':1e-5}
e.reset(spec)
for _ in range(8):e.step(spec)
cp.cuda.get_current_stream().synchronize();plain={k:getattr(e,k).get() for k in ['y','previous','aa','bb']}
e.reset(spec);stream=cp.cuda.Stream(non_blocking=True)
with stream:
 stream.begin_capture()
 for _ in range(2):e.step(spec)
 graph=stream.end_capture()
stream.synchronize()
for _ in range(4):graph.launch(stream=stream)
stream.synchronize();differences={k:float(np.max(abs(a-getattr(e,k).get()))) for k,a in plain.items()}
if max(differences.values())>1e-12:raise ValueError('Capture changes cache state')
e.reset(spec);base=rng.uniform(.2,.8,n);amplitude=rng.uniform(1e-7,1e-3,n);rows=[]
for shift in [0,1,.5,-1,.25,0,1]:
 s=cp.asarray(base+shift*amplitude);e.coefficient(s,1e-5);cp.cuda.get_current_stream().synchronize()
 aa=e.aa.copy();bb=e.bb.copy();e.raw_sums(e.previous);cp.cuda.get_current_stream().synchronize()
 drift=max(float(cp.max(abs(aa-e.aa)).get()),float(cp.max(abs(bb-e.bb)).get()));quant=float(cp.max(abs(s-e.previous)).get())
 if drift>1e-12 or quant>1e-5+1e-15:raise ValueError('Nonmonotone cache invariant fails')
 cp.copyto(e.aa,aa);cp.copyto(e.bb,bb);rows.append({'shift':shift,'sum_drift':drift,'source_quantization_error':quant})
result={'capture_vs_uncaptured':differences,'nonmonotone_updates':rows,'domain':'nonnegative source transmissions, positive scale; crossings into negative release are explicitly outside supported C domain and rejected by Guarded'}
(H/'CACHE_TESTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
