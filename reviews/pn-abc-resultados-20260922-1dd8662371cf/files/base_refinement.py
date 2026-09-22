"""Complete one-second refined base references; no additional model component."""
from pathlib import Path
import sys,time,json
import numpy as np
import cupy as cp
H=Path(__file__).resolve().parent;B=H.parent/'abc_20260922';sys.path.insert(0,str(B))
from safe_engine import SafeEngine
from recurrence import write,digest
fixture=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/runs/motor14_20260922/architecture_input/connections.npz')
z=np.load(fixture,allow_pickle=False);d={k:z[k] for k in z.files};z=np.load(B/'initial.npz',allow_pickle=False);initial=np.r_[z['q'],z['s']]
out=H/'base_refinement';out.mkdir(exist_ok=False);e=SafeEngine(d,initial);traces=[];rows=[]
for h in [.000125,.0000625]:
 spec={'method':'rk4','h_s':h};e.reset(spec);stream=cp.cuda.Stream(non_blocking=True)
 with stream:
  stream.begin_capture()
  for _ in range(round(.001/h)):e.step(spec)
  graph=stream.end_capture()
 stream.synchronize();states=[initial];times=[0.];started=time.perf_counter()
 for i in range(1000):
  if i in (1,3):e.d['photo'][:]=e.d['visual']*(.2 if i==1 else 0.);cp.cuda.get_current_stream().synchronize()
  graph.launch(stream=stream);stream.synchronize()
  if i<5 or (i+1)%10==0:
   e.require_valid();states.append(cp.asnumpy(e.y));times.append((i+1)*.001)
  if time.perf_counter()-started>300:raise TimeoutError('Fixed reference budget exhausted')
 wall=time.perf_counter()-started;state=np.asarray(states);traces.append(state);np.savez_compressed(out/f'reference_{round(h*1e9)}.npz',state=state,time_s=times)
 rows.append({'h_s':h,'simulated_s':1,'wall_s':wall,'intermediate_flags':int(e.flags.get()[0])});print(json.dumps(rows[-1]),flush=True)
original=np.load(B/'long_01/states.npz',allow_pickle=False)['state'];refinement=float(np.max(abs(traces[1]-traces[0])));error=float(np.max(abs(original-traces[1])))
write(out/'RESULT.json',{'scope':'One-second base only, sampled106times','rows':rows,'reference_refinement_max_qs':refinement,'B250_max_qs_error':error,'predeclared_reference_limit':1e-5,'candidate_limit':1e-4,'sampled_error_pass':bool(refinement<=1e-5 and error<=1e-4),'original_sha256':digest(B/'long_01/states.npz')})
