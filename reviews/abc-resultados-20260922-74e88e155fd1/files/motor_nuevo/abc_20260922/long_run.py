"""One preauthorized long base-only run; explicitly no one-second accuracy claim."""
from pathlib import Path
import sys,time,json,resource
sys.path.insert(0,str(Path(__file__).resolve().parent))
from recurrence import Engine,digest,write
from verify import verify
import numpy as np
import cupy as cp
H=Path(__file__).resolve().parent
fixture=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
c=json.loads((H/'contract.json').read_text());v=verify(H/'runs_01');tests=json.loads((H/'INDEPENDENT_TESTS.json').read_text())
if not tests['A_B_A_exact']:raise ValueError('Invalidation test failed')
spec=next(s for s in c['candidates'] if s['id']==v['selected_base_only'])
if digest(fixture)!=c['fixture_sha256']:raise ValueError('Fixture mismatch')
z=np.load(fixture,allow_pickle=False);d={k:z[k] for k in z.files};z=np.load(H/'initial.npz',allow_pickle=False);initial=np.r_[z['q'],z['s']]
write(out/'freeze.json',{'spec':spec,'script_sha256':digest(__file__),'contract_sha256':digest(H/'contract.json'),'duration_s':1,'condition':'photo_pulse','recording':'Every 1ms through5ms, then every10ms. Prefix compared exactly to existing5ms run. No one-second refined reference.'})
e=Engine(d,initial);e.reset(spec);stream=cp.cuda.Stream(non_blocking=True)
with stream:
 stream.begin_capture()
 for _ in range(round(.001/spec['h_s'])):e.step(spec)
 graph=stream.end_capture()
stream.synchronize();states=[initial];times=[0.];block_wall=[];tick=time.perf_counter();running_max=0.
for i in range(1000):
 if i in (1,3):
  e.d['photo'][:]=e.d['visual']*(.2 if i==1 else 0.)
  cp.cuda.get_current_stream().synchronize()
 start=time.perf_counter();graph.launch(stream=stream);stream.synchronize();block_wall.append(time.perf_counter()-start)
 if i<5 or (i+1)%10==0:
  a=cp.asnumpy(e.y)
  if not np.isfinite(a).all():raise FloatingPointError('Nonfinite base state')
  if a.min()<-1e-7 or a.max()>1+1e-7:raise FloatingPointError('Base state outside domain')
  states.append(a);times.append((i+1)*.001)
 if time.perf_counter()-tick>c['budget']['long_base_wall_limit_s']:raise TimeoutError('Long run exceeded fixed budget')
 if (i+1)%250==0:print(json.dumps({'simulated_s':(i+1)*.001,'wall_s':time.perf_counter()-tick}),flush=True)
wall=time.perf_counter()-tick;states=np.asarray(states)
short=np.load(H/'runs_01'/('photo_pulse_'+spec['id'])/'states.npz',allow_pickle=False)['state'][::2]
error=float(np.max(abs(states[:6]-short)))
if error!=0:raise ValueError('Long run changed the saved 5ms prefix')
np.savez_compressed(out/'states.npz',time_s=times,state=states)
write(out/'RESULT.json',{'classification':'PROMETEDOR_NO_CONFIRMADO','scope':c['scope'],'spec':spec,'simulated_s':1,'wall_with_sampling_s':wall,'stepping_wall_s':sum(block_wall),'block_wall_s':block_wall,'one_second_complete_base_run':True,'whole_organism':False,'one_second_refined_accuracy_validated':False,'first_5ms_prefix_exact':True,'first_5ms_prefix_error':error,'samples':len(times),'finite_samples':bool(np.isfinite(states).all()),'sample_min':float(states.min()),'sample_max':float(states.max()),'host_peak_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,'gpu_pool_GiB':cp.get_default_memory_pool().total_bytes()/2**30,'sources_freeze_sha256':digest(out/'freeze.json'),'state_file_sha256':digest(out/'states.npz')})
print(json.dumps({'completed':True,'simulated_s':1,'wall_s':wall,'prefix_exact':True}),flush=True)
