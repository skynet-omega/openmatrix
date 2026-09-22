"""Bounded stage3 whole-organism trajectory with full physical owners.

One fixed preparation and stimulus per process. No tuning, early success exit,
or retrospective admission criterion. Incomplete runs retain prefix evidence.
"""
from pathlib import Path
import sys,argparse,json,time,hashlib,shutil,resource,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0]=[str(ROOT/'work/motor14_20260922'),str(ROOT/'work/motor13_20260922')]
from motor_runtime import load,dump

def main():
 if not __debug__:raise RuntimeError('Legacy organism loaders require normal Python')
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--odor',choices=['sham','uniform','odor_left','odor_right'],required=True);p.add_argument('--ms',type=int,default=1000);a=p.parse_args()
 if a.ms!=1000:raise ValueError('Frozen pilot duration is 1000 ms')
 import cupy as cp,pandas as pd
 from session_io import write_state
 import kc_adaptive,event_coupling,block_midpoint,pn_execution,organism_adapter
 start=time.perf_counter();rows=[];times=[];obj=None;restores=[];status='STARTING';error=None;initial=None;prepared=None;events=adapter=split=pn=None
 out=a.out
 def traces():
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
 def snapshot(name):
  folder=out/name;folder.mkdir()
  # State codec includes CNS, all membrane chemistry/delays, body, world,
  # plasticity, eyes, muscle states, pending inputs and RNG. Static anatomy
  # comes from the separately identified parent checkpoint + intervention NPZ.
  write_state(folder/'session',obj.core.state_dict())
  write_state(folder/'prosthesis',obj.state())
  write_state(folder/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
  dump(folder/'boundary.json',obj.core.world.boundary.metadata())
 def budget():
  if time.perf_counter()-start>1150:raise TimeoutError('Frozen 1200 s process budget: preserving final state before deadline')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('Frozen 18 GiB RAM budget')
 try:
  obj,d,plan,Field,field,ports=load(out);h=obj.core.hybrid;initial=h.time_ns
  frozen=out/'executed_sources';frozen.mkdir()
  for file in HERE.glob('*'):
   if file.suffix in ('.py','.cpp') or file.name=='PLAN.json':shutil.copy2(file,frozen/file.name)
  dump(out/'FROZEN.json',{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in frozen.iterdir()})
  restores.append(kc_adaptive.install(h));events,undo=event_coupling.install(h);restores.append(undo)
  adapter,undo=organism_adapter.install(h,events);restores.append(undo)
  pn,undo=pn_execution.install(h);restores.append(undo)
  split,undo=block_midpoint.install(h,125000);restores.append(undo)
  table=pd.read_parquet(ROOT/'data/male_v10/nodes.parquet',columns=['bodyId','type','somaSide'])
  types=table['type'].fillna('').astype(str)
  mask=types.str.match(r'^(DNa02|DNb05|PFL3|hDeltaK|PFG)(?:$|_)').to_numpy();selected=np.flatnonzero(mask)
  dump(out/'CENTRAL_ROWS.json',{'rows':selected.tolist(),'ids':table.bodyId.to_numpy()[selected].tolist(),'types':types.to_numpy()[selected].tolist(),'somaSide':table.somaSide.fillna('unclear').to_numpy()[selected].tolist(),'scope':'Anatomical labels are not proof of homologous connectivity'})
  dump(out/'RUN_CONTRACT.json',{'condition':a.odor,'preparation_ms':40,'trial_ms':a.ms,'initial_clock_ns':initial,'checkpoint':str(ROOT/plan['checkpoint']),'checkpoint_manifest_sha256':hashlib.sha256((ROOT/plan['checkpoint']/'manifest.json').read_bytes()).hexdigest(),'static_field':'live geometry; uniform intensity 1 per antenna; no dose renormalization','engineering_interventions':str(out/'INTERVENCIONES.json'),'decoder_gain':plan['decoder_gain'],'no_orn_rescaling':True,'budget_s':1200,'central_records':len(selected)})
  yaw0=d.yaw_grados(obj.body.data.qpos)
  for phase,duration in [('preparacion',40),('ensayo',a.ms)]:
   if phase=='ensayo':
    prepared=h.time_ns;snapshot('prepared_state');d.instalar_campo(obj,Field,field,a.odor,0.);yaw0=d.yaw_grados(obj.body.data.qpos)
   status=phase
   for k in range(duration):
    budget();used=obj.core.pending_sensors.copy();t=time.perf_counter();obj.step();cp.cuda.runtime.deviceSynchronize()
    row=d.captura(obj,ports,phase,k+1,used,yaw0,cp)
    row['central_q']=h.release()[selected].copy();row['central_transmission']=h.state[h.transmission_start+selected].copy()
    rows.append(row);timing={'phase':phase,'step':k+1,'clock_ns':int(h.time_ns),'wall_s':time.perf_counter()-t,'elapsed_s':time.perf_counter()-start};times.append(timing)
    with (out/'TIMES.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
    if k==0 or (k+1)%10==0:
     traces();print(json.dumps(dict(timing,yaw_deg=row['yaw_delta_deg'],command_yaw=row['command_yaw_rate_rad_s'])),flush=True)
    if phase=='ensayo' and k+1==100:snapshot('state_100ms')
   if phase=='preparacion' and not np.all([np.all(r['sensores_usados']==0) for r in rows]):raise ValueError('Preparation consumed odor')
  if h.time_ns!=initial+(40+a.ms)*1_000_000:raise ValueError('End clock mismatch')
  status='COMPLETE'
 except Exception as exc:
  error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};status='INCOMPLETE'
  print(json.dumps(error),flush=True)
 finally:
  out.mkdir(parents=True,exist_ok=True);traces()
  if obj:
   try:snapshot('final_state')
   except Exception as exc:
    dump(out/'CHECKPOINT_ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
  trial=[r for r in rows if r['fase']=='ensayo']
  dump(out/'RESULT.json',{'status':status,'error':error,'odor':a.odor,'requested_trial_ms':a.ms,'completed_trial_ms':len(trial),'completed_preparation_ms':len(rows)-len(trial),'wall_total_s':time.perf_counter()-start,'step_wall_s':sum(t['wall_s'] for t in times),'rss_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'last_yaw_deg':None if not trial else trial[-1]['yaw_delta_deg'],'historical_100ms_yaw_deg':None if len(trial)<100 else trial[99]['yaw_delta_deg'],'stage3_pass':None,'reason':'Exploratory trajectory; no biological or historical admission inferred','events':None if events is None else events.report,'native':None if adapter is None else adapter.report,'PN_execution':pn,'partition':split})
  for f in reversed(restores):f()
  if obj:obj.close()
 if status!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
