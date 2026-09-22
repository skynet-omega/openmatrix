"""Actual organism: separately measure membranes, CNS, body and observation."""
from pathlib import Path
import argparse,sys,time,json,resource,hashlib,shutil,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work')
sys.path[:0]=[str(PARENT),str(OLD/'motor14_20260922'),str(OLD/'motor13_20260922')]
from motor_runtime import load,dump

def main():
 if not __debug__:raise RuntimeError('Normal Python required for unchanged organism loaders')
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--ms',type=int,default=5);p.add_argument('--native-cell',action='store_true');p.add_argument('--compressed',action='store_true');p.add_argument('--ros',action='store_true');p.add_argument('--event-boundaries',action='store_true');a=p.parse_args()
 import cupy as cp
 import kc_adaptive,event_coupling,block_midpoint,pn_execution,organism_adapter
 from session_io import write_state
 start=time.perf_counter();restores=[];rows=[];timings=[];obj=None;status='INCOMPLETE';exception=None;phase={};events=adapter=pn=split=cell=None
 def measure(owner,name,label):
  old=getattr(owner,name)
  def wrap(*args,**kw):
   t=time.perf_counter()
   try:return old(*args,**kw)
   finally:phase[label]=phase.get(label,0.)+time.perf_counter()-t
  setattr(owner,name,wrap);restores.append(lambda:setattr(owner,name,old))
 try:
  obj,d,plan,Field,field,ports=load(a.out);h=obj.core.hybrid
  frozen=a.out/'sources';frozen.mkdir()
  for base in (PARENT,HERE):
   for file in base.iterdir():
    if file.suffix in ('.py','.cpp','.cu') or file.name=='PLAN.json':
     target=frozen/base.name/file.name;target.parent.mkdir(exist_ok=True);shutil.copy2(file,target)
  dump(a.out/'FROZEN.json',{str(f.relative_to(frozen)):hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen.rglob('*') if f.is_file()})
  restores.append(kc_adaptive.install(h));events,undo=event_coupling.install(h);restores.append(undo)
  adapter,undo=organism_adapter.install(h,events,event_boundaries=a.event_boundaries);restores.append(undo)
  pn,undo=pn_execution.install(h);restores.append(undo)
  if a.native_cell or a.ros:
   sys.path.insert(0,str(HERE));import native_cell
   cell,undo=native_cell.install(h,events,compressed=a.compressed,method="ros" if a.ros else "midpoint");restores.append(undo)
  split,undo=block_midpoint.install(h,125000);restores.append(undo)
  from rollback_guard import install as guard
  restores.append(guard(h))
  measure(h._spatial_batch,'advance','spatial_membranes_and_events_s');measure(h._online_source,'advance','PN_and_source_s');measure(events,'step','global_CNS_s');measure(obj.body,'advance','body_advance_s')
  yaw=d.yaw_grados(obj.body.data.qpos);origin=h.time_ns
  for k in range(a.ms):
   phase.clear();used=obj.core.pending_sensors.copy();cp.cuda.runtime.deviceSynchronize();t=time.perf_counter();obj.step();cp.cuda.runtime.deviceSynchronize();step=time.perf_counter()-t
   t=time.perf_counter();row=d.captura(obj,ports,'ensayo',k+1,used,yaw,cp);cp.cuda.runtime.deviceSynchronize();capture=time.perf_counter()-t
   rows.append(row);record={'step':k+1,'whole_step_s':step,'observation_s':capture,**phase};timings.append(record)
   with (a.out/'TIMES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(json.dumps(record),flush=True)
  if h.time_ns!=origin+a.ms*1000000:raise ValueError('Wrong physical horizon')
  t=time.perf_counter();np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]});trace_save=time.perf_counter()-t
  write_state(a.out/'brain_final',h.state_dict())
  np.savez_compressed(a.out/'body_final.npz',qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,qacc=obj.body.data.qacc,qacc_warmstart=obj.body.data.qacc_warmstart,act=obj.body.data.act,ctrl=obj.body.data.ctrl,pending_sensors=obj.core.pending_sensors,pending_excitation=obj.core.pending_excitation)
  status='COMPLETE'
 except BaseException as exc:
  exception={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
 finally:
  a.out.mkdir(exist_ok=True,parents=True)
  dump(a.out/'RESULT.json',{'status':status,'error':exception,'ms':len(timings),'requested_ms':a.ms,'wall_total_s':time.perf_counter()-start,'native_cell':a.native_cell,'compressed':a.compressed,'ros':a.ros,'event_boundaries':a.event_boundaries,'times':timings,'rss_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'PN_execution':pn,'native_CNS':None if adapter is None else adapter.report,'events':None if events is None else events.report,'cell':None if cell is None else cell.report,'partition':split})
  for undo in reversed(restores):undo()
  if obj:obj.close()

if __name__=='__main__':main()
