"""Observe the unchanged real trajectory, without modifying its mathematics."""
from pathlib import Path
import sys,json,time,hashlib,types,traceback,resource,shutil
import numpy as np
T=Path(__file__).resolve().parent;ROOT=T.parents[1];H=T.parent/'pipeline_review_20260922'
sys.path.insert(0,str(H));import run_pipeline
from motor_runtime import load
from run_storage import RunStorage,atomic_json
from runtime_session import RuntimeSession

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--engine',choices=['causal_cuda','reference_cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 storage=RunStorage(a.out);start=time.perf_counter();obj=session=None;records=[];status='INCOMPLETE';error=None
 try:
  import cupy as cp
  from session_io import write_state
  from operator_state import OperatorState,LEGACY_BINDINGS
  obj,d,plan,Field,field,ports=load(a.out/'inputs');b=obj.core.hybrid;session=RuntimeSession(b,a.engine)
  write_state(a.out/'initial_brain',b.state_dict())
  atomic_json(a.out/'OPERATOR.json',{'identity':OperatorState(b,LEGACY_BINDINGS).state_dict()['identity_sha256']})
  base=b._spatial_batch
  while hasattr(base,'base'):base=base.base
  event=session.events;old_cell=base.advance;old_step=event.step
  target_row=57073;slot=np.flatnonzero(np.asarray(event.rows)[event.gamma]==target_row)
  atomic_json(a.out/'MAPPING.json',{'event_rows':np.asarray(event.rows).tolist(),'gamma':np.asarray(event.gamma).tolist(),'target_row':target_row,'target_cell_slot':slot.tolist(),'initial_clock_ns':int(b.time_ns)})
  cells=[]
  def cell(self,ns,ge,gi,**kw):
   index=len(cells);record={'index':index,'duration_ns':int(ns),'start_elapsed_ns':int(self.elapsed_ns),'q_before':self.host(self.q).copy(),'soma_counts_before':self.host(self.counts).copy(),'target_delta_before':self.host(self.delta)[slot].copy(),'target_gates_before':self.host(self.gates)[slot].copy(),'ge':np.asarray(ge).copy(),'gi':np.asarray(gi).copy()}
   if index==0:
    fixture={name:self.host(getattr(self,name)).copy() for name in ('delta','gates','q','counts','last_siz','previous_slope','trough','clipped','C','G','chanG','chanb','shuntG','shuntb','ena','caps','tau','obs')}
    fixture.update({('ax_'+k):self.host(v).copy() for k,v in self._motor_axonal_callback.fields.items()})
    fixture['ax_gain']=self.host(self._motor_axonal_callback.gain).copy();fixture['ge']=np.asarray(ge);fixture['gi']=np.asarray(gi)
    np.savez_compressed(a.out/'first_cell_fixture.npz',**fixture)
    atomic_json(a.out/'first_cell_fixture.json',{'rest':float(self.rest),'n':self.n,'ns':int(ns),'inner_step_ns':int(kw.get('inner_step_ns',25000)),'ts':float(self._motor_axonal_callback.wrapper.publisher.synaptic_tau),'clock_start':int(self.elapsed_ns)})
   result=old_cell(ns,ge,gi,**kw)
   record.update(q_after=self.host(self.q).copy(),soma_counts_after=self.host(self.counts).copy(),target_delta_after=self.host(self.delta)[slot].copy(),target_gates_after=self.host(self.gates)[slot].copy())
   # Array records use NPZ; no NumPy scalars in JSON receipts.
   np.savez_compressed(a.out/f'cell_{index:02d}.npz',**record);cells.append({'index':index,'duration_ns':int(ns)})
   return result
  base.advance=types.MethodType(cell,base)
  def step(brain,ns,drive,light):
   index=len(records);w=event.active
   state={'before':brain.state.copy(),'q0':np.asarray(w.q).copy(),'s0':np.asarray(w.s).copy(),'tau':np.asarray(w.tau).copy(),'ts':np.asarray(w.ts),'times':np.asarray(w.times,dtype=float),'rows':np.asarray(w.rows,dtype=np.int64),'jumps':np.asarray(w.jumps,dtype=float),'drive':np.asarray(drive).copy(),'light':np.asarray(light).copy()}
   clock=int(brain.time_ns);result=old_step(brain,ns,drive,light);state['after']=brain.state.copy()
   np.savez_compressed(a.out/f'boundary_{index:02d}.npz',**state)
   records.append({'index':index,'phase':'accepted' if ns==125000 else 'predictor','ns':int(ns),'clock_start_ns':clock,'clock_end_ns':int(brain.time_ns),'events':len(w.times)})
   return result
  event.step=step
  yaw=d.yaw_grados(obj.body.data.qpos);used=obj.core.pending_sensors.copy();t=time.perf_counter();obj.step();cp.cuda.get_current_stream().synchronize();step_wall=time.perf_counter()-t
  row=d.captura(obj,ports,'ensayo',1,used,yaw,cp)
  import pandas as pd
  table=pd.read_parquet(run_pipeline.OLD/'data/male_v10/nodes.parquet',columns=['type']);selected=np.flatnonzero(table['type'].fillna('').astype(str).str.match(r'^(DNa02|DNb05|PFL3|hDeltaK|PFG)(?:$|_)').to_numpy())
  row['central_q']=b.release()[selected].copy();row['central_transmission']=b.state[b.transmission_start+selected].copy()
  np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([v]) for k,v in row.items()})
  write_state(a.out/'brain_final',b.state_dict())
  np.savez_compressed(a.out/'body_final.npz',**{k:getattr(obj.body.data,k).copy() for k in ('qpos','qvel','qacc','qacc_warmstart','act','ctrl')},pending_sensors=obj.core.pending_sensors.copy(),pending_excitation=obj.core.pending_excitation.copy())
  atomic_json(a.out/'BOUNDARIES.json',records);atomic_json(a.out/'CELLS.json',cells)
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('Memory budget')
  status='COMPLETE'
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
 finally:
  atomic_json(a.out/'RESULT.json',{'status':status,'engine':a.engine,'error':error,'wall_s':time.perf_counter()-start,'boundary_count':len(records),'step_wall_s':locals().get('step_wall'),'scientific_admission':False})
  if session:session.close()
  if obj:obj.close()
 print(json.dumps({'status':status,'engine':a.engine,'boundaries':len(records)}))

if __name__=='__main__':main()
