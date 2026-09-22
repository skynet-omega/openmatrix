"""Capture complete PN steps in one real 1ms neural/body interval."""
from pathlib import Path
import sys,time,json,hashlib,resource
import numpy as np
from scipy.sparse import save_npz
if not __debug__:raise RuntimeError('Legacy whole-body loader requires normal Python')
H=Path(__file__).resolve().parent;M=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0]=[str(M/'work/motor14_20260922'),str(M/'src')]
from motor_runtime import load,dump
from session_io import write_state
from threadpoolctl import threadpool_limits
out=H/'capture_01';obj,*_=load(out);pn=obj.core.hybrid._online_source.pn;backend=pn.backend;ca=pn.calcium_port
try:
 save_npz(out/'G.npz',backend.G);save_npz(out/'M.npz',backend.M)
 static=dict(C=backend.C,active=pn.active_nodes,gbar=pn.gbar_nS,reversal=pn.reversal_mV,leak=pn.leak_reversal_mV,
  ca_nodes=ca.nodes,ca_gbar=ca.gbar,ca_reversal=ca.reversal,ca_half=ca.half,ca_slope=ca.slope,ca_tau=ca.tau,ca_power=ca.power,
  ca_sites=ca.nodes[ca.site_slot],ca_fractions=ca.fractions,chem_ids=ca.chemistry.ids,chem_parameters=ca.chemistry.parameters,
  ca_provenance=ca.provenance,chem_provenance=ca.chemistry.provenance,ca_enabled=ca.enabled,release_enabled=ca.release_enabled)
 write_state(out/'static',static)
 write_state(out/'graph_plan',{k:getattr(backend._graph_plan,k) for k in ['steps','core','pairs','base','mass','dg','dm','core_edges']})
 def state():return dict(voltage=pn.voltage.copy(),gates=pn.gates.copy(),charge=pn.ionic_charge_pC.copy(),time_ns=pn.time_ns,calcium=ca.state_dict())
 write_state(out/'initial',state());original=pn.advance;calls=[]
 def capture(dt,current,**options):
  index=len(calls);args=dict(dt_ns=dt,current=current,options=options)
  before=state();start=time.perf_counter();result=original(dt,current,**options);wall=time.perf_counter()-start
  if index in (0,1):write_state(out/f'case_{index:02d}',dict(inputs=args,before=before,after=state(),result=result))
  # Save every prescribed PN input in the real 1ms interval; no input is invented.
  write_state(out/f'input_{index:03d}',args);calls.append({'index':index,'dt_ns':dt,'wall_s':wall,'accepted':result['accepted']})
  return result
 pn.advance=capture
 with threadpool_limits(limits=1,user_api='blas'):
  start=time.perf_counter();obj.step();wall=time.perf_counter()-start
 write_state(out/'final',state())
 dump(out/'RESULT.json',{'scope':'PN inputs and state in real 1ms full-organism prefix; PN replay is conditional on these inputs','calls':calls,'body_interval_wall_s_including_capture':wall,'pn_step_wall_sum_s':sum(x['wall_s'] for x in calls),'PN_coordinates':len(pn.voltage),'retained_junction_core':len(backend._graph_plan.core),'linear_plan_steps':len(backend._graph_plan.steps),'RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})
 print(json.dumps({'calls':len(calls),'wall_s':wall,'pn_s':sum(x['wall_s'] for x in calls),'core':len(backend._graph_plan.core)}),flush=True)
finally:obj.close()
