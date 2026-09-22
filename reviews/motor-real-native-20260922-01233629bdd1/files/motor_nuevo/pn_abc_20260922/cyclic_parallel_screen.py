from pathlib import Path
import sys,json,time
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from portable import PortablePN,read
from cyclic_tree import CyclicTree
from cyclic_plan import CyclicPlan
from scipy.sparse import diags
import cupy as cp
p=PortablePN(H/'capture_01');c=read(H/'capture_01/case_00');f=c['inputs'];solves=[];original=p.backend.solve

def capture(rhs,**kw):
 result=original(rhs,**kw);solves.append((rhs.copy(),kw,result[0].copy()));return result
p.backend.solve=capture;p.restore(c['before']);p.advance(f['dt_ns'],f['current'],**f['options'])
shift=solves[0][1]['shift'];old=p.backend._graph_plan;start=time.perf_counter();extra=p.backend.M-diags(p.backend.C);protected=np.unique(extra.nonzero()[0]);new=CyclicPlan(p.backend.G,p.backend.M,protected);gpu=CyclicTree(new,shift);setup=time.perf_counter()-start
result={'scope':'Real PN Newton linear systems, not complete PN or body','setup_s':setup,'levels':gpu.level_count,'maximum_parallel_steps':gpu.maximum_parallel,'host_transfers_in_end_to_end':True,'rows':[]}
for i,(rhs,kw,expected) in enumerate(solves[:4]):
 diagonal=np.zeros(len(rhs));diagonal[kw['diagonal_update'][0]]=kw['diagonal_update'][1];rows={}
 for name,fn in [('CPU_original',lambda:old.solve(rhs,shift,diagonal)),('GPU_parallel',lambda:gpu.solve(rhs,diagonal))]:
  fn();ticks=[]
  for j in range(5):start=time.perf_counter();x=fn();ticks.append(time.perf_counter()-start)
  rows[name]=float(np.median(ticks))
 A=p.backend.G+shift*p.backend.M;residual=rhs-(A@x+diagonal*x)
 result['rows'].append({'case':i,'median_wall_s':rows,'speedup':rows['CPU_original']/rows['GPU_parallel'],'max_voltage_error':float(np.max(abs(x-expected))),'full_linear_residual_l2':float(np.linalg.norm(residual)),'original_limit':max(kw['atol'],kw['rtol']*float(np.linalg.norm(rhs)))})
start=cp.cuda.Event();end=cp.cuda.Event()
with gpu.stream:
 start.record()
 for i in range(10):gpu.graph.launch(gpu.stream)
 end.record()
end.synchronize();result['GPU_resident_graph_s']=cp.cuda.get_elapsed_time(start,end)*.001/10
result['device_pool_bytes']=cp.get_default_memory_pool().total_bytes();(H/'CYCLIC_PARALLEL_CORE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
