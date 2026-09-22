from pathlib import Path
import sys,json,time
import numpy as np
from numba import njit
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from portable import PortablePN,read
from condensation import Condensation
p=PortablePN(H/'capture_01');c=read(H/'capture_01/case_00');fields=c['inputs'];protected=np.union1d(p.active_nodes,p.calcium_port.nodes)
for x in fields['options']['synaptic_stages']:protected=np.union1d(protected,x['nodes'])
# Preserve the full-mass interface; its dense correction must not be diagonalized.
from scipy.sparse import diags
extra=p.backend.M-diags(p.backend.C);protected=np.union1d(protected,np.unique(extra.nonzero()[0]))
shift=1/((1-1/np.sqrt(2))*fields['dt_ns']*1e-9);start=time.perf_counter();a=Condensation(p.backend,protected,shift);setup=time.perf_counter()-start
old=p.backend._graph_plan;last_node=np.zeros(len(p.voltage),dtype=np.int64);last_edge=np.zeros(len(old.pairs),dtype=np.int64);levels=[]
for step in old.steps:
 i,j,k,e,f,cross=step;nodes=[x for x in [i,j,k] if x>=0];edges=[x for x in [e,f,cross] if x>=0];level=1+max([last_node[x] for x in nodes]+[last_edge[x] for x in edges]);levels.append(level)
 for x in nodes:last_node[x]=level
 for x in edges:last_edge[x]=level
r={'protected_nodes':len(protected),'condensed_core':len(a.core),'original_core':len(old.core),'small_core':len(a.small.core),'passive_eliminated':len(a.plan.steps),'condensation_setup_s':setup,'K_nnz':a.K.nnz,'fixed_order_GPU_dependency_levels':int(max(levels)),'dependency_scope':'Conservative conflict-free schedule of existing order, not a lower bound for a new cyclic-reduction ordering'}
# Capture real Newton linear inputs from replay; do not substitute random RHS.
solves=[];original=p.backend.solve
def capture(rhs,**kw):
 result=original(rhs,**kw);solves.append((rhs.copy(),kw,result[0].copy()));return result
p.backend.solve=capture;p.restore(c['before']);p.advance(fields['dt_ns'],fields['current'],**fields['options'])
rows=[]
for i,(rhs,kw,expected) in enumerate(solves[:4]):
 diagonal=np.zeros(len(rhs));diagonal[kw['diagonal_update'][0]]=kw['diagonal_update'][1]
 a.solve(rhs,diagonal);times={}
 for name,fn in [('original',lambda:old.solve(rhs,shift,diagonal)),('condensed',lambda:a.solve(rhs,diagonal))]:
  ticks=[]
  for _ in range(5):start=time.perf_counter();x=fn();ticks.append(time.perf_counter()-start)
  times[name]=float(np.median(ticks))
 error=float(np.max(abs(x-expected)));rows.append({'case':i,'max_voltage_error':error,'median_wall_s':times})
r['linear_cases']=rows;(H/'LINEAR_SCREEN.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
