from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from portable import PortablePN,read,errors
from compare_pn import write_state
from cyclic_plan import CyclicPlan
from cyclic_tree import CyclicTree
from condensation import Condensation
from compiled_stage import channels
from pn_coupled_ionic import stage_channels
import compact_step,compiled_step

def require(ok,message):
 if not ok:raise ValueError(message)

def main():
 out=H/sys.argv[1];out.mkdir(exist_ok=False);p=PortablePN(H/'capture_01');case=read(H/'capture_01/case_00');f=case['inputs'];solves=[];original=p.backend.solve
 def capture(rhs,**kw):
  x,r=original(rhs,**kw);solves.append((rhs.copy(),kw,x.copy()));return x,r
 p.backend.solve=capture;p.restore(case['before']);p.advance(f['dt_ns'],f['current'],**f['options']);p.backend.solve=original
 shift=solves[0][1]['shift'];extra=p.backend.M-diags(p.backend.C);plan=CyclicPlan(p.backend.G,p.backend.M,np.unique(extra.nonzero()[0]));gpu=CyclicTree(plan,shift);rows=[]
 for i,(rhs,kw,old) in enumerate(solves):
  diagonal=np.zeros(len(rhs));diagonal[kw['diagonal_update'][0]]=kw['diagonal_update'][1];A=(p.backend.G+shift*p.backend.M+diags(diagonal)).tocsc();reference=splu(A).solve(rhs)
  limit=max(kw['atol'],kw['rtol']*np.linalg.norm(rhs));maximum=0.;first=None;identical=True;norm=0.
  for repeat in range(20):
   x=gpu.solve(rhs,diagonal);maximum=max(maximum,float(np.max(abs(x-reference))));norm=max(norm,float(np.linalg.norm(A@x-rhs)))
   if first is None:first=x.copy()
   else:identical=identical and np.array_equal(x,first)
  require(maximum<=1e-7 and norm<=limit,'Independent LU/full residual mismatch');require(identical,'Nondeterministic GPU repetition')
  write_state(out/f'linear_{i}',{'rhs':rhs,'diagonal':diagonal,'original':old,'independent_LU':reference,'candidate':x,'shift':shift,'limit':float(limit)})
  rows.append({'case':i,'max_error_vs_SuperLU':maximum,'maximum_full_residual':norm,'original_limit':float(limit),'20_repeats_identical':identical})
 rejected=[]
 for name,rhs2,diag2 in [('NaN_RHS',np.full_like(rhs,np.nan),diagonal),('negative_pivot',rhs,-(plan.dg+shift*plan.dm)-1)]:
  try:gpu.solve(rhs2,diag2)
  except (ValueError,ArithmeticError):rejected.append(name)
  else:raise ValueError('Invalid GPU input accepted')
 # A's published numerical cache must reject a change outside the retained support.
 protected=np.union1d(p.active_nodes,p.calcium_port.nodes)
 for stage in f['options']['synaptic_stages']:protected=np.union1d(protected,stage['nodes'])
 protected=np.union1d(protected,np.unique(extra.nonzero()[0]));a=Condensation(p.backend,protected,shift);bad=np.zeros(len(rhs));bad[a.plan.steps[0,0]]=1
 try:a.solve(rhs,bad)
 except ValueError:rejected.append('unprotected_diagonal')
 else:raise ValueError('Unprotected nonlinear coefficient silently ignored')
 # Same numerical gate equations across voltage and base-state values.
 v=np.array([-100.,-65.,-40.,0.,40.]);base=np.full((5,4),.25);gbar=p.gbar_nS[:5];ref=stage_channels(v,base,1e-5,gbar,p.reversal_mV);got=channels(v,base,1e-5,gbar,p.reversal_mV)
 gate_error=max(float(np.max(abs(ref[i]-got[j]))) for i,j in [(0,0),(1,1),(2,2)]);require(gate_error<=1e-10,'Compiled kinetics mismatch')
 atomic=[]
 for mod,name,method in [(compact_step,'compact_stage','advance_compact'),(compiled_step,'compiled_stage','advance_compiled')]:
  original_stage=getattr(mod,name);calls=[0]
  def second_failure(*a,**kw):
   calls[0]+=1
   if calls[0]==2:return None,{'accepted':False,'reason':'injected_second_stage_rejection','history':[],'total_iterations':0}
   return original_stage(*a,**kw)
  p.restore(case['before']);before=p.state();setattr(mod,name,second_failure)
  try:r=getattr(mod,method)(p,f['dt_ns'],f['current'],_local_channel=p.calcium_port,**f['options'])
  finally:setattr(mod,name,original_stage)
  require(not r['accepted'] and calls[0]==2 and max(errors(before,p.state()).values())==0,'Rejection committed partial state');atomic.append(method)
 result={'scope':'Numerical verification, not new scientific candidate or timing promotion','linear':rows,'invalid_inputs_rejected':rejected,'compiled_kinetics_max_error':gate_error,'second_stage_atomicity':atomic}
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
