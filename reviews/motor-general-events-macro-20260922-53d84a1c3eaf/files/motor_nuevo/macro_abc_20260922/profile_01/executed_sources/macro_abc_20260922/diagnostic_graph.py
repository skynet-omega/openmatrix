"""Read-only GPU stage capture; unchanged arithmetic and acceptance rules."""
from pathlib import Path
import sys,numpy as np,cupy as cp
R=Path(__file__).resolve().parent;P=R.parents[1]/'campanas/etapa3_motor_nuevo_20260922';sys.path.insert(0,str(P))
from graph_core import NativeGraph
class DiagnosticGraph(NativeGraph):
 def trial(self):
  self.diagnostic_stages=[];self.diagnostic_outputs=[]
  return super().trial()
 def coeff(self,y,frac):
  z,a,b=super().coeff(y,frac)
  self.diagnostic_stages.append((float(frac),y.copy(),z.copy(),a.copy(),b.copy()))
  return z,a,b
 def midpoint(self,y,frac,start):
  z,a,b=self.coeff(y,start);middle=z+(-cp.expm1(-.5*frac*self.clock[1]*b))*(a-z)
  _,a,b=self.coeff(middle,start+.5*frac);out=z+(-cp.expm1(-frac*self.clock[1]*b))*(a-z)
  pre=out.copy();post=self.project(out,self.clock,start+frac) if self.project else out
  self.diagnostic_outputs.append((float(frac),float(start),pre,post.copy()))
  return post
 def failure_arrays(self):
  ids=np.asarray(self.last_failure['sample_indices'],dtype=np.int64)
  result={'indices':ids,'fine':self.fine.get(stream=self.stream),'epoch_initial':self.backup.get(stream=self.stream),'clock':self.clock.get(stream=self.stream)}
  # All six captured stages are read before another kernel trial can overwrite.
  for i,(frac,y,z,a,b) in enumerate(self.diagnostic_stages):
   result[f'stage_{i}_fraction']=np.asarray(frac)
   for name,v in [('raw',y),('projected',z),('target',a),('rate',b)]:result[f'stage_{i}_{name}']=v.get(stream=self.stream)[ids]
  for i,(frac,start,pre,post) in enumerate(self.diagnostic_outputs):
   result[f'out_{i}_fraction']=np.asarray(frac);result[f'out_{i}_start']=np.asarray(start)
   result[f'out_{i}_pre']=pre.get(stream=self.stream)[ids];result[f'out_{i}_post']=post.get(stream=self.stream)[ids]
  return result
