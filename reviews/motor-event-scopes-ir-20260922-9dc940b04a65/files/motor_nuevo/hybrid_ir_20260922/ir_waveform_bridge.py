"""CPU functional bridge to unchanged dimensional IR; not a production executor."""
import copy,numpy as np
from trayectorias_cpu import Bloque,exigir

def from_ir(model,limits=(-10.,10.)):
 exigir(len(model.pops)==1 and model.pops[0]['n']==1 and model.nin==model.nout==1,'Scalar one-entity block required')
 exigir(not model.clamp_mask.any(),'Clamps require an explicit event/domain contract')
 desc=model.spec['populations'][0]
 from model import unit
 exigir(unit(next(iter(desc['inputs'].values()))['unit'])==unit(next(iter(desc['outputs'].values()))['unit'])=={},'Dimensionless interblock ports required')
 p=model.pops[0];bias=model.port_bias.copy();mass=model.mass.toarray()
 def rhs(t,x,u):
  # Shallow numerical descriptor view; no mutation of shared input/model state.
  view=copy.copy(model);view.port_bias=bias+u
  return view.raw_rhs(t,x)  # Return F, not M^-1 F. Bloque owns the mass solve.
 def output(t,x):
  v={'t':t,**{k:x[s] for k,s in p['states'].items()},**{k:model.parameters[s] for k,s in p['params'].items()}}
  result=np.asarray(next(iter(p['outputs'].values()))[1].evaluate(v))
  exigir(result.size==1 and np.isfinite(result).all(),'Output is not finite scalar')
  return float(result.item())
 exigir(np.all(model.scale==model.scale[0]),'External prototype uses one error scale per block')
 return Bloque(p['id'],mass,model.initial,rhs,output,limits,float(model.scale[0]))
