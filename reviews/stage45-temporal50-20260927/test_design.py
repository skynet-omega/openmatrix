"""Algebraic instrument controls, never components of the organism."""
from pathlib import Path
import json,numpy as np
from temporal_input import ARMS,schedule,need
H=Path(__file__).resolve().parent

def measure(function):
 x={a:function(schedule(a)) for a in ARMS}
 I={s:x[s+'00']+x[s+'11']-x[s+'01']-x[s+'10'] for s in ['p','m']}
 return I,float(np.mean((I['p'][10:130]-I['m'][10:130])*.5))
def independent(x):
 y=np.zeros((140,2));a=np.array([.91,.97])
 for k in range(1,140):y[k]=a*y[k-1]+(1-a)*(x[k]**2+.2*x[k])
 return y[:,0]+1.5*y[:,1]
def static(x):return np.tanh(.7*x[:,0]+1.3*x[:,1])
def temporal(x):
 l,r=x.T;dl=np.r_[np.zeros(40),l[:-40]];dr=np.r_[np.zeros(40),r[:-40]]
 return dl*r-dr*l
results={}
for name,fn in [('separable_memory',independent),('instantaneous_joint',static),('delayed_interaction',temporal)]:
 I,J=measure(fn);results[name]=dict(max_abs_I=float(max(np.abs(y).max() for y in I.values())),J=J)
need(results['separable_memory']['max_abs_I']<1e-12,'Separable null failed')
need(abs(results['instantaneous_joint']['J'])<1e-12,'Instantaneous order null failed')
need(abs(results['delayed_interaction']['J'])>.1,'Positive temporal control absent')
for a in ARMS:need(np.array_equal(schedule(a)[:10],np.zeros((10,2))),'Shared baseline')
p=H/'DESIGN_CONTROLS.json';need(not p.exists(),'Preserve controls');p.write_text(json.dumps(dict(results=results,neural_steps=0,evaluator_only=True),indent=2)+'\n');print(json.dumps(results))
