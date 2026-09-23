"""Small analytic and corrupt-input checks; not organism validation."""
import json,sys
import numpy as np
import cupy as cp
from graph_core import NativeGraph
from event_ports import FilterPorts
from types import SimpleNamespace
checks=[]
target=cp.asarray([.7,.3]);rate=cp.asarray([2.,7.])
g=NativeGraph(np.array([.2,.8]),lambda z:(target,rate),rtol=1e-8,atol=1e-10,norm_size=2)
g.advance(100000000,10000000,1,10000000)
expected=cp.asnumpy(target)+(np.array([.2,.8])-cp.asnumpy(target))*np.exp(-.1*cp.asnumpy(rate))
error=float(np.max(abs(g.x.get()-expected)))
if error>1e-13:raise ValueError(('analytic flow',error))
checks.append({'analytic_flow_error':error})
target.set(np.array([.1,.9]));cp.cuda.get_current_stream().synchronize()
old=g.x.get();g.advance(100000000,10000000,1,10000000)
expected=np.array([.1,.9])+(old-np.array([.1,.9]))*np.exp(-.1*np.array([2.,7.]))
error=float(np.max(abs(g.x.get()-expected)))
if error>1e-13:raise ValueError(('mutable input',error))
checks.append({'mutable_input_error':error})
target.set(np.array([np.nan,.9]));cp.cuda.get_current_stream().synchronize()
try:g.advance(1000,1000,1,1000)
except RuntimeError as e:checks.append({'nonfinite_rejected':str(e)})
else:raise ValueError('NaN accepted')
g.close()
p=FilterPorts([0,1],[2,3]);w=SimpleNamespace(q=np.array([.1,.2]),s=np.array([.3,.4]),tau=np.array([.01,.02]),ts=.01,times=[.002,.006,.004],rows=[0,0,1],jumps=[.02,.03,.04])
p.update(w);clock=cp.asarray([0.,.008]);out=p.project(cp.zeros(4),clock,1.).get()
sys.path.insert(0,'/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922')
from event_waveform import Waveform
ref=Waveform(w.q,w.s,w.tau,w.ts,.008);ref.add(w.times,np.asarray(w.rows),w.jumps);a,b=ref.at(.008)
error=float(np.max(abs(out-np.r_[a,b])))
if error>1e-13:raise ValueError(('event port',error))
checks.append({'event_port_error':error})
print(json.dumps({'checks':checks,'passed':True}))
