"""External experimental stimulus only. No state or motor policy writes."""
import types,hashlib,json
import numpy as np
BITS=np.array([1,0,1,1,0,0,0,1,1,0,1,0],dtype=np.float64)
ARMS=tuple(f'{sign}{a}{b}' for sign in ('p','m') for a,b in ((0,0),(0,1),(1,0),(1,1)))
def need(x,m):
 if not x:raise ValueError(m)
def schedule(arm):
 need(arm in ARMS,'Unknown factorial condition')
 lag=40 if arm[0]=='p' else -40;s=np.repeat(BITS,10);r=np.roll(s,lag)
 a=int(arm[1]);b=int(arm[2]);out=np.zeros((140,2),np.float64)
 out[10:130,0]=s if a==0 else 1-s;out[10:130,1]=r if b==0 else 1-r
 need(np.array_equal(out.sum(axis=0),[60.,60.]),'Unbalanced marginal input')
 return out

def install(stimulus,arm):
 table=schedule(arm);start=stimulus.k;original=stimulus.consume;indices=np.where(stimulus.spec['sides']=='L',0,1)
 need(set(stimulus.spec['sides'])=={'L','R'},'Ambiguous antenna identity')
 def consume(owner,k):
  need(k==owner.k+1 and 0<=k-start-1<140,'Temporal input clock')
  j=k-start-1;owner.current=owner.spec['baseline']+owner.spec['delta']['profile']*table[j,indices]
  need(np.all(owner.current<=owner.caps),'No clipping')
  owner.device_rates.set(np.ascontiguousarray(owner.current));owner.errors.fill(0);owner.calls.fill(0);owner.cp.cuda.get_current_stream().synchronize();owner.k=k
  return owner.current.copy()
 stimulus.consume=types.MethodType(consume,stimulus)
 def undo():stimulus.consume=original
 return dict(schema='temporal50_external_owner_v1',arm=arm,origin_consumed_ms=start,nominal_schedule=table.tolist(),profile='Frozen694side-identifiedORN profile48',instantaneous_state_clamped=False,incoming_ORN_target_modulation_bypassed=True),undo
