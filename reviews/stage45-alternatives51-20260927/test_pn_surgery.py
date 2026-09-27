"""CPU serialized-owner surgery, no CNS propagation or live source load claim."""
from pathlib import Path
import json,time,copy,hashlib,resource
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;C=H.parent/'etapa45_composicion_20260927_48'
def need(x,m):
 if not x:raise ValueError(m)
def refpaths(x,p=()):
 if isinstance(x,dict):
  if set(x)=={'__array__'}:return [(p,x['__array__'])]
  return [a for k,v in x.items() for a in refpaths(v,p+(k,))]
 if isinstance(x,list):return [a for k,v in enumerate(x) for a in refpaths(v,p+(k,))]
 return []
def digest(a):return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,61));need(not (H/'PN_SURGERY.json').exists(),'Immutable')
n=pd.read_parquet('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet').sort_values('node_index');pn=np.flatnonzero(n['class'].eq('ALPN'));N=len(n);donors={};metas={};states={}
for arm in ['sham','profile']:
 j=json.loads((C/arm/'final_state/session.json').read_text());h=j['hybrid'];metas[arm]=h['pn_online_state'];paths=refpaths(h['pn_online_state'])
 with np.load(C/arm/'final_state/session.npz') as a:
  states[arm]=a[h['state']['__array__']].copy();donors[arm]={p:a[k].copy() for p,k in paths};start=N+2*len(a[h['photo_ids']['__array__']])
selected=np.r_[pn,start+pn];other=np.ones(len(states['sham']),bool);other[selected]=False
results=[];payload=dict(selected_indices=selected,PN_ids=n.bodyId.to_numpy()[pn])
for receiver,source in [('sham','sham'),('profile','profile'),('sham','profile'),('profile','sham')]:
 x=states[receiver].copy();before=digest(x[other]);x[selected]=states[source][selected]
 own=copy.deepcopy(metas[source]);arr={p:v.copy() for p,v in donors[source].items()}
 need(digest(x[other])==before,'Downstream changed by surgery');need(np.array_equal(x[selected],states[source][selected]),'Source state mismatch')
 no_op=receiver==source
 if no_op:need(np.array_equal(x,states[receiver]) and own==metas[receiver] and all(np.array_equal(arr[p],v) for p,v in donors[receiver].items()),'No-op differs')
 restored=x.copy();restored[selected]=states[receiver][selected];need(np.array_equal(restored,states[receiver]),'Reversal failed')
 changed=copy.deepcopy(x);changed[np.flatnonzero(other)[0]]+=.125;rejected=digest(changed[other])!=before;need(rejected,'Downstream corruption not detected')
 results.append(dict(receiver=receiver,source=source,canonical_ALPNs=len(pn),canonical_dynamic_slots=len(selected),fine_PN_arrays=len(arr),all_nonselected_hybrid_slots_exact=True,no_op=no_op,reversal_exact=True,downstream_corruption_rejected=True,nonselected_sha256=before))
 payload[receiver+'_from_'+source]=x[selected]
np.savez_compressed(H/'pn_surgery_projection.npz',**payload)
r={'scope':'Executed reciprocal serialized CPU transplant/no-op/reversal of all686ALPN q+outgoing common filters plus whole finePN37array/scalar owner tree. No live source loader or recurrent CNS evolution executed; no causal neural-effect claim.','ownership_limit':'Fine-PN owner also contains its input receptors and source output tails. Presynaptic terminal filters in generic hybrid are copied; downstream DN state is kept. External pending events/caches outside serialized PN tree not qualified here, so not yet a complete operational transplant.','checks':results,'CPU_s':time.process_time()-cpu,'classification':'SERIALIZED_OWNER_SURGERY_PASS; LIVE_TRANSPLANT_PENDING'}
(H/'PN_SURGERY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
