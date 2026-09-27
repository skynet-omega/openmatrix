"""Frozen local PN substitution, analytic JVP, conductance and intrinsic probes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,resource,copy
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1];C48=H.parent/'etapa45_composicion_20260927_48';C49=H.parent/'etapa45_operands_20260927_49'
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def refs(d):
 if isinstance(d,dict):
  if set(d)=={'__array__'}:return [d['__array__']]
  return [r for v in d.values() for r in refs(v)]
 if isinstance(d,list):return [r for v in d for r in refs(v)]
 return []
def generic(x):return np.maximum(0,np.tanh(x))
def main():
 cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(180,181));need(not (H/'LOCAL_RESULTS.json').exists(),'Immutable result')
 nodesp=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet');nodes=pd.read_parquet(nodesp).sort_values('node_index');N=len(nodes);pnmask=nodes['class'].eq('ALPN').to_numpy();need(pnmask.sum()==686,'ALPN identity')
 sel=np.load(C49/'frozen_selection.npz');idx=sel['pre_rows'];pns=pnmask[idx];ptr=sel['ptr'];caps=sel['caps'];weights=sel['sham_cuda_weights'].astype(np.float32).astype(float)
 states={};fine={};meta={};hashes={str(nodesp):sha(nodesp),str((H/'LOCAL_CONTRACT.json').relative_to(R)):sha(H/'LOCAL_CONTRACT.json')}
 for arm in ['sham','profile']:
  d=C48/arm/'final_state';m=json.loads((d/'session.json').read_text());meta[arm]=m['hybrid']['pn_online_state'];start=N+2*len(refs(m['hybrid']['photo_ids'])) # replaced below using actual array length
  with np.load(d/'session.npz') as z:
   state=z[m['hybrid']['state']['__array__']];start=N+2*len(z[m['hybrid']['photo_ids']['__array__']]);states[arm]=dict(q=state[:N].copy(),s=state[start:start+N].copy());fine[arm]={k:z[k].copy() for k in refs(meta[arm])}
  hashes[str((d/'session.json').relative_to(R))]=sha(d/'session.json');hashes[str((d/'session.npz').relative_to(R))]=sha(d/'session.npz')
 need(meta['sham']==meta['profile'],'Same PN ownership identities and clocks; scalar state differs?')
 # This exact comparison covers refs/identity/clock, not array contents; PN charge scalars may differ.
 records={};results={};raw={}
 for arm,donor in [('sham','profile'),('profile','sham')]:
  p=C49/(arm+'_off_01')/'operands_001ms.npz';hashes[str(p.relative_to(R))]=sha(p)
  with np.load(p) as a:r=a['records'][0,:384].reshape(4,6,16)[0].copy();edge=a['operands'][0,0].astype(float);need(np.array_equal(a['pre_ids'],sel['pre_ids']),'Incoming identity')
  records[arm]=r
  delta_s=(states[donor]['s']-states[arm]['s'])[idx]
  # Exact consumed weights/caps/gates, replacing only ALPN transmission.
  term=edge[:,0]*edge[:,2]*edge[:,3]*delta_s*pns
  dx=np.array([term[ptr[j]:ptr[j+1]].sum() for j in range(6)])
  x=r[:,6]*(r[:,1]+r[:,4]-r[:,5]);t=generic(x);finite=(generic(x+r[:,6]*dx)-t)/r[:,9]
  jac=np.where(x>0,1-np.tanh(x)**2,0)*r[:,6]*dx/r[:,9]
  norm=float(np.linalg.norm(finite));err=None if norm==0 else float(np.linalg.norm(finite-jac)/norm)
  results[arm]={'delta_PN_input':dx.tolist(),'finite_RHS_delta':finite.tolist(),'analytic_Jv':jac.tolist(),'relative_error':err,'predicted_DNb05_signs_match':bool(np.array_equal(np.sign(finite[2:4]),np.sign(jac[2:4]))),'generic_equation_FP64_target_difference_from_capture':float(np.max(np.abs(t-r[:,7])))}
  raw[arm+'_record']=r;raw[arm+'_delta_input']=dx
 # Source tree swap reversibility, identity no-op and explicit exact rejection of downstream writes.
 owner_counts={};changed=[]
 for k in fine['sham']:
  a,b=fine['sham'][k],fine['profile'][k];need(a.shape==b.shape and a.dtype==b.dtype,'PN owner layout')
  if not np.array_equal(a,b):changed.append(k)
  owner_counts[k]=a.size
 # Complete-state transplant not implemented here: scalar state/queued histories require live source loader.
 r=records['sham'];E=r[:,2];I=-r[:,3];gain=r[:,6];tau=r[:,9];q=r[:,0];v=(q+.375)/2;total=1+gain*(E+I);eq=(.25+gain*E)/total;drift=(eq-v)*total/tau
 probes=[]
 for frac in [0.,.01,.1]:
  # Balanced E/I increments hold signed current constant; reversal currents need not cancel at V0.
  z=frac*(E+I)/2;e=E+z;i=I+z;tt=1+gain*(e+i);veq=(.25+gain*e)/tt
  f0=(.25-v+gain*(e*(1-v)-i*v))/tau
  vcur=v+f0*tau/total
  probes.append({'fraction':frac,'conductance_equilibrium_q':np.clip(2*veq-.375,0,1).tolist(),'matched_current_equilibrium_q':np.clip(2*vcur-.375,0,1).tolist(),'time_constant_s':(tau/tt).tolist(),'initial_RHS_match_max':float(np.max(np.abs((veq-v)*tt/tau-f0)))})
 intrinsic={}
 for arm in ['sham','profile']:
  rr=records[arm];q0=states[arm]['q'][sel['target_rows']];q=q0.copy();avg=q.copy();bias=np.zeros(6);x=rr[:,6]*(rr[:,1]+rr[:,4]-rr[:,5]);dt=.0001;hist=[]
  for k in range(500):
   q+=dt*(generic(x+bias)-q)/rr[:,9];avg+=dt*(q-avg)/.1;bias=np.clip(bias+dt*(q0-avg),-.1,.1);hist.append(bias.copy())
  bmean=np.mean(hist,axis=0);static=generic(x+bmean)+(q0-generic(x+bmean))*np.exp(-.05/rr[:,9]);parent=generic(x)+(q0-generic(x))*np.exp(-.05/rr[:,9])
  intrinsic[arm]={'q_final_dynamic':q.tolist(),'q_final_static_matched_mean_bias':static.tolist(),'q_final_no_bias':parent.tolist(),'mean_bias':bmean.tolist(),'dynamic_minus_static_max':float(np.max(np.abs(q-static))), 'DNg100_rescued':bool(np.any(q[:2]>1e-6))}
 np.savez_compressed(H/'local_projection.npz',**raw,ids=sel['target_ids'],PN_pre_mask=pns,ptr=ptr,**{arm+'_PN_transmission':states[arm]['s'][pnmask] for arm in states})
 result={'PN_JVP':results,'PN_owner_arrays':len(owner_counts),'PN_owner_changed_arrays':len(changed),'PN_transplant_status':'ARRAY_OWNERSHIP_SCREEN_ONLY; live load/no-op and scalar/queue ownership still required','conductance':{'initial_voltage_normalized':v.tolist(),'baseline_equilibrium_release':np.clip(2*eq-.375,0,1).tolist(),'initial_voltage_drift_s':drift.tolist(),'probes':probes,'interpretation':'Engineering transfer of inherited visual voltage/readout conventions to six generic cells; substantial basal drift is adoption, not smell effect'},'intrinsic':intrinsic,'source_hashes':hashes,'CPU_s':time.process_time()-cpu,'new_CNS_ms':0}
 (H/'LOCAL_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes']}))
if __name__=='__main__':main()
