"""Prespecified population readers on complete published time series; no fit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,resource
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1];C=H.parent/'etapa45_composicion_20260927_48'
TYPES=['DNa01','DNa02','DNb05','DNb06','DNg13'];ARMS=['sham','dm1','profile','permuted']
PAIR_IDS=np.array([[10442,10760],[523769,10360],[10118,10065],[10888,11067],[11074,512006]])
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def calculate(a):
 q=a['q'];times=a['times_ms'];base=q[:,(times>=500)&(times<=1000)].mean(axis=1,keepdims=True)
 delta=(q-base);signed=(delta[:,:,:,0]-delta[:,:,:,1])*np.array([1,1,1,-1,1])
 return {'five_pair_median':np.median(signed,axis=2),'five_pair_equal_mean':signed.mean(axis=2),
 'DNb05_DNb06_opponent':signed[:,:,2:4].mean(axis=2),'DNa02_DNg13_actions':signed[:,:,[1,4]].mean(axis=2)}
def main():
 cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,61));need(not (H/'READERS.json').exists(),'Immutable')
 nodesp=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet');nodes=pd.read_parquet(nodesp).sort_values('node_index');ids=nodes.bodyId.to_numpy();idx=np.searchsorted(ids,PAIR_IDS);need(np.array_equal(ids[idx],PAIR_IDS),'Identities')
 brainp=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain/state.npz')
 with np.load(brainp) as z:need(np.array_equal(ids,z['node_ids']),'Caps identity');caps=z['r_max'].astype(float)
 hashes={str(nodesp):sha(nodesp),str(brainp):sha(brainp)};qs=[];ts=[]
 for arm in ARMS:
  qq=[];tt=[]
  for p in sorted((C/arm/'blocks').glob('*/published.npz')):
   m=json.loads((p.parent/'MANIFEST.json').read_text());entry=m.get('files',m).get('published.npz');actual=sha(p)
   if isinstance(entry,dict):entry=entry.get('sha256')
   if entry is not None:need(entry==actual,'Publication hash')
   hashes[str(p.relative_to(R))]=actual
   with np.load(p) as z:qq.append(z['output'][idx]/caps[idx]);tt.append(int(p.parent.name.split('_')[-1].removesuffix('ms')))
  need(len(tt)==30,'30 endpoints');ts.append(tt);qs.append(qq)
 need(all(x==ts[0] for x in ts),'Aligned times');q=np.array(qs);need(np.isfinite(q).all(),'Finite')
 a={'q':q,'times_ms':np.array(ts[0]),'ids':PAIR_IDS,'caps':caps[idx],'types':np.array(TYPES),'arms':np.array(ARMS)}
 out=calculate(a);np.savez_compressed(H/'reader_projection.npz',**a,**out)
 summary={name:{arm:{'max_abs_effect_vs_sham':float(np.max(np.abs(v[k]-v[0]))),'mean_final_effect_vs_sham':float(np.mean((v[k]-v[0])[a['times_ms']>=2500])),'max_abs_own_baseline_change':float(np.max(np.abs(v[k])))} for k,arm in enumerate(ARMS)} for name,v in out.items()}
 result={'scope':'Readers evaluated on 100ms published endpoints divided by unchanged caps; not native FP64 q, not calcium; bilateral input so no directional success claim. Baseline500..1000ms per arm. No new CNS.','selection':'Five pairs and signs fixed in review08/Yang; means are transparent equal-weight controls, no fit.','readers':summary,'sampled_activity_max_by_type':{t:float(q[:,:,k,:].max()) for k,t in enumerate(TYPES)},'source_hashes':hashes,'CPU_s':time.process_time()-cpu,'classification':'DESCRIPTIVE_SCREEN_COMPLETE'}
 (H/'READERS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['readers','sampled_activity_max_by_type','CPU_s']}))
if __name__=='__main__':main()
