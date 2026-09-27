"""Fixed two-hop anatomical projection to motor soma sides, without training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,resource,hashlib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz,diags
H=Path(__file__).resolve().parent;D=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,61));need(not (H/'ANATOMICAL_PROJECTION.json').exists(),'Immutable')
n=pd.read_parquet(D/'nodes.parquet').sort_values('node_index');C=load_npz(D/'counts_pre_post.npz').tocsr();need(C.shape==(len(n),len(n)),'Shape')
a=np.load(H/'reader_projection.npz');ids=a['ids'].ravel();rows=np.searchsorted(n.bodyId.to_numpy(),ids);need(np.array_equal(n.bodyId.to_numpy()[rows],ids),'Identity')
mid=np.flatnonzero(n.superclass.eq('vnc_intrinsic').to_numpy());mn=np.flatnonzero(n.superclass.eq('vnc_motor').to_numpy());sides=n.iloc[mn].somaSide.to_numpy();known=np.isin(sides,['L','R']);need(known.all(),'MN side metadata incomplete');total=np.asarray(C.sum(axis=1)).ravel();inv=np.divide(1.,total,out=np.zeros_like(total,dtype=float),where=total>0)
first=C[rows][:,mid].multiply(inv[rows,None]);second=C[mid][:,mn].multiply(inv[mid,None]);two=(first@second).toarray();direct=C[rows][:,mn].toarray()*inv[rows,None];A=np.stack([(two+direct)[:,sides==s].sum(axis=1) for s in ['L','R']],axis=1)
q=a['q'].reshape(4,30,10);base=q[:,(a['times_ms']>=500)&(a['times_ms']<=1000)].mean(axis=1,keepdims=True);dq=q-base
full=dq@A;dn=np.array([x=='DNb05' for x in np.repeat(a['types'],2)]);only=dq[:,:,dn]@A[dn];delta=full[:,:,1]-full[:,:,0];control=(dq@A[:,::-1]);need(np.array_equal(control[:,:,1]-control[:,:,0],-delta),'Side inversion')
np.savez_compressed(H/'anatomical_projection.npz',ids=ids,A=A,full_channels=full,DNb05_channels=only,signed=delta)
res={'scope':'Structural projection using canonical MaleCNS two-hop DN→vnc_intrinsic→vnc_motor plus direct connections, row-normalized anatomical counts. Sides are soma labels, not validated muscle actions. All intermediate neurons retained; no simulation or signed physiology. MANC type correspondence available but no cross-specimen IDs silently mixed. No learned decoder.','motor_cells':len(mn),'motor_soma_side_known':int(known.sum()),'matrix':A.tolist(),'DN_ids':ids.tolist(),'max_extra_ensemble_signal_vs_DNb05':float(np.max(np.abs((full-only)[:,:,1]-(full-only)[:,:,0]))),'effect_vs_sham_by_arm':{str(arm):float(np.max(np.abs(delta[k]-delta[0]))) for k,arm in enumerate(a['arms'])},'bilateral_symmetric_anatomical_bias':float((A[:,1]-A[:,0]).sum()),'side_relabel_exact':True,'classification':'STRUCTURAL_SIGNAL_ONLY; report ensemble minus DNb05 without presuming it is nonzero or a turning law','source_hashes':{str(D/'nodes.parquet'):sha(D/'nodes.parquet'),str(D/'counts_pre_post.npz'):sha(D/'counts_pre_post.npz')},'CPU_s':time.process_time()-cpu}
(H/'ANATOMICAL_PROJECTION.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in res.items() if k not in ['matrix','source_hashes']}))
