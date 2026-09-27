"""Export just the union of annotated groups used in the endpoint screen."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
resource.setrlimit(resource.RLIMIT_CPU,(6,8))
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
start=time.process_time();H=Path(__file__).resolve().parent
C=Path('/home/daroch/AXIOMA_ASTRA/campanas/etapa45_alternativas_20260927_51')
M=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (H/'PROYECCION.json').exists():raise ValueError('Preserve previous projection')
n=pd.read_parquet(M/'nodes.parquet',columns=['bodyId','type','superclass']).fillna('')
ids=np.load(M/'node_ids.npy',allow_pickle=False)
if not np.array_equal(ids,n.bodyId):raise ValueError('Canonical identity')
with np.load(C/'aporte_motor/JO_anatomy_arrays.npz',allow_pickle=False) as a:jo_ids=a['source_ids']
with np.load(C/'air0_odor0/neural_and_inputs.npz',allow_pickle=False) as a:observed=a['ids']
dn=n.superclass.eq('descending_neuron').to_numpy()
am=n.type.str.match(r'^(AMMC|WED)').to_numpy()
jo=np.isin(ids,jo_ids)
rows=np.flatnonzero(dn|am|jo);chosen=ids[rows]
groups=np.array(['JO_CE','all_DN','DN_with_temporal_q','DN_without_temporal_q','AMMC_WED'])
masks=np.stack([jo,dn,dn&np.isin(ids,observed),dn&~np.isin(ids,observed),am])[:,rows]
arm_names=np.array(['air0_odor0','airL_odor0','airR_odor0','airL_odor1','airR_odor1'])
endpoints=[];sources={}
for name in arm_names:
 p=C/str(name)/'neural_and_inputs.npz'
 with np.load(p,allow_pickle=False) as a:endpoints.append(a['final_q'][rows].copy())
 sources[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'projection':'final_q[canonical_rows]'}
path=H/'PROYECCION.npz'
np.savez_compressed(path,ids=chosen,canonical_rows=rows,types=n.type.to_numpy(dtype='U')[rows],
 superclass=n.superclass.to_numpy(dtype='U')[rows],observed_ids=observed,JO_ids=jo_ids,
 group_names=groups,group_masks=masks,arm_names=arm_names,final_q=np.array(endpoints))
out=dict(schema='endpoint51_group_projection_v1',artifact=path.name,sha256=sha(path),bytes=path.stat().st_size,
 original_neurons=166700,projected_neurons=len(rows),sources=sources,
 annotation_sources={str(M/f):sha(M/f) for f in ['nodes.parquet','node_ids.npy']},
 reference_result='ENDPOINTS.json',reference_sha256=sha(H/'ENDPOINTS.json'),
 scope='Only90ms endpoints and exact group membership; no CNS rerun or temporal reconstruction',
 CPU_s=time.process_time()-start,process_CPU_total_s=time.process_time(),
 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(H/'PROYECCION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['sources','annotation_sources']},indent=2))
