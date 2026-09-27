"""Bounded CPU observation/metadata screen, no model imports or training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,resource,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

resource.setrlimit(resource.RLIMIT_CPU,(15,18))
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
H=Path(__file__).resolve().parent
A=Path('/home/daroch/AXIOMA_ASTRA')
M=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
C=A/'campanas/etapa45_alternativas_20260927_51'
start=time.process_time()
def need(v,m):
 if not v:raise ValueError(m)
def schema(path):
 records={}
 with zipfile.ZipFile(path) as z:
  for info in z.infolist():
   if not info.filename.endswith('.npy'):continue
   with z.open(info) as f:
    version=np.lib.format.read_magic(f)
    if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
    else:raise ValueError('Unexpected array header')
   records[info.filename[:-4]]=dict(shape=shape,dtype=str(dtype),compressed_bytes=info.compress_size,
     member_bytes=info.file_size,header_only=True,zip_crc32=hex(info.CRC))
 return records

need(not (H/'COBERTURA.json').exists(),'Preserve previous probe')
plan=json.loads((C/'PILOT_PLAN.json').read_text())
schemas={};states=[];parent_q=[];all_q=[];ids=None;paths=[]
for name in plan['arms']:
 folder=C/name
 need(json.loads((folder/'RESULT.json').read_text())['status']=='COMPLETE',name)
 schemas[name]={p:schema(folder/p) for p in ['neural_and_inputs.npz','traces.npz','dng100_observed.npz']}
 with np.load(folder/'neural_and_inputs.npz',allow_pickle=False) as a:
  q=a['q'];nid=a['ids'];need(q.shape==(90,16),'q shape')
  need(ids is None or np.array_equal(ids,nid),'id consistency');ids=nid
  all_q.append(q)
  if name.startswith('air'):parent_q.append(q)
  states.append(hashlib.sha256(a['initial_q'].tobytes()).hexdigest())
  paths.append({'path':str(folder/'neural_and_inputs.npz'),'bytes':(folder/'neural_and_inputs.npz').stat().st_size,
                'q_sha256':hashlib.sha256(q.tobytes()).hexdigest()})
nodes=pd.read_parquet(M/'data/male_v10/nodes.parquet',columns=['bodyId','type','rootSide','somaSide','superclass','class']).fillna('')
selected=nodes.set_index('bodyId').loc[ids].reset_index()
selected.to_csv(H/'NEURONAS_OBSERVADAS.csv',index=False)
X=np.concatenate(parent_q);allX=np.concatenate(all_q)
centered=X-X.mean(axis=0)
sv=np.linalg.svd(centered,compute_uv=False)
energy=sv**2;ratios=np.cumsum(energy)/energy.sum()
ranges=np.ptp(X,axis=0)
statistics=dict(parent_matrix_shape=X.shape,all_matrix_shape=allX.shape,
  independent_preparations_from_plan=1,unique_initial_q_arrays=len(set(states)),
  recorded_neuron_ids=ids.tolist(),ranges_per_neuron=ranges.tolist(),
  neurons_with_range_above_1e_9=int(np.count_nonzero(ranges>1e-9)),
  descriptive_threshold_not_scientific_gate=1e-9,
  raw_q_centered_singular_values=sv.tolist(),raw_q_PCA_cumulative_energy=ratios.tolist(),
  no_feature_standardization=True,interpretation='Numerical variance of selected heterogeneous q, not biological information or neural energy',
  fully_independent_validation_preparations=0)

# Check availability using original containers and receipts, without unpacking responses.
receipts=[]
for filename in ['work/additional_library_arrivals_20260915/inventory.json','work/new_dryad_arrivals_20260915/inventory.json']:
 p=M/filename;d=json.loads(p.read_text())
 receipts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'top_keys':list(d)})
suver=M/'work/fast_force_data_20260914/doi_10_5061_dryad_k06kh8f__v20190402.zip'
with zipfile.ZipFile(suver) as z:
 suver_members=[dict(name=i.filename,size=i.file_size,stored=i.compress_type==zipfile.ZIP_STORED,
                      compressed_bytes=i.compress_size,crc32=hex(i.CRC)) for i in z.infolist()]
resources=dict(suver=dict(path=str(suver),present=suver.exists(),bytes=suver.stat().st_size,
 members=suver_members,internal_7z_not_opened=True,full_archive_not_rehashed=True),receipts=receipts)
out=dict(scope='Schema of all ten arms, selected-q descriptive screen, original container availability',
 plan_sha256=hashlib.sha256((C/'PILOT_PLAN.json').read_bytes()).hexdigest(),schemas=schemas,
 statistics=statistics,resources=resources,raw_array_sources=paths,
 no_dictionary_or_predictor_fit=True,new_CNS_steps=0,new_GPU_calls=0,
 CPU_s=time.process_time()-start,process_CPU_total_s=time.process_time(),
 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(H/'COBERTURA.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(statistics=statistics,first_arm_schema=schemas['air0_odor0'],resources=resources,
 CPU_s=out['CPU_s'],process_CPU_total_s=out['process_CPU_total_s'],peak_RSS_bytes=out['peak_RSS_bytes']),indent=2,ensure_ascii=False))
