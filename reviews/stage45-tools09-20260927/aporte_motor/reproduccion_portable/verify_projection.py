"""Portable independent verifier: only the NPZ and JSON files beside this script."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(6,8))
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
H=Path(__file__).resolve().parent;start=time.process_time()
def need(ok,message):
 if not ok:raise ValueError(message)
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def calculate(a):
 ids=a['ids'];q=a['final_q'];types=a['types'];classes=a['superclass']
 need(ids.ndim==1 and ids.dtype.kind in 'iu' and len(np.unique(ids))==len(ids),'Unique IDs')
 need(q.shape==(5,len(ids)) and np.isfinite(q).all(),'Finite aligned endpoints')
 need(np.array_equal(a['arm_names'],['air0_odor0','airL_odor0','airR_odor0','airL_odor1','airR_odor1']),'Arm order')
 need(a['group_masks'].dtype==np.bool_,'Boolean groups')
 jo=np.isin(ids,a['JO_ids']);dn=classes=='descending_neuron';obs=np.isin(ids,a['observed_ids'])
 am=np.char.startswith(types,'AMMC')|np.char.startswith(types,'WED')
 names=['JO_CE','all_DN','DN_with_temporal_q','DN_without_temporal_q','AMMC_WED']
 masks=[jo,dn,dn&obs,dn&~obs,am]
 need(np.array_equal(a['group_names'],names) and np.array_equal(a['group_masks'],masks),'Semantic group masks')
 need(np.array_equal(jo,np.char.startswith(types,'JO-C')|np.char.startswith(types,'JO-E')),'JO annotations')
 pairs={'airL_minus_air0_no_odor':(1,0),'airR_minus_air0_no_odor':(2,0),
        'airL_minus_airR_no_odor':(1,2),'airL_minus_airR_with_odor':(3,4)}
 out={}
 for label,(i,j) in pairs.items():
  out[label]={}
  for name,mask in zip(names,masks):
   d=np.abs(q[i,mask]-q[j,mask])
   out[label][name]={'neurons':int(mask.sum()),'max_abs_delta_q':float(max(d)),
      'different_exactly':int(sum(d!=0)),'above_1e_9':int(sum(d>1e-9)),
      'above_1e_6':int(sum(d>1e-6)),'above_1e_4':int(sum(d>1e-4))}
 return out
def compare(a,expected):need(calculate(a)==expected['contrasts'],'Endpoint statistics differ')
m=json.loads((H/'PROYECCION.json').read_text())
need(hashlib.sha256((H/m['artifact']).read_bytes()).hexdigest()==m['sha256'],'Projection digest')
need(hashlib.sha256((H/m['reference_result']).read_bytes()).hexdigest()==m['reference_sha256'],'Reference digest')
a=load(H/m['artifact']);expected=json.loads((H/m['reference_result']).read_text());compare(a,expected)
caught=[]
for mutation in ['nonfinite_q','duplicate_id','wrong_group','changed_count']:
 b={k:v.copy() for k,v in a.items()};e=json.loads(json.dumps(expected))
 if mutation=='nonfinite_q':b['final_q'][0,0]=np.nan
 elif mutation=='duplicate_id':b['ids'][0]=b['ids'][1]
 elif mutation=='wrong_group':b['group_masks'][0,0]=~b['group_masks'][0,0]
 else:e['contrasts']['airL_minus_airR_no_odor']['all_DN']['above_1e_6']+=1
 try:compare(b,e)
 except ValueError:caught.append(mutation)
 else:raise ValueError('Corruption not detected: '+mutation)
out=dict(status='PASS',portable_inputs_only=True,optimization=__debug__ is False,
 recomputed=calculate(a),corruptions_detected=caught,
 source_reproduction_scope='Recompute projected raw endpoint statistics, not original CNS or biological data',
 CPU_s=time.process_time()-start,process_CPU_total_s=time.process_time(),
 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(H/'VERIFICACION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='recomputed'},indent=2))
