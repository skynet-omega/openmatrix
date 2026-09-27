"""Full-network endpoint coverage check; no interpolation, fitting or ranking."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
resource.setrlimit(resource.RLIMIT_CPU,(8,10))
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
H=Path(__file__).resolve().parent
C=Path('/home/daroch/AXIOMA_ASTRA/campanas/etapa45_alternativas_20260927_51')
M=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
start=time.process_time()
if (H/'ENDPOINTS.json').exists():raise ValueError('Preserve previous result')
nodes=pd.read_parquet(M/'nodes.parquet',columns=['bodyId','type','superclass']).fillna('')
ids=np.load(M/'node_ids.npy',allow_pickle=False)
if not np.array_equal(nodes.bodyId.to_numpy(),ids):raise ValueError('Row identity')
with np.load(C/'aporte_motor/JO_anatomy_arrays.npz',allow_pickle=False) as a:jo=a['source_rows']
with np.load(C/'air0_odor0/neural_and_inputs.npz',allow_pickle=False) as a:sampled=a['ids']
groups={'JO_CE':np.isin(np.arange(len(ids)),jo),
        'all_DN':nodes.superclass.eq('descending_neuron').to_numpy(),
        'DN_with_temporal_q':nodes.superclass.eq('descending_neuron').to_numpy()&np.isin(ids,sampled),
        'DN_without_temporal_q':nodes.superclass.eq('descending_neuron').to_numpy()&~np.isin(ids,sampled),
        'AMMC_WED':nodes.type.str.match(r'^(AMMC|WED)').to_numpy()}
def final(name):
 with np.load(C/name/'neural_and_inputs.npz',allow_pickle=False) as a:
  value=a['final_q'];initial=a['initial_q']
 if value.shape!=(166700,) or not np.isfinite(value).all():raise ValueError(name)
 return value,initial
conditions={'airL_minus_air0_no_odor':('airL_odor0','air0_odor0'),
 'airR_minus_air0_no_odor':('airR_odor0','air0_odor0'),
 'airL_minus_airR_no_odor':('airL_odor0','airR_odor0'),
 'airL_minus_airR_with_odor':('airL_odor1','airR_odor1')}
contrasts={}
for label,(one,two) in conditions.items():
 a,ia=final(one);b,ib=final(two)
 if not np.array_equal(ia,ib):raise ValueError('Initial q differs')
 d=a-b;contrasts[label]={}
 for name,mask in groups.items():
  x=np.abs(d[mask]);contrasts[label][name]=dict(neurons=int(mask.sum()),max_abs_delta_q=float(x.max()),
   different_exactly=int(np.count_nonzero(x)),above_1e_9=int(np.count_nonzero(x>1e-9)),
   above_1e_6=int(np.count_nonzero(x>1e-6)),above_1e_4=int(np.count_nonzero(x>1e-4)))
out=dict(scope='Endpoint only at90ms; thresholds descriptive, not numerical or biological gates',
 observable='Heterogeneous q model outputs; no Vm/Hz equivalence; no ordering by strongest neuron',
 groups={k:int(v.sum()) for k,v in groups.items()},contrasts=contrasts,
 new_CNS_steps=0,new_GPU_calls=0,CPU_s=time.process_time()-start,
 process_CPU_total_s=time.process_time(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(H/'ENDPOINTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
