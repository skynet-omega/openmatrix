from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parent;P=R.parents[1]/'campanas/etapa3_motor_nuevo_20260922';sys.path.insert(0,str(P/'verification_vendor'))
from compare import read_state
n=json.loads((R/'guard_reset_01/OPERATOR_INVENTORY.json').read_text())['base_neurons']
out=[]
for start,end in ((1,5),(5,20)):
 a=read_state(R/f'guard_reset_01/brain_{start:02d}ms');b=read_state(R/f'guard_reset_01/brain_{end:02d}ms')
 if b['time_ns']-a['time_ns']!=(end-start)*1000000:raise ValueError('Wrong physical interval')
 if not np.array_equal(a['photo_ids'],b['photo_ids']):raise ValueError('Photo layout changed')
 offset=n+2*len(a['photo_ids'])
 row={'start_ms':start,'end_ms':end,'n':n,'transmission_offset':offset,'variables':{}}
 for name,sl in (('normalized_neural_state',slice(0,n)),('synaptic_transmission',slice(offset,offset+n))):
  x,y=a['state'][sl],b['state'][sl]
  if x.shape!=(n,) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Bad state layout')
  delta=abs(y-x)
  row['variables'][name]={'positive_at_start':int(np.count_nonzero(x>0)),'positive_at_end':int(np.count_nonzero(y>0)),'changed':[{ 'threshold':t,'count':int(np.count_nonzero(delta>t)),'fraction':float(np.mean(delta>t))} for t in (0.,1e-12,1e-6,1e-4)],'maximum_delta':float(delta.max())}
 out.append(row)
result={'intervals':out,'source_snapshots':'guard_reset_01','scope':'Endpoint activity only: not computational sparsity, passive-flow test or an omitted-error bound. No assumption that a nonspiking neuron contributes zero.'}
(R/'ACTIVITY_DESCRIPTIVE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
