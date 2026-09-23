"""Declared direct-q reads of the pinned complete legacy coefficient program.

All other live terms read transmissions/extra states/epoch inputs, not a source
q directly. This small IR is an event dependency bridge, not full equation IR.
"""
from pathlib import Path
import hashlib,inspect,numpy as np
from dependency_ir import EventProgram,Read

def declare(brain,event_rows):
 b=brain;n=b.brain.n_neurons;N=len(b.state);tr=b.transmission_start
 def host(x):return x.get() if hasattr(x,'get') else np.asarray(x)
 reads=[Read('own_coordinate',np.arange(N),np.arange(N)),Read('release_to_filter',np.arange(n),tr+np.arange(n))]
 # Own-coordinate dependence is inactive on all source-owned q/s and other frozen states.
 zero=list(np.r_[event_rows,tr+event_rows])
 # Other direct-q readers, from the full coefficient source chain:
 pp=host(b._pvlp_cuda_rows).astype(np.int64)
 reads.append(Read('adaptation_drive',pp,b.inherited_state_size+np.arange(len(pp))))
 rows=host(b._orn_pn_cuda['rows']).astype(np.int64);begin=b.parent_state_size
 if b.olfactory_endogenous_manifest['enabled']:
  for k in range(4):reads.append(Read('resource_kinetics_'+str(k),rows,begin+k*len(rows)+np.arange(len(rows))))
 if b.pnkc_receptor_manifest['enabled']:
  rows=host(b._pnkc_cuda['source_rows']).astype(np.int64)
  import pnkc_receptor_brain
  reads.append(Read('receptor_kinetics',rows,pnkc_receptor_brain.PARENT_STATE_SIZE+np.arange(len(rows))))
 if b.regional_manifest['enabled']:
  rows=host(b._regional_cuda['kc_rows']).astype(np.int64);start=b.regional_parent_state_size
  reads.append(Read('compartment_soma',rows,start+np.arange(len(rows))))
  if b.kc_axonal_manifest['enabled']:zero.extend(np.r_[start+np.arange(len(rows)),start+2*len(rows)+np.arange(len(rows))])
 if b.kc_apl_dynamic_manifest['enabled']:
  zero.extend(host(b._dynamic_gpu_rows));zero.extend(tr+host(b._apl_gpu_rows))
 program=EventProgram(N,reads,np.unique(zero).astype(np.int64))
 # Pin each declared implementation to the reviewed local donor copy. A code
 # change invalidates this model adapter instead of inheriting a smooth flag.
 import gpu_coefficient_layout,coefficient_buffer_brain
 modules=[gpu_coefficient_layout,coefficient_buffer_brain]+[getattr(gpu_coefficient_layout,'_m'+str(i)) for i in range(17)]
 donor=Path(__file__).resolve().parent.parent/'native_hybrid_20260922/legacy_sources';sources={}
 for module in modules:
  path=Path(inspect.getsourcefile(module));expected=donor/path.name
  if not expected.exists() or path.read_bytes()!=expected.read_bytes():raise ValueError('Unreviewed coefficient source: '+path.name)
  sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 result=program.classify(event_rows);result.update(identity=program.identity(),sources=sources,jump_rows=len(event_rows),scope='Pinned complete target/rate operator. Does not include source membrane RHS, mass, delays or body in this certificate.')
 return program,result
