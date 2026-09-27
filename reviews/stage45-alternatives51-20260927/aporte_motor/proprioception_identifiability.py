"""Geometric counterexamples through the existing FeCO encode source method.

The witnesses are synthetic geometry, not fabricated Fujiwara recordings.
No model, MuJoCo, neuron, body or CUDA runtime is initialized.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,copy,hashlib,json,resource,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,25))
H=Path(__file__).resolve().parent
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
DOC=Path('/home/daroch/AXIOMA_ASTRA/investigacion/alternativas_post50_20260927_08/fuentes')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 start=time.process_time();need(not (H/'PROPRIOCEPTION.json').exists(),'Preserve result')
 source=OLD/'src/anatomical_proprioception.py';tree=ast.parse(source.read_text())
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AnatomicalProprioception')
 method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='encode')
 constants=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('MAX_CURRENT_MODEL_UNITS','VELOCITY_HALF_RAD_S') for t in x.targets)]
 need(len(constants)==2,'Source constants changed')
 # Compile the exact method and its source constants, with no module imports.
 module=ast.fix_missing_locations(ast.Module(body=[*copy.deepcopy(constants),copy.deepcopy(method)],type_ignores=[]))
 env={'np':np};exec(compile(module,str(source)+'::encode-extraction','exec'),env)
 ports=json.loads((OLD/'config/anatomical_proprioception_ports.json').read_text())['ports']
 ids=np.load(OLD/'data/male_v10/node_ids.npy',allow_pickle=False)
 sensory_ids=np.array([p['body_id'] for p in ports]);rows=np.searchsorted(ids,sensory_ids)
 need(np.array_equal(ids[rows],sensory_ids),'FeCO identity mismatch')
 legs=('LF','RF','LM','RM','LH','RH')
 sensor=SimpleNamespace(brain=SimpleNamespace(node_ids=ids),indices=rows,
  cell_types=tuple(p['type'] for p in ports),leg_indices=np.array([legs.index(p['leg']) for p in ports]),
  polarity={'claw_50':'extension','hook_39':'extension'})
 # Both limbs share projection AND segment lengths; only hidden depth differs.
 a=np.array([[-1.,0.,1.],[0.,0.,0.],[1.,1.,1.]])
 b=np.array([[-1.,0.,1.],[0.,0.,0.],[1.,1.,-1.]])
 def interior(x):
  u=x[0]-x[1];v=x[2]-x[1];axis=np.cross(u,v);axis/=np.linalg.norm(axis)
  u=u-np.dot(u,axis)*axis;v=v-np.dot(v,axis)*axis
  norm=np.linalg.norm(u)*np.linalg.norm(v)
  return np.arctan2(abs(np.dot(axis,np.cross(u,v))/norm),np.clip(np.dot(u,v)/norm,-1.,1.)),axis
 aa,na=interior(a);ab,nb=interior(b)
 need(np.array_equal(a[:,:2],b[:,:2]),'Witness projections differ')
 lengths=lambda x:np.array([np.linalg.norm(x[0]-x[1]),np.linalg.norm(x[2]-x[1])])
 need(np.array_equal(lengths(a),lengths(b)) and abs(aa-ab)>.1,'Witness does not prove ambiguity')
 angles_a=np.full(6,np.pi/2);angles_b=angles_a.copy();angles_a[0]=aa;angles_b[0]=ab
 velocity=np.zeros(6)
 changes={};signals={}
 for claw in ('extension','flexion'):
  for hook in ('extension','flexion'):
   sensor.polarity={'claw_50':claw,'hook_39':hook}
   xa=env['encode'](sensor,angles_a,velocity)['normalized_afferent_drive']
   xb=env['encode'](sensor,angles_b,velocity)['normalized_afferent_drive']
   key=claw+'_'+hook
   changes[key]=dict(changed_ports=int(np.count_nonzero(xa!=xb)),maximum_difference=float(np.max(np.abs(xa-xb))))
   need(changes[key]['changed_ports']>0,'FeCO did not distinguish physical witnesses')
   signals[key+'_a']=xa;signals[key+'_b']=xb
 rejected={}
 for name,angles,vel in [('three_left_angles_only',np.zeros(3),np.zeros(3)),
                         ('cycle_phase_used_as_joint_angle',np.full(6,1.5*np.pi),np.zeros(6))]:
  try:env['encode'](sensor,angles,vel)
  except ValueError as exc:rejected[name]=str(exc)
  else:raise ValueError('Semantically invalid geometry accepted: '+name)
 receipts=json.loads((DOC/'FUJIWARA_SOURCE_RECEIPT.json').read_text())
 source_files={source.name:sha(source),'anatomical_proprioception_ports.json':sha(OLD/'config/anatomical_proprioception_ports.json')}
 for r in receipts:
  name='fujiwara_'+r['name'];path=DOC/name
  need(sha(path)==r['sha256'],'Changed Fujiwara source '+name);source_files[name]=r['sha256']
 np.savez_compressed(H/'proprioception_witnesses.npz',projection=a[:,:2],limb_a=a,limb_b=b,
  hinge_a=na,hinge_b=nb,angles_a=angles_a,angles_b=angles_b,velocity=velocity,port_ids=sensory_ids,**signals)
 result=dict(scope='Identifiability test using documented observable schema and exact current encoder method; no raw biological replay',
  status='NONIDENTIFIABLE_FOR_EXISTING_3D_FeCO_PORT',
  source_observables=['Three left-leg cycle phases','2D FT/TT landmarks','Treadmill velocity','HS membrane potential'],
  required_observables=['Six interior femur-tibia angles','Six signed angular velocities','3D hinge geometry or justified calibrated planar constraint'],
  hidden_depth_witness=dict(identical_projected_coordinates=True,identical_segment_lengths=True,
    interior_angles_deg=np.rad2deg([aa,ab]).tolist(),encoder_disagreements=changes),
  rejected_inputs=rejected,ports=len(ports),code_method_sha256=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest(),
  source_hashes=source_files,
  limits=['Unknown right-leg motion cannot be filled with an inferred tripod phase as measured data.',
   'Cycle phase is not an absolute joint angle.',
   'A declared planar mechanical model would be a new engineering hypothesis, not recovered 3D data.',
   'Fujiwara ascending-neuron AN terminology is not the antennal entryNerve=AN annotation.'],
  CPU_s=time.process_time()-start,process_CPU_total_s=time.process_time(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  CNS_steps=0,body_steps=0,GPU_calls=0,raw_data_downloaded=False,download_bytes=0)
 (H/'PROPRIOCEPTION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False))
if __name__=='__main__':main()
