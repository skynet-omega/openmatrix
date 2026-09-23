"""Descriptive spatial sensitivity on saved poses; never a neural/body replay."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=ROOT/'campanas/etapa3_pn629_intervention_20260923_15'
GEOMETRY=HERE/'chatgpt_cpu_geometry_01/RESULTADO.json'
RAW_GEOMETRY=ROOT/'intercambio/stage4_design_20260923_verified/design/GEOMETRY.json'
TRACE_HASHES={
 'sham':'097ee4ef5501bcf6df296c3d2f0b1c108e738da46d2e0cf779cfd3b2359ec5b8',
 'odor_left':'b47a8ad07b8b2f8fd1ac0e59f175d762487f2989a3c476fee65a00130cc8fd56',
 'odor_right':'82b0bf25b2de5d019056be1f0b9bac6e25d2ea45b5f2fa29e84b2a8c71d515e3',
 'uniform':'e9817f978a629ca64b0967eb2850cb259e33decbba7225588ffe8571f0566615'}
GEOMETRY_HASH='a851f2d6f56ebdba33c78da03d43484bf74b113c5b61334d6161026fe7f095a9'
RAW_GEOMETRY_HASH='3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def gaussian(x,source,sigma):
 return np.exp(-np.sum((x-np.asarray(source))**2,axis=-1)/(2*sigma*sigma))
def main():
 start=time.perf_counter();cpu=time.process_time()
 plan=json.loads((HERE/'OBSERVABILITY_PLAN.json').read_text())
 need(sha(GEOMETRY)==GEOMETRY_HASH and sha(RAW_GEOMETRY)==RAW_GEOMETRY_HASH,'Geometry source changed')
 geom=json.loads(GEOMETRY.read_text())
 raw_geometry=json.loads(RAW_GEOMETRY.read_text())
 need(geom['status']=='CPU_GEOMETRY_COMPLETE' and not geom['stage4_admission'],'Unexpected design result')
 sources={k:v for k,v in geom['sources'].items() if k in ('donor','transfer')}
 need(set(sources)=={'donor','transfer'},'Missing sources')
 output={'schema':plan['schema'],'classification':'DESCRIPTIVE_ONLY','input_hashes':{'geometry':GEOMETRY_HASH,'raw_geometry':RAW_GEOMETRY_HASH},
         'arms':{},'stage3_admission':False,'stage4_admission':False,
         'limits':plan['scope']}
 prep=[]
 scale=float(raw_geometry['body_native_units_per_mm'])
 need(scale>0,'Invalid body coordinate scale')
 for arm,digest in TRACE_HASHES.items():
  p=NATIVE/('full_'+arm+'_01')/'traces.npz'
  need(sha(p)==digest,'Trace hash changed: '+arm)
  with np.load(p,allow_pickle=False) as z:
   a=z['antenas_mm'].copy();q=z['qpos'][:,:2].copy();steps=z['paso'].copy();phase=z['fase'].copy()
  need(a.shape==(440,2,3) and q.shape==(440,2),'Unexpected geometry trace '+arm)
  need(np.array_equal(phase[:40],np.full(40,'preparacion')) and np.array_equal(phase[40:],np.full(400,'ensayo')),'Phase mismatch')
  need(np.array_equal(steps[40:],np.arange(1,401)),'Trial clock mismatch')
  need(np.isfinite(a).all() and np.isfinite(q).all(),'Nonfinite poses')
  prep.append(a[:40].copy())
  arm_result={'trace_sha256':digest,'source_samples':{}}
  for name,spec in sources.items():
   c=gaussian(a[:,:,:2],spec['source_mm'],float(spec['sigma_mm']))
   c0=c[39];trial=c[40:]
   body=q/scale;src=np.asarray(spec['source_mm']);dist=np.linalg.norm(body-src,axis=1)
   arm_result['source_samples'][name]={
    'prepared_L_R':c0.tolist(),
    'trial_last_L_R':trial[-1].tolist(),
    'max_abs_change_from_prepared_L_R':np.max(np.abs(trial-c0),axis=0).tolist(),
    'rms_change_from_prepared_L_R':np.sqrt(np.mean((trial-c0)**2,axis=0)).tolist(),
    'body_distance_mm_prepared':float(dist[39]),
    'body_distance_mm_trial_last':float(dist[-1]),
    'descriptive_progress_mm':float(dist[39]-dist[-1])}
   arm_result.setdefault('_series',{})[name]=trial
  donor=arm_result['_series']['donor'];transfer=arm_result['_series']['transfer']
  arm_result['same_pose_source_gap_max_L_R']=np.max(np.abs(donor-transfer),axis=0).tolist()
  arm_result['same_pose_source_gap_last_L_R']=np.abs(donor[-1]-transfer[-1]).tolist()
  del arm_result['_series']
  output['arms'][arm]=arm_result
  output['input_hashes'][arm]=digest
 need(all(np.array_equal(prep[0],v) for v in prep[1:]),'Different preparation antenna histories')
 output['preparation_antennae_identical_all_arms']=True
 output['cpu_s']=time.process_time()-cpu;output['wall_s']=time.perf_counter()-start
 output['peak_ram_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 b=plan['budget'];need(output['cpu_s']<=b['cpu_s_max'] and output['wall_s']<=b['wall_s_max'] and output['peak_ram_MiB']<=b['ram_MiB_max'],'Budget exceeded')
 output['code_sha256']=sha(__file__);output['plan_sha256']=sha(HERE/'OBSERVABILITY_PLAN.json')
 target=HERE/'pose_gaussian_sensitivity_01'
 target.mkdir(exist_ok=False)
 (target/'RESULT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'classification':output['classification'],'wall_s':output['wall_s'],'arm_keys':list(output['arms'])}))
if __name__=='__main__':main()
