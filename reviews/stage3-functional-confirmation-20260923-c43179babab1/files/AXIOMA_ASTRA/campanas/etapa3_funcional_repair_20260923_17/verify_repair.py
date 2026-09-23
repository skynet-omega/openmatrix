"""Read-only raw verification of the repaired cohort, with explicit interruption cost."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'etapa3_funcional_20260923_16'
NATIVE=HERE.parent/'etapa3_pn629_intervention_20260923_15'
sys.path.insert(0,str(OLD))
from verify_all import verify as original_verify
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def upright(qpos,recorded):
 q=np.asarray(qpos,dtype=np.float64);u=np.asarray(recorded,dtype=np.float64)
 need(q.ndim==2 and q.shape[1]>=7 and u.shape==(len(q),),'Upright schema')
 expected=1-2*(q[:,4]**2+q[:,5]**2)
 need(np.isfinite(expected).all() and np.isfinite(u).all(),'Nonfinite upright')
 error=float(np.max(np.abs(expected-u)))
 need(error<=1e-12,'Recorded upright differs from qpos')
 return {'max_abs_recorded_vs_qpos':error,'min_reconstructed':float(expected.min())}
def sources(folder,kind,expected,original_plan_sha):
 c=js(folder/('RUN_CONTRACT.json' if kind=='continuation' else 'EXPERIMENT_CONTRACT.json'))
 need(c['sources']==expected,'Output source map differs from frozen map')
 need(c['plan_sha256']==original_plan_sha,'Output contract differs from original prospective plan')
 if kind=='continuation':
  need(c['engine']=='reference_cuda' and c['odor']=='odor_right' and c['first_step']==101 and c['last_step']==400,'Continuation contract profile/interval differs')
  need(c['field_reinstalled'] is False and c['baseline_reset'] is False,'Continuation modified field/baseline')
  need(c['checkpoint_manifest_sha256']==sha(OLD/'reference_odor_right_01/state_100ms/MANIFEST.json'),'Continuation source checkpoint differs')
 return True
def verify():
 frozen=js(HERE/'REPAIR_SOURCES.json')
 for path,digest in frozen.items():need(sha(path)==digest,'Frozen repair input changed: '+path)
 contract=js(HERE/'REPAIR_CONTRACT.json');oldplan=sha(OLD/'PLAN.json');budget=contract['budget']
 need(sha(HERE/'PLAN.json')==oldplan==contract['original_contract_sha256'],'Changed original plan')
 interrupted=js(OLD/'INTERRUPTION.json');need(interrupted['original_contract_unmet'] is True and interrupted['uniform_result_exists'] is False,'Missing external interruption')
 need(sha(OLD/'reference_uniform_01/PROGRESS.jsonl')==interrupted['uniform_progress_sha256'],'Interrupted prefix changed')
 need(sha(OLD/'reference_uniform_01/prepared_state/MANIFEST.json')==interrupted['uniform_prepared_manifest_sha256'],'Interrupted preparation changed')
 need(sha(OLD/'QUEUE_AT_INTERRUPTION.json')==interrupted['queue_sha256_before_edit'],'Old queue changed')
 need(js(OLD/'QUEUE.json')['state']=='INTERRUPTED_EXTERNAL_PROCESS','Old campaign still running')
 extref=js(OLD/'EXECUTION_SOURCES.json');extwithdraw=js(OLD/'WITHDRAWAL_SOURCES.json');extresume=js(OLD/'CONTINUATION_SOURCES.json')
 checks={}
 for arm in ('sham','odor_left','odor_right','uniform'):
  folder=HERE/('reference_'+arm+'_01')
  need(folder.is_dir(),'Missing complete reference: '+arm)
  if arm!='uniform':need(folder.is_symlink() and folder.resolve()==(OLD/('reference_'+arm+'_01')).resolve(),'Prior reference redirect differs')
  sources(folder,'reference',extref,oldplan)
  with np.load(folder/'traces.npz',allow_pickle=False) as z:checks['reference_'+arm]=upright(z['qpos'],z['upright'])
  with np.load(NATIVE/('full_'+arm+'_01')/'traces.npz',allow_pickle=False) as z:checks['native_'+arm]=upright(z['qpos'],z['upright'])
 for arm in ('odor_left','odor_right'):
  folder=HERE/('withdrawal_'+arm+'_01');sources(folder,'withdrawal',extwithdraw,oldplan)
  with np.load(folder/'traces.npz',allow_pickle=False) as z:checks['withdrawal_'+arm]=upright(z['qpos'],z['upright'])
 resume=HERE/'continuation_right_01';sources(resume,'continuation',extresume,oldplan)
 with np.load(resume/'traces.npz',allow_pickle=False) as z:checks['continuation_right']={'samples':len(z['qpos']),'min_reconstructed':float((1-2*(z['qpos'][:,4]**2+z['qpos'][:,5]**2)).min())}
 need(checks['continuation_right']['samples']==300,'Continuation samples missing')
 original=original_verify(HERE,NATIVE)
 need(original['classification']=='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA' and original['functional_stage3_pass'] is True,'Underlying raw verifier did not qualify')
 partial=float(interrupted['partial_last_elapsed_s']);complete=float(original['organism_wall_s']);total=partial+complete
 need(abs(partial-300.977753877989)<1e-9,'Partial elapsed evidence differs')
 need(total<=budget['aggregate_wall_including_partial_s_max'],'Historical aggregate wall budget exceeded')
 need(original['organism_runs']==budget['total_complete_runs_max']==7 and budget['one_interrupted_partial_attempt']==1 and budget['total_attempts_max']==8,'Attempt accounting changed')
 q=js(HERE/'REPAIR_QUEUE.json');need(q['state']=='EXPERIMENTS_COMPLETE_REQUIRES_RAW_VERIFICATION' and q['attempts_started']==8,'Repair queue state differs')
 # The completed native/references flow and physical support are reconstructed
 # by the original verifier. This addendum checks missing provenance/posture.
 return {'schema':'stage3_repair_raw_verified_v1','classification':'CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
  'functional_stage3_pass':True,'original_campaign_status':'BLOQUEADO_POR_INTERRUPCION_EXTERNA','strict_original_contract_fulfilled':False,
  'attempts_started':8,'complete_organism_runs':7,'partial_attempts':1,'complete_wall_s':complete,'partial_observed_wall_s':partial,
  'aggregate_observed_wall_s':total,'aggregate_limit_s':budget['aggregate_wall_including_partial_s_max'],
  'source_maps_and_continuation_contract_rechecked':True,'upright_from_qpos':checks,'original_raw_result':original,
  'limitations':['Three prior references were already exposed; this is a disclosed repair, not a new blind cohort.',
   'Historical full hidden-state KC gate remains FAIL.',
   'GPU peak across the original three references was not measured; the 12GiB device budget cannot be retrospectively certified.',
   'Body uses motorized contact rollers and an engineered DNb05 decoder; no biological gait or DNb05 gain equivalence.',
   'Fixed lateral binary field cannot demonstrate navigation toward a localized source.',
   'The general intelligence objective is not established by this functional orientation gate.'],
  'stage4_completed':False,'contract_sha256':sha(HERE/'REPAIR_CONTRACT.json'),'verifier_sha256':sha(__file__)}
if __name__=='__main__':
 result=verify();print(json.dumps({k:v for k,v in result.items() if k not in ('original_raw_result','upright_from_qpos')},indent=2))
