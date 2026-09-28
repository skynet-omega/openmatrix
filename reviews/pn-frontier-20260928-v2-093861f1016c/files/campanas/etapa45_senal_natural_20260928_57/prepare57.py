"""Freeze the natural spatial observation before any new CNS exposure."""
from pathlib import Path
import json,hashlib,time,shutil,ast
import numpy as np
H=Path(__file__).resolve().parent;C54=H.parent/'etapa45_fuente_cuerpo_20260928_54';C56=H.parent/'etapa45_transferencia_temporal_20260928_56'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def need(x,m):
 if not x:raise ValueError(m)
def main():
 start=time.process_time();need(not (H/'FREEZE.json').exists(),'already frozen');ref=H/'reference';ref.mkdir(exist_ok=False);provenance={}
 def cp(src,rel):
  dst=ref/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst);provenance[rel]=dict(original=str(src),sha256=sha(src));need(sha(dst)==sha(src),'copy')
 cp(C56/'reference/TERMINALS.npz','TERMINALS.npz');cp(C54/'analysis_refs/anchor.npz','anchor.npz')
 for filename in ['traces.npz','neural_and_inputs.npz','PN_consumed.npz','wide_observation.npz','dng100_observed.npz','SCIENTIFIC_WITNESS.json','EVENTS.json']:
  cp(C56/'repair01/qual_disabled'/filename,'qualification56/'+filename)
 for arm in ['parent_none','parent_L','parent_R']:
  for filename in ['traces.npz','neural_and_inputs.npz','wide_observation.npz','dng100_observed.npz','SPATIAL_OWNER_FINAL.json']:
   cp(C54/'repair02'/arm/filename,arm+'/'+filename)
 save(ref/'PROVENANCE.json',provenance)
 plan=dict(schema='natural_spatial57_v1',source_checkpoint='campanas/etapa45_composicion_20260927_48/sham/final_state',arms={'qual':dict(source='none',duration_ms=2),**{'natural_'+s:dict(source=s,duration_ms=384) for s in ['none','L','R']}},prefix_ms=10,early_window_ms=[51,89],analysis_window_ms=[257,384],planned_CNS_ms=1154,CNS_ms=1200,queue_CPU_s=6000,queue_wall_s=5000,qualification=dict(CPU_s=170,wall_s=160),per_arm=dict(CPU_s=1600,wall_s=1500),RAM_bytes=24*1024**3,VRAM_bytes=14*1024**3,new_disk_bytes=10*1024**3,preparation_CPU_s=180,analysis_delivery_CPU_s=300,independent_review_CPU_s=60,changes=dict(neural='none; parent law, weights, thresholds, reader and initial history preserved',sensors='same prescribed694ORN spatial proxy54, not validated chemical transduction; genericORN/PN witness read-only',body='same applied yaw/forward, support and rollers54; no wind',duration='extend peripheral guard3200->3384 for scientific arms only; cold-load same3000ms state within49 horizon; verify exact body recurrence everyms; portable restart remains unqualified'),criteria=dict(neural_half_L_minus_R_q_min=1.6e-5,yaw_half_L_minus_R_deg_s_min=.02,stage4=False,stage5=False,scope='Developmental diagnostic, one exposed preparation. No biological equivalence or stage admission.',screen='Material DNb and applied yaw halfL-R contrasts, positive direction-compatible contrast and improvement of source angular error vs same-source none trajectory in BOTH mirrors. Does not require forward, and does not admit stage4.',signal='Report vectors, L-R half contrast and (L+R)/2-none common component. No selected-cell subset or input-size PASS cutoff.',persistence='For each registered frontier: early51:89 mean halfL-R vector forms within-run descriptive template; late257:384 mean amplitude/norm/cosine/projection ratio; zero-denominator explicitly null. Report L-none and R-none. Same-run template is not independent confirmation and never enters organism.',frontiers=['nominal_ORN_Hz','ORN_generic_first_RHS','ORN_q_committed','PN_generic_first_RHS','PN_generic_last_RHS','PN_q_committed','DM1_filters_committed','fine_PN_KC_gamma_and_additional'],body='Compare absolute error, same-source benefit versus none, signed yaw, physical heading/displacement; absolute positive yaw alone is not correct orientation.',confounds='Anatomical ORN counts and total prescribed Hz differ with side; source contrasts do not isolate pure direction from amount. CSR output is not finePN input; specialized paths also observed. AllRHS witnesses summarize first,last,min,max,count, not full continuous tape.'),stop='qual2ms exact then3x384ms, all first89ms exact54. Any discrepancy/nonfinite/budget stops; no automatic retries or parameter tuning. Complete hito even if negative.')
 save(H/'PLAN.json',plan)
 # Preserve existing source pinning and add new execution sources/references.
 sources=json.loads((C56/'repair01/A_SOURCES.json').read_text())
 for path,d in sources.items():need(sha(path)==d,'inherited frozen file changed '+path)
 for path in list(H.glob('*.py'))+[H/'PLAN.json',H/'PROPUESTA_PROPIA.md']+list(ref.rglob('*')):
  if path.is_file():sources[str(path)]=sha(path)
 save(H/'SOURCES.json',sources)
 for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 # Clock continuation test uses independent recurrence from anchor, never resets time.
 anchor=44.486000000027154;dt=2.5e-5;t=anchor
 for _ in range(3000*40):t+=dt
 initial=t;times=[]
 for j in range(384):
  for _ in range(40):t+=dt
  times.append(t)
 direct=anchor
 for _ in range(3384*40):direct+=dt
 need(t==direct and np.all(np.diff(times)>0),'clock continuation')
 save(H/'PREPARATION.json',dict(CPU_s=time.process_time()-start,CNS_ms=0,reference_files=len(provenance),source_files=len(sources),clock_initial_time_s=initial,clock_final_time_s=t,clock_method='Exact IEEE64 repeated addition, same25us; no tolerance enlarged. Live body checked separately.',static_syntax=True))
 save(H/'FREEZE.json',dict(plan_sha256=sha(H/'PLAN.json'),sources_sha256=sha(H/'SOURCES.json'),reference_provenance_sha256=sha(ref/'PROVENANCE.json'),unix_s=time.time()))
 print(json.dumps(dict(planned_CNS_ms=1154,plan_sha256=sha(H/'PLAN.json'),CPU_s=time.process_time()-start)))
if __name__=='__main__':main()
