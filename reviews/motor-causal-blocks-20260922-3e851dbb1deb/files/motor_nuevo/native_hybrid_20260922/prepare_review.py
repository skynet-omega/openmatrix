from pathlib import Path
import ast,shutil,json,hashlib,sys,platform
H=Path(__file__).resolve().parent;ROOT=H.parents[1];P=ROOT/'campanas/etapa3_motor_nuevo_20260922';OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix');source=OLD/'src'
legacy=H/'legacy_sources';legacy.mkdir(exist_ok=True);todo=['coefficient_buffer_brain.py','execution_parent_snapshot.py','kc_spatial_brain.py','visuomotor_session.py','matrix_olfactory_diagnostic.py'];seen=set();provenance={}
while todo:
 name=todo.pop()
 if name in seen or not (source/name).is_file():continue
 seen.add(name);file=source/name;data=file.read_bytes();(legacy/name).write_bytes(data)
 provenance['legacy_sources/'+name]={'source':str(file),'sha256':hashlib.sha256(data).hexdigest()}
 for node in ast.walk(ast.parse(data.decode('utf-8'))):
  names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
  todo.extend(n.split('.')[0]+'.py' for n in names)
for name in ('kc_fused_step.py','kc_fused_warp.py','kc_adaptive.py'):
 f=OLD/'work/motor13_20260922'/name;provenance['vendor/'+name]={'source':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
f=source/'session_io.py';provenance['vendor/session_io.py']={'source':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(H/'SOURCES.json').write_text(json.dumps(provenance,indent=2)+'\n')
import numpy,scipy,cupy
(H/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,'cupy':cupy.__version__,'cuda_runtime':cupy.cuda.runtime.runtimeGetVersion(),'gpu':cupy.cuda.runtime.getDeviceProperties(0)['name'].decode(),'resource_scope':'One local RTX 4070 Ti SUPER, host CPU; no Codex agents'},indent=2)+'\n')
known=sum(json.loads((H/n/'RESULT.json').read_text())['wall_total_s'] for n in ('baseline_01','native_01','compressed_01','ros_01','certified_01','certified_02','aligned_01'))
(H/'BUDGET_RECEIPT.json').write_text(json.dumps({'scientific_arrivals_conservative':15,'counting':'Seven whole-organism benchmark invocations plus four recovery checks with two load/run branches each, including failed branches. Maximum16. No third complete prototype.','benchmark_process_wall_s':known,'recovery_process_wall_upper_bound_s':4*240,'scientific_process_total_upper_bound_s':known+4*240,'operator_check_invocations_including_debug_failures_upper_bound_before_clean':13,'clean_check_invocations_reserved':3,'aggregate_process_wall_upper_bound_with_checks_s':known+4*240+13*240+3*120,'aggregate_budget_s':5400,'stop_reason':'Two complete numerical prototypes compared; generic fast+faithful target not achieved; unsafe cold-recovery retry remains rejected. Preserved negative evidence and requested external review.'},indent=2)+'\n')
files=[];allowed={'.py','.cpp','.cu','.md','.json','.jsonl','.npz','.npy','.sha256','.log','.txt'}
# Include scientific receipts/frozen source versions, not binaries or prior
# administrative context, caches, prepared publication copies or clean builds.
exclude={'continuity_before','__pycache__','clean_verification','prepared'}
for f in H.rglob('*'):
 rel=f.relative_to(H)
 if f.is_file() and f.suffix in allowed and not any(x in exclude for x in rel.parts) and f.name not in ('PUBLICATION_MANIFEST.json','PUBLICATION.json','DELIVERY.json','CLEAN_CHECK.json'):
  files.append({'source':str(f),'destination':str(f.relative_to(ROOT))})
for f in P.iterdir():
 if f.is_file() and f.suffix in ('.py','.cpp'):files.append({'source':str(f),'destination':str(f.relative_to(ROOT))})
for f in (P/'verification_vendor').rglob('*'):
 if f.is_file() and f.suffix in ('.py','.json'):files.append({'source':str(f),'destination':str(f.relative_to(ROOT))})
for name in ('transport_resident_left_01','transport_reference_left_01'):
 for file in ('brain_final.json','brain_final.npz','traces.npz','body_final.npz'):
  f=P/name/file;files.append({'source':str(f),'destination':str(f.relative_to(ROOT))})
for base in (ROOT/'motor_nuevo/resident_pn_20260922',ROOT/'motor_nuevo/pn_abc_20260922'):
 for f in base.rglob('*.py'):
  if len(f.relative_to(base).parts)>2 or 'sources' in f.parts:continue
  files.append({'source':str(f),'destination':str(f.relative_to(ROOT))})
# Fine sham reference is already explicitly authorized scientific evidence.
for file in ('brain_final.json','brain_final.npz','traces.npz','body_final.npz'):
 f=OLD/'runs/motor14_20260922/event_reference_fine'/file;files.append({'source':str(f),'destination':'references/sham_fine/'+file})
files=sorted({x['destination']:x for x in files}.values(),key=lambda x:x['destination'])
spec={'label':'motor-real-native-20260922','description':'Measured actual-organism bottlenecks; A provisional 1.51x, B rejected, late-event correction slower, cold recovery discrepancy preserved. Not an efficient or stable final motor. Sources, short numerical arrays and portable checks; full static organism assets not distributed.','files':files}
(H/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'files':len(files),'legacy_sources':len(seen),'bytes':sum(Path(x['source']).stat().st_size for x in files),'known_process_wall_s':known}))
