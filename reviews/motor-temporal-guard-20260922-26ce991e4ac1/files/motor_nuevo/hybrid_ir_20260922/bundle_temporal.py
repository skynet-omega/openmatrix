"""Small review/repair addendum; the original measured workload remains immutable."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,os
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
names=['README.md','CHATGPT_FINAL_REVIEW.md','CHATGPT_RECEIPT.json','CHATGPT_ARCHITECTURE.md','TEMPORAL_GUARD_PLAN.md','scoped_graph.py','scoped_graph_before_temporal_guard.py','dependency_ir.py','model_event_contract.py','check_temporal_fallback.py','TEMPORAL_FALLBACK_CHECK.json','check_late_event_cuda.py','verify_temporal.py','bundle_temporal.py','run_scoped.py','RUNNER_CLI_REPAIR.md','RUNNER_CLI_CHECK.json','BUDGET_USED.json','ENVIRONMENT.json','late_event_cuda_01/RESULT.json','late_event_cuda_01/states.npz','late_event_cuda_01.log','DELIVERY.json','REMOTE_CHECK.json']
files=[R/x for x in names]+[ROOT/x for x in ('campanas/etapa3_motor_nuevo_20260922/graph_core.py','campanas/etapa3_motor_nuevo_20260922/event_ports.py','motor_nuevo/macro_abc_20260922/reset_ports.py','motor_nuevo/native_hybrid_20260922/graph_control_v2.cpp')]
payload=R/'temporal_sources';payload.mkdir(exist_ok=False);rows=[]
for source in files:
 relative=source.relative_to(ROOT);dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());rows.append({'source':str(dest),'destination':str(relative)})
spec={'label':'motor-temporal-guard-20260922','description':'External review found a late-event false negative; reproduced with actual C++/CUDA. C0-only crossing is not promoted; current wrapper retains event cuts. Earlier2.08x whole-body measurement remains experimental. Complete small GPU reproducer, verified errors and corrected runner included; no extra organism campaign.','files':rows}
(R/'TEMPORAL_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');folder,manifest=publish.prepare(spec,R/'temporal_prepared');meta=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_TEMPORAL_GUARD_2026-09-22.zip')
with archive.open('xb') as f:
 for part in meta['parts']:f.write((folder/part['path']).read_bytes())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:raise ValueError('Archive identity')
clean=R/'temporal_clean';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
env=os.environ.copy();env.pop('PYTHONPATH',None)
with (R/'TEMPORAL_CLEAN.log').open('x') as log:
 subprocess.run([sys.executable,'-B',str(clean/R.relative_to(ROOT)/'verify_temporal.py')],cwd=clean,env=env,stdout=log,stderr=log,timeout=90,check=True)
result={'archive':str(archive),'sha256':meta['archive_sha256'],'bytes':archive.stat().st_size,'files':len(manifest['files']),'clean_native_cuda_reproduction':True,'extra_organism_loads':0,'original_experiment_commit':'4df0dbc0247bc073a8ace8cc2886f2c1f39e1c33'}
(R/'TEMPORAL_DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
