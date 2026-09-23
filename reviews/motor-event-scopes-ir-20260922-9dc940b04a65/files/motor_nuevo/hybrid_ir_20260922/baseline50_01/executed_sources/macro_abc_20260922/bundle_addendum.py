from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,os
R=Path(__file__).resolve().parent;ROOT=R.parents[1];P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
names=['README_ADDENDUM.md','GEMINI_CRITIQUE_APPLIED.md','GEMINI_OBSERVATION_PLAN.json','ACTIVITY_DESCRIPTIVE.json','analyze_activity.py','CHATGPT_CODE_REVIEW.md','reset_ledger.py','reset_adapter.py','reset_adapter_before_review.py','reset_ports.py','lif_event_metadata.py','LIF_PROVENANCE.json','check_ledger_bridge.py','LEDGER_BRIDGE_CHECK.json','ledger_bridge.log','bundle_addendum.py','ENVIRONMENT.json','verification_vendor/event_waveform.py']
paths=[R/name for name in names]+[P/'event_ports.py',P/'verification_vendor/compare.py',R/'guard_reset_01/OPERATOR_INVENTORY.json']
paths += [R/f'guard_reset_01/brain_{ms:02d}ms.{suffix}' for ms in (1,5,20) for suffix in ('json','npz')]
payload=R/'addendum_sources';payload.mkdir(exist_ok=False);files=[]
for source in paths:
 relative=source.relative_to(ROOT);dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());files.append({'source':str(dest),'destination':str(relative)})
spec={'label':'motor-ledger-critique-20260922','description':'Post-review ADD/SET ledger fix with real CPU-to-CUDA counterexample; Gemini criticism tested against existing snapshots. No new organism simulation or speedup. Self-contained fixtures and descriptive snapshot analysis; previous full campaign77611a3 remains immutable.','files':files}
(R/'ADDENDUM_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');folder,manifest=publish.prepare(spec,R/'addendum_prepared');a=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_LEDGER_CRITIQUE_2026-09-22.zip')
with archive.open('xb') as f:
 for part in a['parts']:f.write((folder/part['path']).read_bytes())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=a['archive_sha256']:raise ValueError('ZIP hash')
clean=R/'addendum_clean';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
for row in manifest['files']:
 if hashlib.sha256((clean/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Extracted hash')
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OPENBLAS_NUM_THREADS='1',CUPY_CACHE_DIR=str(clean/'cuda_cache'),NUMBA_CACHE_DIR=str(clean/'numba_cache'))
with (R/'ADDENDUM_CLEAN.log').open('x') as log:
 for name in ('check_ledger_bridge.py','analyze_activity.py'):
  subprocess.run([sys.executable,'-B','-O',str(clean/R.relative_to(ROOT)/name)],cwd=clean,env=env,stdout=log,stderr=log,timeout=30,check=True)
v={'archive':str(archive),'sha256':a['archive_sha256'],'bytes':archive.stat().st_size,'files':len(manifest['files']),'clean_checks':['hashes','check_ledger_bridge.py','analyze_activity.py'],'whole_organism_rerun':False}
(R/'ADDENDUM_DELIVERY.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
