"""Explicit evidence sources; immutable ZIP and isolated short reproduction."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,os,shutil
R=Path(__file__).resolve().parent;ROOT=R.parents[1];M=R.parent;P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
selected={}
def add(p):
 if p.is_file() and p.suffix in publish.EXT:selected[str(p.relative_to(ROOT))]=p
skip={'PUBLICATION.json','PUBLICATION_MANIFEST.json','DELIVERY.json','CLEAN_CHECK.log','REMOTE_CHECK.json','CHATGPT_DELIVERY.json'}
for p in R.iterdir():
 if p.is_file() and p.name not in skip:add(p)
for name in ('jev_01','external_cpu_01','ir_bridge_01','legacy_runtime'):
 for p in (R/name).iterdir():add(p)
for name in ('baseline20_01','scoped20_01','baseline50_01','scoped50_01'):
 for p in (R/name).iterdir():
  if p.is_file():add(p)
 for sub in ('executed_sources','inputs'):
  for p in (R/name/sub).rglob('*'):add(p)
for base in (P,M/'pipeline_review_20260922',M/'native_hybrid_20260922',M/'causal_runtime_20260922',M/'resident_pn_20260922'):
 for p in base.iterdir():
  if p.suffix in ('.py','.cpp','.cu','.hpp') and p.name not in ('package.py','bundle_review.py'):add(p)
for base in (P/'verification_vendor',M/'native_hybrid_20260922/vendor',M/'native_hybrid_20260922/legacy_sources'):
 for p in base.iterdir():add(p)
for name in ('model.py','autodiff.py'):add(M/'general_v2_20260922'/name)
for name in ('reset_adapter.py','reset_ports.py','reset_ledger.py','lif_event_metadata.py','operator_inventory.py'):
 add(M/'macro_abc_20260922'/name)
add(M/'transient_localization_20260922/compare_candidates.py')
add(M/'native_hybrid_20260922/PETSC_LICENSE.txt');add(ROOT/'instrumentos/openmatrix/publish.py')
payload=R/'publication_sources';payload.mkdir(exist_ok=False);files=[]
for relative,source in sorted(selected.items()):
 dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());files.append({'source':str(dest),'destination':relative})
spec={'label':'motor-event-scopes-ir-20260922','description':'Complete-organism20/50ms event-scope comparisons:2.06/2.08x faster, original numerical short criteria retained. ChatGPT executable CPU prototype reproduced, negative serial cost retained, dimensional IR bridge tested. Source/arrays, not stage3 admission. Static organism assets excluded; short verification self-contained.','files':files}
(R/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');folder,manifest=publish.prepare(spec,R/'prepared');meta=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_EVENT_SCOPES_IR_2026-09-22.zip')
with archive.open('xb') as stream:
 for row in meta['parts']:
  data=(folder/row['path']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Part hash')
  stream.write(data)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:raise ValueError('ZIP hash')
clean=R/'clean_verification';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
env=os.environ.copy();env.pop('PYTHONPATH',None)
with (R/'CLEAN_CHECK.log').open('x') as log:
 subprocess.run([sys.executable,'-B',str(clean/R.relative_to(ROOT)/'reproduce_short.py')],cwd=clean,env=env,stdout=log,stderr=log,timeout=180,check=True)
result={'archive':str(archive),'sha256':meta['archive_sha256'],'bytes':archive.stat().st_size,'files':len(manifest['files']),'prepared':str(folder),'clean_checks':['all hashes','three file corruptions','dependency fallback','20ms arrays','50ms arrays','external CPU arrays and semantic corruption controls','CPU waveform rerun','dimensional IR bridge rerun'],'full_organism_rerun':False,'scope':'Offline numerical reconstruction and actual CPU prototypes from isolated extraction; original whole-body workloads have separate execution receipts.'}
(R/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
