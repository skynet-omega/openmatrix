"""Explicit evidence freeze and one clean-extraction verification."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,os
R=Path(__file__).resolve().parent;ROOT=R.parents[1];M=R.parent;T=M/'transient_localization_20260922';P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
selected={}
def add(p):
 if p.is_file() and p.suffix in publish.EXT:selected[str(p.relative_to(ROOT))]=p
skip={'PUBLICATION.json','PUBLICATION_MANIFEST.json','DELIVERY.json','CLEAN_CHECK.log','REMOTE_CHECK.json'}
for p in R.iterdir():
 if p.is_file() and p.name not in skip:add(p)
for base in (R/'jev_01',R/'verification_vendor'):
 for p in base.iterdir():add(p)
for name in ('domain_01','fine_reset_01','guard_reset_01','profile_01'):
 base=R/name
 for p in base.iterdir():
  if p.is_file():add(p)
 for sub in ('executed_sources','inputs'):
  if (base/sub).exists():
   for p in (base/sub).rglob('*'):add(p)
for name in ('brain_01ms.json','brain_01ms.npz','brain_05ms.json','brain_05ms.npz','RESULT.json'):
 add(T/'fine1562_20'/name)
for p in T.iterdir():
 if p.suffix=='.py' and p.name not in ('bundle_round.py',):add(p)
for base in (P,M/'pipeline_review_20260922',M/'native_hybrid_20260922',M/'causal_runtime_20260922',M/'resident_pn_20260922'):
 for p in base.iterdir():
  if p.suffix in ('.py','.cpp','.cu','.hpp') and p.name not in ('package.py','bundle_review.py'):add(p)
for base in (P/'verification_vendor',M/'native_hybrid_20260922/vendor',M/'native_hybrid_20260922/legacy_sources'):
 for p in base.iterdir():add(p)
add(M/'native_hybrid_20260922/PETSC_LICENSE.txt');add(ROOT/'instrumentos/openmatrix/publish.py')
payload=R/'publication_sources';payload.mkdir(exist_ok=False);files=[]
for relative,source in sorted(selected.items()):
 dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());files.append({'source':str(dest),'destination':relative})
spec={'label':'motor-general-events-macro-20260922','description':'Generic ADD/SET event transport and model state domains; whole-organism20ms repaired comparison, original domain failure and macro cost discrimination. Still73seconds for20ms; no stage3 admission. Local and ChatGPT alternatives, Jev403 disclosed. Short reproduction self-contained; complete static organism assets excluded.','files':files}
(R/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');folder,manifest=publish.prepare(spec,R/'prepared');meta=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_GENERAL_EVENTS_MACRO_2026-09-22.zip')
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
result={'archive':str(archive),'sha256':meta['archive_sha256'],'bytes':archive.stat().st_size,'files':len(manifest['files']),'prepared':str(folder),'clean_checks':['hashes','3 deliberate corruptions','native build','verify_domain_payload','verify_round','check_event_contract','check_diagnostic'],'full_organism_rerun':False,'scope':'Offline reconstruction and small fixtures from isolated extraction, not full-organism distribution or resume.'}
(R/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
