"""Explicit round evidence, immutable freeze, clean extraction and short checks."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,os
T=Path(__file__).resolve().parent;ROOT=T.parents[1];M=T.parent
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
selected={}
def add(p):
 if p.is_file() and p.suffix in publish.EXT:selected[str(p.relative_to(ROOT))]=p
skip={'PUBLICATION.json','PUBLICATION_MANIFEST.json','FINAL_PUBLICATION.json','FINAL_MANIFEST.json','DELIVERY.json','CLEAN_CHECK.log','REMOTE_CHECK.json'}
for p in T.iterdir():
 if p.is_file() and p.name not in skip:add(p)
for p in (T/'jev_01').iterdir():add(p)
for name in ('reference_01','causal_01','fine1562_01','guard20_01','guard20_02','fine1562_20'):
 base=T/name
 for p in base.iterdir():
  if p.is_file():add(p)
 frozen=base/'executed_sources'
 if frozen.exists():
  for group in frozen.iterdir():
   if group.is_dir():
    for p in group.iterdir():add(p)
   else:add(group)
P=ROOT/'campanas/etapa3_motor_nuevo_20260922';N=M/'native_hybrid_20260922';C=M/'causal_runtime_20260922';H=M/'pipeline_review_20260922'
for base in (P,N,C,H,M/'resident_pn_20260922'):
 for p in base.iterdir():
  if p.suffix in ('.py','.cpp','.cu','.hpp') and p.name not in ('package.py','bundle_review.py'):add(p)
for base in (P/'verification_vendor',N/'vendor',N/'legacy_sources'):
 for p in base.iterdir():add(p)
add(N/'PETSC_LICENSE.txt');add(ROOT/'instrumentos/openmatrix/publish.py')
for base in (H/'smoke_causal_01',H/'smoke_reference_01',C/'reference20_01'):
 for name in ('brain_final.npz','brain_final.json','body_final.npz','traces.npz','RESULT.json'):add(base/name)
payload=T/'publication_sources';payload.mkdir(exist_ok=False);files=[]
for relative,source in sorted(selected.items()):
 dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());files.append({'source':str(dest),'destination':relative})
spec={'label':'motor-event-resolution-20260922','description':'Actual event-time localization; guarded membrane prototype improves1/5ms versus declared refinement, still slow. Refined organism fails domain during15thms. Both failures and original arrays preserved. Offline reconstruction and short CUDA checks; no full static organism assets, no stage3 admission.','files':files}
(T/'FINAL_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n');folder,manifest=publish.prepare(spec,T/'prepared_final');meta=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_EVENT_RESOLUTION_2026-09-22.zip')
with archive.open('xb') as stream:
 for row in meta['parts']:
  data=(folder/row['path']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('Part hash')
  stream.write(data)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:raise RuntimeError('Archive hash')
clean=T/'clean_verification';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
for row in manifest['files']:
 if hashlib.sha256((clean/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Extraction hash')
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUPY_CACHE_DIR=str(clean/'cuda_cache'))
base=clean/T.relative_to(ROOT);cp=clean/P.relative_to(ROOT)
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(cp/'graph_control.cpp'),'-o',str(cp/'libgraph_control.so'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64','-lcudart'],check=True,timeout=60)
checks=[]
with (T/'CLEAN_CHECK.log').open('x') as log:
 for name in ('analyze_boundaries.py','compare_candidates.py','reconstruct_membranes.py','check_guard_attach.py','check_port_replay.py','check_guard_threshold.py','check_failure_diagnostic.py'):
  subprocess.run([sys.executable,'-B','-O',str(base/name)],cwd=clean,env=env,stdout=log,stderr=log,timeout=60,check=True);checks.append(name)
result={'archive':str(archive),'sha256':meta['archive_sha256'],'bytes':archive.stat().st_size,'files':len(manifest['files']),'clean_checks':checks,'prepared':str(folder),'full_organism_rerun':False,'scope':'Full recorded round reconstruction and small fixtures from isolated extraction, not a clean full-organism replay or resume.'}
(T/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
