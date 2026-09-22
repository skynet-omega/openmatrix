from pathlib import Path
import json,hashlib,subprocess,sys,os,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parents[1];N=H.parent/'native_hybrid_20260922';P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'));import publish
files={}
def include(f):
 if f.is_file() and f.suffix in publish.EXT and '__pycache__' not in f.parts:
  files[str(f.relative_to(ROOT))]={'source':str(f),'destination':str(f.relative_to(ROOT))}
exclude={'prepared','clean_verification','PUBLICATION_MANIFEST.json','PUBLICATION.json','PREPARED.json','CLEAN_CHECK.json','DELIVERY.json','CLEAN_CHECK.log'}
for f in H.rglob('*'):
 if not any(k in exclude for k in f.relative_to(H).parts):include(f)
for f in N.iterdir():
 if f.suffix in ('.py','.cpp','.cu') or f.name=='PETSC_LICENSE.txt':include(f)
for folder in ('vendor','legacy_sources'):
 for f in (N/folder).rglob('*'):include(f)
for name in ('brain_final.json','brain_final.npz'):include(N/'baseline_01'/name)
for name in ('brain_final.json','brain_final.npz','body_final.npz','traces.npz'):include(N/'aligned_01'/name)
for f in P.iterdir():
 if f.suffix in ('.py','.cpp'):include(f)
for f in (P/'verification_vendor').rglob('*'):include(f)
for f in (H.parent/'resident_pn_20260922').iterdir():
 if f.suffix in ('.py','.cpp','.cu'):include(f)
include(ROOT/'instrumentos/openmatrix/publish.py')
spec={'label':'motor-causal-blocks-20260922','description':'CUDA independent adaptive blocks: actual organism20ms, reference82.82s to candidate60.32s; native membranes5.21x. Effective-operator cold recovery repaired in1ms test. Affine receiver prototype separate; nonlinear global bottleneck and whole-engine speed goal remain open. Sources, arrays and portable checks; full static anatomy/body assets not included.','files':[files[k] for k in sorted(files)]}
(H/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n')
folder,manifest=publish.prepare(spec,H/'prepared');(H/'PREPARED.json').write_text(json.dumps({'prepared':str(folder),'files':len(files)},indent=2)+'\n')
a=json.loads((folder/'ARCHIVE.json').read_text());archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_MOTOR_CAUSAL_BLOCKS_2026-09-22.zip')
with archive.open('xb') as out:
 for part in a['parts']:
  data=(folder/part['path']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('ZIP part hash mismatch')
  out.write(data)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=a['archive_sha256']:raise RuntimeError('ZIP hash mismatch')
clean=H/'clean_verification';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
for row in manifest['files']:
 if hashlib.sha256((clean/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Extracted hash mismatch')
Q=clean/'motor_nuevo/causal_runtime_20260922';env=os.environ.copy();env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUPY_CACHE_DIR=str(clean/'cuda_cache'))
# No PYTHONPATH from the working tree. All short-check sources are extracted.
env.pop('PYTHONPATH',None);steps=[]
with (H/'CLEAN_CHECK.log').open('w') as log:
 for name in ('check_scheduler.py','check_affine.py','check_device.py','check_operator_state.py','report.py'):
  command=[sys.executable,'-B']+(['-O'] if name in ('check_operator_state.py','report.py') else [])+[str(Q/name)]
  subprocess.run(command,cwd=clean,env=env,stdout=log,stderr=log,check=True,timeout=120);steps.append(name)
orig=json.loads((H/'VERIFIED.json').read_text());new=json.loads((Q/'VERIFIED.json').read_text())
for key in orig['comparisons']:
 for metric in ('metrics','limits','screen_pass','histories','changed_exact_fields'):
  if orig['comparisons'][key][metric]!=new['comparisons'][key][metric]:raise RuntimeError('Clean reconstruction changed '+key+'/'+metric)
if orig['recovery_changed_paths']!=new['recovery_changed_paths']:raise RuntimeError('Clean recovery reconstruction differs')
result={'archive':str(archive),'sha256':a['archive_sha256'],'bytes':archive.stat().st_size,'verified_files':len(files),'short_checks':steps,'clean_cuda_compilation_cache':str(clean/'cuda_cache'),'full_organism_rerun':False,'scope':'Clean extraction runs both new CUDA components, operator registry and device rollback/grouping fixtures, and reconstructs full saved-state comparisons. It does not rerun the whole organism.'}
(H/'CLEAN_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');(H/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
