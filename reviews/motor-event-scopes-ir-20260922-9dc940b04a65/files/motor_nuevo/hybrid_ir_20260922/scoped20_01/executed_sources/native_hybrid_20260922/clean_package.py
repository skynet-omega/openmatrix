from pathlib import Path
import json,zipfile,hashlib,subprocess,time,os
H=Path(__file__).resolve().parent;prepared=Path(json.loads((H/'PREPARED.json').read_text())['prepared']);a=json.loads((prepared/'ARCHIVE.json').read_text())
exchange=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida');exchange.mkdir(exist_ok=True,parents=True)
archive=exchange/'MATRIX_MOTOR_REAL_NATIVE_2026-09-22.zip'
with archive.open('xb') as out:
 for part in a['parts']:
  data=(prepared/part['path']).read_bytes()
  if hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('Part hash failed')
  out.write(data)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=a['archive_sha256']:raise RuntimeError('Archive hash failed')
root=H/'clean_verification';root.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(root)
manifest=json.loads((root/'MANIFEST.json').read_text())
for f in manifest['files']:
 if hashlib.sha256((root/f['path']).read_bytes()).hexdigest()!=f['sha256']:raise RuntimeError('Extracted hash failed '+f['path'])
N=root/'motor_nuevo/native_hybrid_20260922';P=root/'campanas/etapa3_motor_nuevo_20260922';py='/home/daroch/miniconda3/envs/GPU/bin/python';env=os.environ.copy();env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(N/'vendor')+':'+str(P)+':'+str(N))
steps=[];start=time.perf_counter()
with (H/'CLEAN_CHECK.log').open('w') as log:
 for cpp,so in ((N/'cell_control.cpp',N/'libcell_control.so'),(N/'graph_control_v2.cpp',N/'libgraph_control_v2.so'),(P/'graph_control.cpp',P/'libgraph_control.so')):
  subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(cpp),'-lcudart','-o',str(so)],cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=60)
 for file,optimized in (('report.py',True),('test_structure.py',True),('check_event_boundary.py',False),('check_transaction.py',False)):
  t=time.perf_counter();command=[py,'-B']+(['-O'] if optimized else [])+[str(N/file)]
  subprocess.run(command,cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=120)
  steps.append({'file':file,'passed':True,'seconds':time.perf_counter()-t})
# Read-only extracted-state comparison; references may legitimately differ.
orig=json.loads((H/'VERIFIED.json').read_text());new=json.loads((N/'VERIFIED.json').read_text())
for run in orig['comparisons']:
 for key in ('metrics','histories','limits','screen_pass','changed_exact_fields'):
  if orig['comparisons'][run][key]!=new['comparisons'][run][key]:raise RuntimeError('Clean numerical reconstruction differs')
result={'archive':str(archive),'sha256':a['archive_sha256'],'bytes':archive.stat().st_size,'verified_files':len(manifest['files']),'steps':steps,'wall_s':time.perf_counter()-start,'full_organism_rerun':False,'scope':'Clean extraction builds native libraries, reconstructs saved whole-state comparisons, runs structural corruption tests and GPU event/transaction checks. Static organism assets not included.'}
(H/'CLEAN_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');(H/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
