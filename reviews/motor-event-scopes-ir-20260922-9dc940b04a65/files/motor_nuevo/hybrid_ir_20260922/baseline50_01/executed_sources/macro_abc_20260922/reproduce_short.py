"""Clean-extraction checks; never load or advance the historical organism."""
from pathlib import Path
import sys,json,hashlib,subprocess,os
R=Path(__file__).resolve().parent;ROOT=R.parents[1];P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
m=json.loads((ROOT/'MANIFEST.json').read_text())
def check(row):
 if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Evidence hash mismatch: '+row['path'])
for row in m['files']:check(row)
# Deliberately corrupt a criterion, an observed input and a state descriptor.
corruptions=[]
for suffix in ('macro_abc_20260922/PLAN.json','macro_abc_20260922/domain_01/failure_arrays.npz','macro_abc_20260922/guard_reset_01/brain_01ms.json'):
 row=next(x for x in m['files'] if x['path'].endswith(suffix));path=ROOT/row['path'];original=path.read_bytes()
 try:
  changed=bytearray(original);changed[len(changed)//2]^=1;path.write_bytes(changed)
  try:check(row)
  except ValueError:corruptions.append(suffix)
  else:raise RuntimeError('Undetected evidence corruption')
 finally:path.write_bytes(original)
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(P/'graph_control.cpp'),'-o',str(P/'libgraph_control.so'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64','-lcudart'],check=True,timeout=60)
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUPY_CACHE_DIR=str(ROOT/'cuda_cache'),NUMBA_CACHE_DIR=str(ROOT/'numba_cache'))
checks=[]
for name in ('verify_domain_payload.py','verify_round.py','check_event_contract.py','check_diagnostic.py'):
 subprocess.run([sys.executable,'-B','-O',str(R/name)],cwd=ROOT,env=env,check=True,timeout=60);checks.append(name)
print(json.dumps({'checks':checks,'corruptions_detected':corruptions,'full_organism_replayed':False,'scope':'Offline reconstruction and independent small fixtures from this extraction.'}))
