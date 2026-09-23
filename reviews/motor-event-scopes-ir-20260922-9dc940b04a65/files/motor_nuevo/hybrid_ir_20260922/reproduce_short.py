"""Clean extraction: verify arrays, corruption controls and execute CPU prototypes."""
from pathlib import Path
import sys,json,hashlib,subprocess,os
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
m=json.loads((ROOT/'MANIFEST.json').read_text())
def check(row):
 if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Hash mismatch: '+row['path'])
for row in m['files']:check(row)
corruptions=[]
for suffix in ('hybrid_ir_20260922/PLAN.json','hybrid_ir_20260922/scoped50_01/brain_50ms.npz','hybrid_ir_20260922/scoped20_01/EVENT_CONTRACT.json'):
 row=next(x for x in m['files'] if x['path'].endswith(suffix));p=ROOT/row['path'];original=p.read_bytes()
 try:
  changed=bytearray(original);changed[len(changed)//2]^=1;p.write_bytes(changed)
  try:check(row)
  except ValueError:corruptions.append(suffix)
  else:raise RuntimeError('Corruption not detected')
 finally:p.write_bytes(original)
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
commands=[['check_dependencies.py'],['compare_round.py','20'],['compare_round.py','50'],['verify_external.py'],['trayectorias_cpu.py',str(R/'clean_cpu_replay')],['check_ir_bridge.py',str(R/'clean_ir_replay')]]
for name,*args in commands:
 subprocess.run([sys.executable,'-B','-O',str(R/name),*args],cwd=ROOT,env=env,check=True,timeout=60)
print(json.dumps({'commands':commands,'corruptions_detected':corruptions,'organism_replayed':False,'CPU_prototypes_replayed':True}))
