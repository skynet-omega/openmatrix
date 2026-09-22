from pathlib import Path
import sys,subprocess,shutil,json,time,hashlib
h=Path(__file__).resolve().parent;out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
reused={}
for role in ['reference','reference_fine']:
 shutil.copytree(h/'large_01'/role,out/role)
 reused[role]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/role).iterdir() if p.is_file()}
(out/'REUSED_CPU_REFERENCES.json').write_text(json.dumps({'reason':'float64 GPU buffer bug does not affect CPU references; no CPU rerun','files':reused},indent=2)+'\n')
ledger=[]
for role in ['A_fast','A_precise','C_fast','C_precise']:
 start=time.perf_counter()
 with (out/(role+'.log')).open('w') as log:
  r=subprocess.run([sys.executable,'-B','-O',str(h/'benchmark.py'),role,str(out/role)],stdout=log,stderr=subprocess.STDOUT,timeout=310)
 ledger.append({'role':role,'returncode':r.returncode,'wall_s':time.perf_counter()-start,'reason':'mandatory affected-arm repetition after generic dtype bug repair'})
 (out/'LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(ledger[-1]),flush=True)
