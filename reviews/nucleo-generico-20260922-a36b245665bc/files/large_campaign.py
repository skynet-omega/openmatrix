from pathlib import Path
import subprocess,sys,json,time
h=Path(__file__).resolve().parent;out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
roles=['reference','A_fast','A_precise','reference_fine','C_fast','C_precise']
ledger=[];start=time.perf_counter()
for role in roles:
    require_budget=time.perf_counter()-start<3500
    if not require_budget:break
    began=time.perf_counter()
    with (out/(role+'.log')).open('w') as log:
        try:r=subprocess.run([sys.executable,'-B','-O',str(h/'benchmark.py'),role,str(out/role)],stdout=log,stderr=subprocess.STDOUT,timeout=310);rc=r.returncode
        except subprocess.TimeoutExpired:rc=124
    ledger.append({'role':role,'returncode':rc,'wall_s':time.perf_counter()-began})
    (out/'LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(ledger[-1]),flush=True)
