"""Repeat all 16 scientific solves of this round in a new self-contained copy.
Large DOP853 references are prior verified evidence, deliberately not regenerated.
"""
from pathlib import Path
import sys,subprocess,shutil,time,json
H=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
for p in H.glob('*.py'):shutil.copy2(p,out/p.name)
for f in ['bridge.cpp','PLAN.json','PILOT_FREEZE.json','DEPENDENCY.json']:shutil.copy2(H/f,out/f)
shutil.copytree(H/'prior_evidence',out/'prior_evidence');(out/'vendor').mkdir();shutil.copy2(H/'vendor/sundials-7.6.0-build.tar.gz',out/'vendor/sundials-7.6.0-build.tar.gz')
ledger=[]
def run(args,log,timeout=180):
    begin=time.perf_counter()
    with (out/log).open('w') as f:
        result=subprocess.run([sys.executable,'-B','-O',*args],cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    ledger.append({'args':args,'subprocess_wall_s':time.perf_counter()-begin,'exit':result.returncode})
    (out/'REPRODUCTION_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    if result.returncode:raise RuntimeError('failed reproduction: '+log)
run(['build_dependency.py'],'dependency.log',600)
for name in ['hh','mixed']:run(['pilot.py','pilots/'+name+'_reference','--case',name,'--reference'],name+'_reference.log')
for name in ['clamp','stiff','hh','mixed']:
    for profile in ['fast','precise']:
        args=['pilot.py','pilots/'+name+'_'+profile,'--case',name,'--profile',profile]
        if name in ['hh','mixed']:args+=['--reference-dir','pilots/'+name+'_reference']
        run(args,name+'_'+profile+'.log')
run(['compare_explicit.py','explicit_hh_01','pilots/hh_reference'],'explicit.log')
for folder,profile in [('large_fast_02','fast'),('large_precise_01','precise')]:run(['large_implicit.py',folder,'prior_evidence',profile],folder+'.log')
run(['verify.py'],'verify.log');print('Reproduced 16 solver trajectories and verified all results in '+str(out))
