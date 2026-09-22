"""Bounded clean reproduction: defaults to checks; --full creates an isolated campaign copy."""
from pathlib import Path
import subprocess,sys,time,json,shutil,uuid
H=Path(__file__).resolve().parent
def run(args,cwd,log,timeout):
    start=time.perf_counter()
    with log.open('x') as f:r=subprocess.run([sys.executable,'-B','-O',*args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    return {'args':args,'exit':r.returncode,'wall_s':time.perf_counter()-start}
def main():
    out=H/'reproductions'/uuid.uuid4().hex[:12];out.mkdir(parents=True);rows=[]
    if '--full' not in sys.argv:
        cmds=[['verify_package.py'],['verify_results.py',str(out/'VERIFIED.json')],['test_contracts.py',str(out/'contracts')],['test_native.py',str(out/'native')],['test_blocks.py',str(out/'blocks')],['test_edge_contracts.py',str(out/'edges')],['test_session.py',str(out/'session')],['pilot.py',str(out/'clamp'),'--case','clamp','--profile','precise']]
        for i,c in enumerate(cmds):
            row=run(c,H,out/(str(i)+'.log'),240);rows.append(row)
            (out/'RECEIPT.json').write_text(json.dumps(rows,indent=2)+'\n')
            if row['exit']!=0:raise RuntimeError('short verification failed: '+str(c))
    else:
        # New output directories; existing evidence is never overwritten. Local build artifacts are reused explicitly.
        w=out/'campaign';w.mkdir()
        for p in H.iterdir():
            if p.suffix in {'.py','.cpp','.so'}:shutil.copy2(p,w/p.name)
        for name in ['PLAN.json','FREEZE.json','BLOCKS_PLAN.json','REFERENCE_PLAN.json','DEPENDENCY.json']:shutil.copy2(H/name,w/name)
        shutil.copytree(H/'references',w/'references');(w/'vendor').mkdir();shutil.copytree(H/'vendor/install',w/'vendor/install')
        cmds=[]
        for case in ['clamp','stiff','hh','mixed']:
            for profile in ['fast','precise']:
                c=['pilot.py','runs/'+case+'_'+profile,'--case',case,'--profile',profile]
                if case in ['hh','mixed']:c+=['--reference-dir','references/'+case+'_reference']
                cmds.append(c)
        for profile in ['fast','precise']:cmds.append(['pilot.py','runs/hh_python_'+profile,'--case','hh','--profile',profile,'--implementation','python','--reference-dir','references/hh_reference'])
        for profile in ['fast','precise']:
            for impl in ['python','native']:cmds.append(['bench.py','runs/one_'+impl+'_'+profile,'--implementation',impl,'--profile',profile,'--reference-dir','references/large_fine'])
        cmds += [['bench.py','runs/ten_reference','--implementation','reference','--seconds','10'],['reference_erk.py','runs/ten_reference_erk'],['bench.py','runs/ten_reference_fine','--implementation','reference_fine','--seconds','10']]
        for profile in ['fast','precise']:cmds.append(['bench.py','runs/ten_native_'+profile,'--implementation','native','--profile',profile,'--seconds','10','--reference-dir','runs/ten_reference_erk'])
        args=['bench.py','runs/one_blocks_precise','--implementation','python','--profile','precise','--reference-dir','references/large_fine'];cmds.append(['-c','import sys,implicit_blocks;sys.modules["implicit"]=implicit_blocks;sys.argv='+repr(args)+';import bench;bench.main()'])
        start=time.perf_counter()
        for i,c in enumerate(cmds):
            if time.perf_counter()-start>3360:break
            try:row=run(c,w,out/(str(i)+'.log'),240)
            except subprocess.TimeoutExpired:row={'args':c,'exit':'timeout240','wall_s':240}
            rows.append(row);(out/'RECEIPT.json').write_text(json.dumps(rows,indent=2)+'\n')
        # This is a new exposure; original scientific verdict is not reused.
    print(str(out))
if __name__=='__main__':main()
