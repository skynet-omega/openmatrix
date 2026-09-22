from pathlib import Path
import sys,json,time,subprocess,os,threading
H=Path(__file__).resolve().parent
# The outer runner invokes this only once timed campaign A has stopped.
ledger=[]
def run(name,code):
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if r.returncode==0:
                try:samples.append([float(x.strip()) for x in r.stdout.strip().split(',')])
                except ValueError:pass
    th=threading.Thread(target=monitor,daemon=True);th.start();start=time.perf_counter()
    with (H/(name+'.log')).open('w') as log:
        try:r=subprocess.run([sys.executable,'-B','-O','-c',code],cwd=H,stdout=log,stderr=subprocess.STDOUT,timeout=240);rc=r.returncode
        except subprocess.TimeoutExpired:rc='timeout240'
    stop.set();th.join(timeout=5)
    row={'name':name,'exit':rc,'subprocess_wall_s':time.perf_counter()-start,'device_wide_peak_MiB':max([x[0] for x in samples],default=None)};ledger.append(row)
    (H/'BLOCKS_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(row),flush=True)
    return rc
if run('blocks_validation',"import sys;sys.argv=['test_blocks.py','blocks_validation'];import test_blocks;test_blocks.main()")!=0:raise RuntimeError('block operator validation failed')
# Supply the independent B class to existing unchanged benchmark runners.
for name,mod,args in [('hh_blocks_precise','pilot',['pilot.py','runs/hh_blocks_precise','--case','hh','--profile','precise','--implementation','python','--reference-dir','references/hh_reference']),('one_blocks_precise','bench',['bench.py','runs/one_blocks_precise','--implementation','python','--profile','precise','--reference-dir','references/large_fine'])]:
    code='import sys,implicit_blocks;sys.modules["implicit"]=implicit_blocks;sys.argv='+repr(args)+';import '+mod+';'+mod+'.main()'
    if run(name,code)!=0:raise RuntimeError('block candidate failed: '+name)
