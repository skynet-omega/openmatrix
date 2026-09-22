"""Finish the reserved runs without expanding PLAN's20-science budget."""
from pathlib import Path
import json,sys,subprocess,time,threading
H=Path(__file__).resolve().parent
ledger=[]
def run(name,cmd,scientific):
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if r.returncode==0:
                try:samples.append([float(x.strip()) for x in r.stdout.strip().split(',')])
                except ValueError:pass
    th=threading.Thread(target=monitor,daemon=True);th.start();start=time.perf_counter()
    with (H/(name+'.log')).open('x') as log:
        try:r=subprocess.run([sys.executable,'-B','-O',*cmd],cwd=H,stdout=log,stderr=subprocess.STDOUT,timeout=240);rc=r.returncode
        except subprocess.TimeoutExpired:rc='timeout240'
    stop.set();th.join(timeout=5)
    row={'name':name,'scientific':scientific,'args':cmd,'exit':rc,'subprocess_wall_s':time.perf_counter()-start,'device_wide_peak_MiB':max([x[0] for x in samples],default=None),'monitor':'device-wide sampled every0.5s, includes other processes'};ledger.append(row)
    (H/'FINISH_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(row),flush=True)
    return rc
if __name__=='__main__':
    if len(json.loads((H/'CAMPAIGN.json').read_text()))!=17:raise ValueError('unexpected consumed budget')
    ref=json.loads((H/'runs/ten_reference_erk/RESULT.json').read_text())
    if not ref['completed']:raise ValueError('reference unfinished')
    run('ten_native_precise',['bench.py','runs/ten_native_precise','--implementation','native','--seconds','10','--profile','precise','--reference-dir','runs/ten_reference_erk'],True)
    if run('blocks_validation',['test_blocks.py','blocks_validation'],False)!=0:raise RuntimeError('block operator validation failed')
    args=['bench.py','runs/one_blocks_precise','--implementation','python','--profile','precise','--reference-dir','references/large_fine']
    code='import sys,implicit_blocks;sys.modules["implicit"]=implicit_blocks;sys.argv='+repr(args)+';import bench;bench.main()'
    run('one_blocks_precise',['-c',code],True)
