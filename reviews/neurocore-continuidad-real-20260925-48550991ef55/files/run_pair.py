"""The single prospective 100 ms pair; at most two organism processes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    receipt=HERE/'RUNS.json'
    if receipt.exists():raise RuntimeError('This pair has already been launched; no implicit rerun')
    rows=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    for mode in ('candidate','reference'):
        out=mode+'100_01'
        command=[sys.executable,'-u','-B','run_real.py','--mode',mode,'--ms','100',
                 '--field','odor_left','--out',out,'--compact-state','--wall-limit','665']
        started=time.perf_counter()
        context=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                                '--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        row={'mode':mode,'command':command,'gpu_before':context,'exclusive_gpu':False,'status':'STARTED'}
        rows.append(row);receipt.write_text(json.dumps(rows,indent=2)+'\n')
        with (HERE/(out+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=env)
            row['pid']=p.pid;receipt.write_text(json.dumps(rows,indent=2)+'\n')
            try:code=p.wait(timeout=700)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:code=p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
                row['timeout']=True
        row.update(exit_code=code,launcher_wall_s=time.perf_counter()-started,status='FINISHED')
        result=HERE/out/'RESULT.json'
        if result.exists():
            r=json.loads(result.read_text())
            row.update(simulation_status=r['status'],completed_ms=r['completed_ms'],
                       cleanup_errors=r.get('cleanup_errors',[]))
        receipt.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if code or row.get('simulation_status')!='COMPLETE' or row.get('cleanup_errors'):return 2
    with (HERE/'comparison.log').open('x') as log:
        result=subprocess.run([sys.executable,'-B','compare_real.py','reference100_01','candidate100_01',
             '--out','PAIR100.json'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=env)
    print((HERE/'PAIR100.json').read_text(),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
