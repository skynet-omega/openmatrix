"""Finite sequential four-arm queue; stop on first noninterpretable arm."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    plan=json.loads((HERE/'DYNAMIC_PLAN.json').read_text());start=time.monotonic()
    out=HERE/'QUEUE.json'
    if out.exists():raise FileExistsError(out)
    state=dict(status='RUNNING',pid=os.getpid(),completed=[],current=None,budget=plan['budget'])
    def save():
        state['elapsed_s']=time.monotonic()-start
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
    save()
    for condition in plan['conditions']:
        remaining=plan['budget']['aggregate_wall_s']-(time.monotonic()-start)
        if remaining<plan['budget']['per_run_wall_s']:
            state.update(status='BUDGET_EXHAUSTED',current=None);save();return 2
        folder=HERE/(condition+'_01');state['current']=condition;save()
        with (HERE/(condition+'_01.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'run_probe.py'),'--condition',condition,'--out',str(folder)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=proc.pid;save()
            try:code=proc.wait(timeout=plan['budget']['per_run_wall_s'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:code=proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
        receipt=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else {}
        state['completed'].append(dict(condition=condition,exit_code=code,status=receipt.get('status','NO_RECEIPT'),wall_s=receipt.get('wall_total_s')))
        if code or receipt.get('status')!='COMPLETE':
            state.update(status='STOPPED_AT_FAILURE',current=None);save();return 2
        save()
    state.update(status='COMPLETE',current=None);save();return 0


if __name__=='__main__':raise SystemExit(main())
