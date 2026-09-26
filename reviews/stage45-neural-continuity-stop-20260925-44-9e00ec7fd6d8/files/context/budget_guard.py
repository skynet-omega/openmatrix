"""External resource guard; executed neural sources remain immutable."""
from __future__ import annotations
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise RuntimeError(message)

def remaining(plan,records):
    cpu=0.0;wall=0.0;steps=0
    for row in records:
        for key in ('cpu_s','wall_s'):
            require(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key]>=0,'Invalid resource accounting')
        require(type(row['attempted_ms']) is int and row['attempted_ms']>=0,'Invalid step accounting')
        cpu+=row['cpu_s'];wall+=row['wall_s'];steps+=row['attempted_ms']
    b=plan['budget']
    require(len(records)<b['neural_processes_max'],'Process budget exhausted')
    require(cpu<b['cpu_s_aggregate_max'],'CPU budget exhausted; no new neural process')
    require(wall<b['wall_s_aggregate_max'],'Wall budget exhausted')
    require(steps<b['neural_ms_max'],'Interaction budget exhausted')
    return {'cpu_s':b['cpu_s_aggregate_max']-cpu,'wall_s':min(b['wall_s_per_process_max'],b['wall_s_aggregate_max']-wall),
            'neural_ms':b['neural_ms_max']-steps}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['sham','common','virtual'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads((HERE/'PLAN.json').read_text());records=[]
    for name in ('sham_01','common_01','virtual_01'):
        directory=HERE/name
        if directory.exists():
            require((directory/'RESULT.json').is_file(),'Another unclosed run exists')
            records.append(json.loads((directory/'RESULT.json').read_text()))
    limits=remaining(plan,records)
    require(limits['neural_ms']>=120,'Insufficient fixed-window budget')
    require(not a.out.exists(),'Output directory already exists')
    import psutil
    command=[sys.executable,'-B',str(HERE/'run_branch.py'),'--arm',a.arm,'--out',str(a.out)]
    start=time.monotonic();child=subprocess.Popen(command,start_new_session=True)
    process=psutil.Process(child.pid);reason=None;cpu=0.0
    while child.poll() is None:
        try:
            cpu=process.cpu_times().user+process.cpu_times().system
            rss=process.memory_info().rss/1024**3
        except psutil.NoSuchProcess:break
        if cpu>=limits['cpu_s']:reason='CPU'
        elif time.monotonic()-start>=limits['wall_s']:reason='WALL'
        elif rss>plan['budget']['RAM_GiB_max']:reason='RAM'
        if reason:
            os.killpg(child.pid,signal.SIGTERM)
            # Preserve the runner's failure/checkpoint evidence. Cleanup CPU is
            # reported, never counted as more allowed neural exposure.
            with (a.out.parent/(a.out.name+'_GUARD_STOP.json')).open('x') as f:
                json.dump({'reason':reason,'observed_cpu_s':cpu,'elapsed_wall_s':time.monotonic()-start,
                           'remaining_before_run':limits,'no_budget_extension':True},f,indent=2);f.write('\n')
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
        time.sleep(0.1)
    require(reason is None and child.returncode==0,'Guard stopped or branch failed: '+str(reason))

if __name__=='__main__':main()
