"""Serial queue with one aggregate wall/CPU budget; no automatic retry or retune."""
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'motor_nuevo/dynamics_12s_20260926_13'))
from supervise import guard
from observations import save,need


def main():
    plan=json.loads((HERE/'PLAN.json').read_text())
    with (HERE/'LAUNCH.json').open('x') as f:json.dump({'started_unix_s':time.time()},f)
    start=time.monotonic()
    used_cpu=0.
    results=[]
    for arm in plan['arms']:
        remaining=plan['aggregate_wall_s_max']-(time.monotonic()-start)
        need(remaining>60,'Aggregate wall budget exhausted')
        result=guard([sys.executable,str(HERE/'run.py'),'--arm',arm,'--out',str(HERE/arm)],
            log=HERE/(arm+'.log'),receipt=HERE/(arm+'_SUPERVISOR.json'),folder=HERE,
            wall_s=min(remaining,plan['per_arm_wall_s_max']),cpu_s=plan['aggregate_cpu_s_max']-used_cpu,
            ram_bytes=plan['RAM_GiB_max']*1024**3,disk_limit=plan['disk_GiB_max']*1024**3,
            grace_s=30.,interval_s=2.)
        results.append(dict(arm=arm,**result))
        used_cpu+=result['final_wait4_cpu_s']
        save(HERE/'QUEUE.json',dict(status='RUNNING',completed_arms=results,
            elapsed_s=time.monotonic()-start,used_cpu_s=used_cpu))
        if result['status']!='EXIT_OK':
            save(HERE/'QUEUE.json',dict(status='INCOMPLETE',completed_arms=results,
                elapsed_s=time.monotonic()-start,automatic_retries=0))
            return 2
    result=guard([sys.executable,str(HERE/'analyze.py')],log=HERE/'analysis.log',
        receipt=HERE/'ANALYSIS_SUPERVISOR.json',folder=HERE,
        wall_s=min(300,plan['aggregate_wall_s_max']-(time.monotonic()-start)),cpu_s=300,
        ram_bytes=4*1024**3,disk_limit=plan['disk_GiB_max']*1024**3,grace_s=10.,interval_s=1.)
    save(HERE/'QUEUE.json',dict(status='COMPLETE' if result['status']=='EXIT_OK' else 'INCOMPLETE',
        completed_arms=results,analysis=result,elapsed_s=time.monotonic()-start,
        used_cpu_s=used_cpu+result['final_wait4_cpu_s'],automatic_retries=0))
    return 0 if result['status']=='EXIT_OK' else 2


if __name__=='__main__':raise SystemExit(main())
