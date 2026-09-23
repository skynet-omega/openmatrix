"""Execute the three predeclared odor arms only after the sham gates pass."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


H=Path(__file__).resolve().parent
ROOT=H.parents[1]
ARMS=(('odor_left','full_odor_left_01'),('odor_right','full_odor_right_01'),('uniform','full_uniform_01'))


def load(path):return json.loads(path.read_text())


def check_arm(folder):
    r=load(folder/'RESULT.json')
    f=load(folder/'flow/RESULT.json')
    if (r['status']!='COMPLETE' or r['completed_preparation_ms']!=40 or
        r['completed_trial_ms']!=400 or r['cleanup_errors'] or f['samples']!=440 or
        f['max_target_error']>1e-9 or f['max_rate_error']>1e-12):
        raise ValueError('Arm failed frozen physical or observer gate: '+str(folder))
    return r


def record(row):
    with (H/'ORCHESTRATOR.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    plan=load(H/'PLAN.json')
    if not load(H/'SMOKE_COMPARE.json')['exact_scientific_state']:
        raise ValueError('Observer smoke did not preserve the organism')
    sham=check_arm(H/'full_sham_01')
    used=sum(load(H/name/'RESULT.json')['wall_total_s'] for name in
             ('smoke_sham_on_01','smoke_sham_off_01','full_sham_01'))
    if used>plan['budgets']['wall_total_s_max']:raise ValueError('Budget exhausted after sham')
    for odor,name in ARMS:
        target=H/name
        if target.exists():raise FileExistsError('No overwrite/retry: '+str(target))
        if used+plan['budgets']['wall_single_400ms_s_max']>plan['budgets']['wall_total_s_max']:
            raise RuntimeError('Not enough aggregate budget for next frozen run')
        record({'event':'START','odor':odor,'folder':name,'wall_used_before_s':used,'time_utc_epoch_s':time.time()})
        code=subprocess.run([sys.executable,str(H/'run_set.py'),'--out',str(target),
                             '--odor',odor,'--engine','causal_cuda','--ms','400','--observe','on'],
                            cwd=ROOT,check=False).returncode
        result=load(target/'RESULT.json')
        used+=result['wall_total_s']
        record({'event':'FINISH','odor':odor,'folder':name,'returncode':code,
                'status':result['status'],'completed_trial_ms':result['completed_trial_ms'],
                'wall_used_after_s':used,'time_utc_epoch_s':time.time()})
        if code or used>plan['budgets']['wall_total_s_max']:
            raise RuntimeError('Frozen arm failed/budget exhausted; remaining arms not started')
        check_arm(target)
    print(json.dumps({'completed_odors':[x[0] for x in ARMS],'total_wall_s':used}))


if __name__=='__main__':main()
