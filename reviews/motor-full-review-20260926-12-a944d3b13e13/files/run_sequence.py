"""One serial bounded queue; no long arm unless the frozen short pair passes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def child(command, log, budget):
    started = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=HERE.parents[1], stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget-1)
        except subprocess.TimeoutExpired:
            # Kill the process group, including a compiler if setup got stuck.
            import os
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
            raise TimeoutError('External hard deadline: '+str(command))
    elapsed = time.monotonic()-started
    if code != 0:
        raise RuntimeError('Child failed with exit '+str(code)+': '+str(log))
    if elapsed > budget:
        raise TimeoutError('Process consumed its wall budget')
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('short','long'), required=True)
    args = parser.parse_args()
    from source_inventory import verify
    plan = json.loads((HERE/'PLAN.json').read_text())
    verify(json.loads((HERE/'SOURCES.json').read_text()))
    if args.stage == 'long':
        gate = json.loads((HERE/'PAIR100.json').read_text())
        if gate['status'] != 'PASS':
            raise ValueError('Short pair has not passed')
        from source_inventory import sha
        if (gate['provenance']['plan_sha256'] != sha(HERE/'PLAN.json') or
                gate['provenance']['sources_sha256'] != sha(HERE/'SOURCES.json')):
            raise ValueError('Short-pair plan or sources changed')
        ms = 2000
        budgets = dict(stable=plan['confirmation']['stable_wall_s_max'],
                       reviewed=plan['confirmation']['candidate_wall_s_max'])
    else:
        ms = 100
        budgets = dict.fromkeys(('stable','reviewed'), plan['short_pair']['wall_s_each_max'])
    queue = HERE/('QUEUE_'+args.stage+'.json')
    if queue.exists():
        raise FileExistsError('Do not rerun or overwrite this queue')
    record = dict(status='RUNNING', stage=args.stage, completed=[], process_wall_s={})
    save(queue, record)
    try:
        for engine in ('stable','reviewed'):
            name = engine+'_'+str(ms)+'ms_01'
            record['active'] = name
            save(queue, record)
            print(json.dumps(dict(starting=name, budget_s=budgets[engine])), flush=True)
            elapsed = child([sys.executable, str(HERE/'run_trial.py'), '--engine', engine,
                             '--ms', str(ms), '--out', str(HERE/name), '--wall-limit',
                             str(budgets[engine]-15)], HERE/(name+'.log'), budgets[engine])
            record['completed'].append(name)
            record['process_wall_s'][engine] = elapsed
            save(queue, record)
        target = HERE/('PAIR'+str(ms)+'.json')
        child([sys.executable, str(HERE/'verify_runs.py'), '--stable',
               str(HERE/('stable_'+str(ms)+'ms_01')), '--reviewed',
               str(HERE/('reviewed_'+str(ms)+'ms_01')), '--ms', str(ms),
               '--out', str(target)], HERE/('verify_'+str(ms)+'.log'), 120)
        report = json.loads(target.read_text())
        if report['status'] != 'PASS':
            raise ValueError('Functional comparison failed; no further trials')
        record.update(status='PASS', active=None, verification=str(target))
    except BaseException as error:
        record.update(status='STOPPED', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        save(queue, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
