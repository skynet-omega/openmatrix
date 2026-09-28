"""Qualification and ten corrected arms, one GPU owner, bounded, no retry."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

H = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    plan = json.loads((H / 'PILOT_PLAN.json').read_text())
    freeze = json.loads((H / 'PILOT_FREEZE.json').read_text())
    if hashlib.sha256((H / 'PILOT_PLAN.json').read_bytes()).hexdigest() != freeze['plan_sha256']:
        raise ValueError('Frozen contract changed')
    if (H / 'QUEUE_START.json').exists():
        raise ValueError('This campaign has already started; no retries')
    start = time.monotonic()
    save(H / 'QUEUE_START.json', dict(unix_s=time.time(), pid=os.getpid(),
                                     cpu_scope='Measured subprocess CPU plus capped auxiliary tasks; wall starts at this queue'))
    reports = []
    used_cpu = used_ms = 0
    status, error, active = 'RUNNING', None, None
    try:
        schedule = [(True, k, v['duration_ms']) for k, v in plan['qualifications'].items()]
        schedule += [(False, k, plan['duration_ms']) for k in plan['arms']]
        for qualification, arm, duration in schedule:
            if not qualification and not (H / 'QUALIFICATION.json').exists():
                q = subprocess.run([sys.executable, '-B', '-O', str(H / 'verify_qualification52.py')],
                                   capture_output=True, text=True, timeout=30)
                (H / 'qualification_analysis.log').write_text(q.stdout + q.stderr)
                if q.returncode:
                    raise RuntimeError('Live qualification failed')
            b = plan['budgets']
            wall_cap = b['per_qualification_wall_s' if qualification else 'per_arm_wall_s']
            cpu_cap = b['per_qualification_CPU_s' if qualification else 'per_arm_CPU_s']
            if time.monotonic() - start + wall_cap > b['wall_s']:
                raise RuntimeError('Insufficient wall budget for another bounded arm')
            if used_cpu + cpu_cap + b['auxiliary_CPU_reserved_s'] > b['CPU_s']:
                raise RuntimeError('Insufficient CPU budget')
            if used_ms + duration > b['neural_ms_total']:
                raise RuntimeError('Insufficient neural budget')
            active = ('qual_' if qualification else '') + arm
            save(H / 'QUEUE_STATUS.json', dict(status='RUNNING', active=active, completed=reports,
                                              CPU_s=used_cpu, attempted_ms=used_ms,
                                              elapsed_wall_s=time.monotonic() - start))
            command = [sys.executable, '-B', '-O', str(H / 'run_pilot.py'), arm]
            if qualification:
                command.append('--qualification')
            with (H / (active + '.log')).open('x') as log:
                worker = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=wall_cap + 10)
            result = json.loads((H / active / 'RESULT.json').read_text())
            summary = {k: result[k] for k in ['arm', 'qualification', 'status', 'attempted_ms',
                                            'committed_ms', 'CPU_s', 'wall_s']}
            reports.append(summary)
            used_cpu += result['CPU_s']
            used_ms += result['attempted_ms']
            print(json.dumps(summary), flush=True)
            if worker.returncode or result['status'] != 'COMPLETE':
                raise RuntimeError('Arm failed: ' + active)
            disk = sum(f.stat().st_size for f in H.rglob('*') if f.is_file())
            if disk > b['new_disk_bytes']:
                raise RuntimeError('Disk budget')
        status = 'COMPLETE'
    except BaseException as exc:
        status, error = 'STOPPED', repr(exc)
    finally:
        out = dict(status=status, active=active, arms=reports, CPU_s=used_cpu,
                   attempted_ms=used_ms, committed_ms=sum(r['committed_ms'] for r in reports),
                   queue_wall_s=time.monotonic() - start, error=error)
        if status != 'COMPLETE' and active is not None and not (H / active / 'RESULT.json').exists():
            out['unclosed_worker_accounting'] = dict(active=active,
                reserved_attempt_ms=duration, CPU_cap_s=cpu_cap,
                note='Failure without closed receipt; do not undercount as zero or retry.')
        save(H / 'QUEUE_RESULT.json', out)
        save(H / 'QUEUE_STATUS.json', out)
        print(json.dumps(out), flush=True)
    if status != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
