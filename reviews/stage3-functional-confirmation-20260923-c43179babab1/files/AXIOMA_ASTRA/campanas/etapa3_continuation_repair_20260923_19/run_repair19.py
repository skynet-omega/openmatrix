"""One bounded continuation attempt after the disclosed body-alias restore bug."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import traceback

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent / 'etapa3_funcional_repair_20260923_17'
PY = '/home/daroch/miniconda3/envs/GPU/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def guard():
    for path, digest in read(HERE / 'RUN_LOCK.json').items():
        need(sha(path) == digest, 'Frozen source or receipt changed: ' + path)
    for name, target in read(HERE / 'LINK_TARGETS.json').items():
        path = HERE / name
        need(path.is_symlink() and path.resolve() == Path(target),
             'Reused evidence link changed: ' + name)


def main():
    need(not (HERE / 'QUEUE.json').exists(), 'This one-attempt campaign was already started')
    guard()
    contract = read(HERE / 'REPAIR19_CONTRACT.json')
    failed = read(REPAIR / 'continuation_right_01/RESULT.json')
    need(failed['status'] == 'INCOMPLETE' and failed['completed_continuation_ms'] == 0
         and failed['error']['message'] == 'World is not bound to this continuing body',
         'Repair17 failure is not the frozen bug')
    need(read(REPAIR / 'REPAIR_QUEUE.json')['attempts_started'] == 8,
         'Previous attempts changed')
    prior_wall = (float(read(REPAIR / 'REPAIR_QUEUE.json')['wall_consumed_s'])
                  + float(failed['wall_total_s']))
    need(prior_wall + contract['budget']['new_run_wall_s_max']
         <= contract['budget']['aggregate_all_attempts_wall_s_max'],
         'No aggregate budget for the single corrected attempt')
    out = HERE / 'continuation_right_01'
    need(not out.exists(), 'Attempt already exists')
    save(HERE / 'QUEUE.json', {'state': 'RUNNING', 'attempts_total_started': 9,
                              'prior_attempts': 8, 'prior_observed_wall_s': prior_wall,
                              'current': out.name, 'stage3_admission': False})
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (HERE / 'continuation_right_01.log').open('x') as log:
        child = subprocess.Popen([PY, '-B', str(HERE / 'run_continuation.py'), str(out)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=contract['budget']['new_run_wall_s_max'])
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise TimeoutError('One corrected continuation exceeded the frozen wall budget')
    need(code == 0, 'Corrected continuation failed; see its RESULT.json')
    result = read(out / 'RESULT.json')
    need(result['status'] == 'COMPLETE' and result['error'] is None
         and not result['cleanup_errors'] and result['completed_continuation_ms'] == 300,
         'Corrected continuation incomplete')
    need(result['wall_total_s'] <= contract['budget']['new_run_wall_s_max'],
         'Corrected continuation exceeded its own wall receipt')
    save(HERE / 'QUEUE.json', {'state': 'EXPERIMENT_COMPLETE_REQUIRES_RAW_VERIFICATION',
                              'attempts_total_started': 9, 'prior_attempts': 8,
                              'aggregate_observed_wall_s': prior_wall + result['wall_total_s'],
                              'stage3_admission': False})
    guard()
    from verify_repair19 import verify
    final = verify()
    save(HERE / 'FINAL_RAW_VERIFIED.json', final)
    save(HERE / 'QUEUE.json', {'state': 'COMPLETE', 'attempts_total_started': 9,
                              'aggregate_observed_wall_s': final['aggregate_all_attempts_wall_s'],
                              'stage3_admission': final['functional_stage3_pass'],
                              'receipt': 'FINAL_RAW_VERIFIED.json'})
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except BaseException as exc:
        save(HERE / 'REPAIR19_ERROR.json', {'type': type(exc).__name__, 'message': str(exc),
                                           'traceback': traceback.format_exc(),
                                           'stage3_admission': False})
        if (HERE / 'QUEUE.json').exists():
            save(HERE / 'QUEUE.json', {'state': 'FAILED', 'attempts_total_started': 9,
                                      'stage3_admission': False, 'error': str(exc)})
        raise
    raise SystemExit(code)
