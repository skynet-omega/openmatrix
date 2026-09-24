"""Raw, read-only scientific verification after one disclosed restore bug fix.

The original frozen verifier owns every scientific threshold.  This wrapper
accounts for both prior incomplete attempts and checks the changed loader and
the new continuation receipt.  It does not promote the old strict contract.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'etapa3_funcional_20260923_16'
REPAIR = HERE.parent / 'etapa3_funcional_repair_20260923_17'
NATIVE = HERE.parent / 'etapa3_pn629_intervention_20260923_15'
sys.path.insert(0, str(OLD))
from verify_all import verify as verify_original
sys.path.insert(0, str(REPAIR))
from verify_repair import upright, sources


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    lock = read(HERE / 'RUN_LOCK.json')
    for path, digest in lock.items():
        need(sha(path) == digest, 'Frozen source or receipt changed: ' + path)
    for name, target in read(HERE / 'LINK_TARGETS.json').items():
        path = HERE / name
        need(path.is_symlink() and path.resolve() == Path(target),
             'Reused evidence link changed: ' + name)
    contract = read(HERE / 'REPAIR19_CONTRACT.json')
    need(sha(HERE / 'PLAN.json') == contract['original_plan_sha256'], 'Original criteria changed')
    need(read(OLD / 'PLAN.json') == read(HERE / 'PLAN.json'), 'Plan copy differs')
    need(read(REPAIR / 'REPAIR_ERROR.json')['stage3_admission'] is False, 'Prior failure receipt missing')
    failed = read(REPAIR / 'continuation_right_01/RESULT.json')
    need(failed['status'] == 'INCOMPLETE' and failed['completed_continuation_ms'] == 0
         and failed['initial_exact'] is None and failed['stage3_admission'] is False,
         'Prior failed continuation was obscured')
    need(failed['error']['message'] == 'World is not bound to this continuing body',
         'Prior failure changed')
    need(read(REPAIR / 'REPAIR_QUEUE.json')['attempts_started'] == 8,
         'Prior attempt accounting changed')
    current = read(HERE / 'continuation_right_01/RESULT.json')
    receipt = read(HERE / 'continuation_right_01/RESTORATION.json')
    need(current['status'] == 'COMPLETE' and current['error'] is None
         and not current['cleanup_errors'] and current['completed_continuation_ms'] == 300
         and current['initial_exact'] is True, 'Corrected continuation incomplete')
    need(receipt['initial']['exact'] is True
         and receipt['construction']['neural_or_physical_steps_during_restore'] == 0
         and receipt['construction']['field_metadata_exact'] is True,
         'Restore changed the saved initial boundary')
    need(0 < current['wall_total_s'] <= contract['budget']['new_run_wall_s_max'],
         'New attempt exceeds finite per-run budget')
    source_map = read(HERE / 'CONTINUATION_SOURCES.json')
    run_contract = read(HERE / 'continuation_right_01/RUN_CONTRACT.json')
    need(run_contract['sources'] == source_map
         and run_contract['plan_sha256'] == contract['original_plan_sha256'],
         'Executed continuation used different sources or plan')
    need(source_map[str(HERE / 'restore_checkpoint.py')] == sha(HERE / 'restore_checkpoint.py')
         and source_map[str(HERE / 'run_continuation.py')] == sha(HERE / 'run_continuation.py'),
         'Corrected loader was not frozen')
    need(run_contract['engine'] == 'reference_cuda' and run_contract['odor'] == 'odor_right'
         and run_contract['first_step'] == 101 and run_contract['last_step'] == 400
         and run_contract['field_reinstalled'] is False and run_contract['baseline_reset'] is False
         and run_contract['checkpoint_manifest_sha256']
         == sha(OLD / 'reference_odor_right_01/state_100ms/MANIFEST.json'),
         'Corrected continuation changed profile, interval or initial boundary')
    posture = {}
    for arm in ('sham', 'odor_left', 'odor_right', 'uniform'):
        reference = HERE / ('reference_' + arm + '_01')
        sources(reference, 'reference', read(OLD / 'EXECUTION_SOURCES.json'),
                contract['original_plan_sha256'])
        with np.load(reference / 'traces.npz', allow_pickle=False) as z:
            posture['reference_' + arm] = upright(z['qpos'], z['upright'])
        with np.load(NATIVE / ('full_' + arm + '_01') / 'traces.npz', allow_pickle=False) as z:
            posture['native_' + arm] = upright(z['qpos'], z['upright'])
    for arm in ('odor_left', 'odor_right'):
        withdrawal = HERE / ('withdrawal_' + arm + '_01')
        sources(withdrawal, 'withdrawal', read(OLD / 'WITHDRAWAL_SOURCES.json'),
                contract['original_plan_sha256'])
        with np.load(withdrawal / 'traces.npz', allow_pickle=False) as z:
            posture['withdrawal_' + arm] = upright(z['qpos'], z['upright'])
    with np.load(HERE / 'continuation_right_01/traces.npz', allow_pickle=False) as z:
        need(len(z['qpos']) == 300, 'Continuation trace length changed')
        posture['continuation_right'] = upright(z['qpos'], z['upright'])

    raw = verify_original(HERE, NATIVE)
    need(raw['functional_stage3_pass'] is True, 'Frozen scientific verifier failed')
    aggregate = (float(raw['organism_wall_s'])
                 + float(contract['prior_incomplete']['interrupted_uniform_wall_s'])
                 + float(failed['wall_total_s']))
    need(aggregate <= contract['budget']['aggregate_all_attempts_wall_s_max'],
         'All-attempt aggregate wall budget exceeded')
    need(raw['organism_runs'] == 7 and contract['budget']['complete_runs_max'] == 7
         and contract['budget']['prior_attempts'] == 8
         and contract['budget']['all_attempts_max'] == 9,
         'Success and failure accounting changed')
    return {
        'schema': 'stage3_restore_bugfix_raw_verified_v1',
        'classification': 'CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
        'functional_stage3_pass': True,
        'strict_original_contract_fulfilled': False,
        'attempts_total': 9,
        'complete_organism_runs': 7,
        'incomplete_attempts': 2,
        'aggregate_all_attempts_wall_s': aggregate,
        'aggregate_limit_s': contract['budget']['aggregate_all_attempts_wall_s_max'],
        'previous_failure_preserved': True,
        'changed_operation': 'Rebind cached outer body identity to the restored carrier body',
        'upright_reconstructed_from_pose': posture,
        'original_raw_result': raw,
        'limitations': [
            'The three initial references and the failed restore were observed before this repair; confirmation is not blind.',
            'The historical KC hidden-state gate remains failed.',
            'This is functional orientation in a PN629-off model with an engineered DNb05 decoder and roller body, not biological equivalence.',
            'A fixed binary lateral field does not establish navigation to a localized source.',
            'This functional gate does not establish a general brain engine or the strong intelligence objective.'
        ],
        'contract_sha256': sha(HERE / 'REPAIR19_CONTRACT.json'),
        'verifier_sha256': sha(__file__)
    }


if __name__ == '__main__':
    result = verify()
    print(json.dumps({k: v for k, v in result.items() if k != 'original_raw_result'}, indent=2, allow_nan=False))
