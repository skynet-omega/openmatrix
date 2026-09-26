"""CPU-only supplement to the unchanged, frozen review12 pair verifier.

Checks executor identity and the existing runtime/event/coupling contracts.
It adds no scientific tolerances, performance limits, or neural admission gate.
Run alongside verify_runs.py; PASS_RUNTIME_CONTRACT alone does not admit a pair.
Only standard-library code is imported; no shared library or GPU is loaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[1]
FLYWIRE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
ENGINE = HERE / 'engine'
EVENT_OWNER = ROOT / 'campanas/etapa3_pn629_intervention_20260923_15'
FROZEN_SOURCES_SHA256 = '0ee67395796b3a3399581a42667e5f1130d8744499124725e7f25bca5bdd5bea'
REVIEWED_METHOD = 'resident_RK3(2)_CSR_PERSISTENT_FP32'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=reject)


def exact(value, expected, label):
    need(type(value) is type(expected) and value == expected, label)


def count(value, label):
    need(type(value) is int and value >= 0, 'Invalid counter: ' + label)
    return value


def implementations(engine):
    return {
        'membrane': ('device_cell', EVENT_OWNER / 'device_cell.py'),
        'CNS': (('real_model', ENGINE / 'real_model.py') if engine == 'reviewed' else
                ('organism_adapter', ROOT / 'campanas/etapa3_motor_nuevo_20260922/organism_adapter.py')),
        'PN': ('pn_inhibitory_closure_source', FLYWIRE / 'src/pn_inhibitory_closure_source.py'),
    }


def source_record(record, path, lock, label, module=None):
    need(isinstance(record, dict), 'Missing source record: ' + label)
    exact(record.get('source'), str(path), 'Wrong implementation source: ' + label)
    need(str(path) in lock, 'Implementation absent from frozen SOURCES: ' + label)
    exact(record.get('sha256'), lock[str(path)], 'Wrong implementation hash: ' + label)
    if module is not None:
        exact(record.get('module'), module, 'Wrong implementation module: ' + label)


def validate_runtime(result, events, engine, ms, lock):
    """Cross-check runtime reports against source identities and event records.

    ms=20 is accepted here only for a fixture using the existing diagnostic.
    The admission CLI below permits complete 100/2000ms pairs exclusively.
    """
    need(engine in ('stable', 'reviewed'), 'Unknown engine')
    runtime = result.get('runtime')
    need(isinstance(runtime, dict), 'Missing runtime report')
    installed = runtime.get('installed_implementations')
    need(isinstance(installed, dict) and set(installed) == {'membrane', 'CNS', 'PN'},
         'Missing/extra installed scientific owners')
    for owner, (module, path) in implementations(engine).items():
        source_record(installed[owner], path, lock, owner, module)
    exact(runtime.get('profile'), 'causal_cuda', 'Wrong runtime profile')
    exact(runtime.get('event_boundaries'), True, 'Event boundaries disabled')
    exact(runtime.get('geometry_compression'), False, 'Geometry compression changed')
    exact(runtime.get('coupling_ns'), 125000, 'Runtime coupling changed')

    overlay = result.get('pn_overlay')
    need(isinstance(overlay, dict) and set(overlay) == {'cyclic_tree', 'resident_cyclic'},
         'Missing/extra PN overlay owners')
    for name, record in overlay.items():
        source_record(record, HERE / 'pn' / (name + '.py'), lock, 'PN overlay ' + name)

    for owner in ('cell', 'CNS', 'PN', 'events', 'partition'):
        need(isinstance(runtime.get(owner), dict), 'Missing runtime owner: ' + owner)
    cell, cns, pn, event, partition = (runtime[name] for name in
                                     ('cell', 'CNS', 'PN', 'events', 'partition'))
    # Every 125us exchange runs a discarded 62.5us CNS/cell predictor and a
    # committed 125us CNS/cell advance. PN advances twice by 62.5us.
    epochs = 16 * ms
    exact(cns.get('epochs'), epochs, 'CNS epoch count disagrees with duration')
    exact(cell.get('calls'), epochs, 'Cell call count disagrees with duration')
    exact(pn.get('calls'), epochs, 'PN call count disagrees with duration')
    exact(pn.get('accepted'), epochs, 'PN accepted count disagrees with duration')
    exact(pn.get('rejected'), 0, 'PN rejection in complete run')
    exact(pn.get('state_owner_preserved'), True, 'PN state owner changed')
    need(isinstance(pn.get('model_identity'), str) and bool(pn['model_identity']), 'Missing PN identity')
    exact(cns.get('mandatory_event_boundaries'), True, 'CNS event boundaries disabled')
    exact(cell.get('compressed'), False, 'Cell geometry compressed')
    exact(cell.get('scheduler'), 'device_independent_block', 'Wrong cell scheduler')
    exact(cell.get('count_unit'), 'cell_trials', 'Wrong cell counter unit')
    exact(partition.get('step_ns'), 125000, 'Partition coupling changed')
    exact(partition.get('steps'), 8 * ms, 'Partition exchange count disagrees with duration')
    exact(partition.get('predictor_restore_exact'), True, 'Predictor restoration not exact')
    exact(partition.get('pn_rejections'), 0, 'Partition reports PN rejection')
    accepted, rejected = (count(cns.get(name), 'CNS/' + name) for name in ('accepted', 'rejected'))
    need(accepted >= epochs, 'CNS accepted trials cannot cover every nonempty epoch')
    need(count(cell.get('accepted'), 'cell/accepted') > 0, 'No accepted physical cell trials')
    count(cell.get('rejected'), 'cell/rejected')

    exact(event.get('blocks'), epochs, 'Event block count disagrees with duration')
    exact(event.get('global_trials'), accepted + rejected, 'Event/CNS trial accounting differs')
    exact(event.get('time_resolved'), True, 'Event waveform is not time resolved')
    # This is the already enforced event_coupling.py:86 invariant, not a new
    # stable/candidate event-time tolerance or neural-equivalence threshold.
    error = event.get('max_q_reconstruction_difference')
    need(type(error) in (float, int) and math.isfinite(error) and 0. <= error <= 2e-12,
         'Existing physical-q reconstruction invariant failed')
    need(isinstance(events, list) and len(events) == epochs, 'Missing event audit epochs')
    total_events = 0
    for block in events:
        need(isinstance(block, dict) and isinstance(block.get('events'), list), 'Malformed event audit')
        total_events += len(block['events'])
    exact(event.get('events'), total_events, 'Event runtime total differs from raw audit')

    if engine == 'reviewed':
        exact(result.get('engine_path'), str(ENGINE), 'Wrong reviewed engine_path')
        library = ENGINE / 'libresident_controller.so'
        exact(result.get('engine_library_sha256'), lock[str(library)], 'Wrong reviewed library hash')
        exact(cns.get('method'), REVIEWED_METHOD, 'Wrong reviewed CNS method')
        exact(cns.get('rhs_evaluations'), 4 * (accepted + rejected), 'Wrong RK3(2) RHS accounting')
        need(isinstance(result.get('weight_mirror'), dict), 'Missing persistent FP32 mirror report')
    else:
        need('engine_path' not in result and 'engine_library_sha256' not in result,
             'Stable run carries reviewed-only engine identity')
        need('method' not in cns and 'rhs_evaluations' not in cns,
             'Stable adapter report contains reviewed-only method accounting')
        library = ROOT / 'motor_nuevo/native_hybrid_20260922/libgraph_control_v2.so'
    return {'installed_implementations': installed, 'pn_overlay': overlay,
            'native_library': {'path': str(library), 'sha256': lock[str(library)],
                               'identity_basis': 'Frozen binary; not loaded by this CPU verifier'},
            'CNS_method': cns.get('method', 'stable exponential midpoint (frozen organism_adapter/graph_core)'),
            'CNS_accepted': accepted, 'CNS_rejected': rejected, 'epochs': epochs,
            'partition_exchanges': partition['steps'], 'coupling_ns': partition['step_ns'],
            'predictor_restore_exact': partition['predictor_restore_exact'],
            'event_records_including_discarded_predictors': total_events,
            'max_q_reconstruction_difference': error, 'PN_model_identity': pn['model_identity']}


def frozen_sources():
    path = HERE / 'SOURCES.json'
    exact(sha(path), FROZEN_SOURCES_SHA256, 'Frozen SOURCES manifest changed')
    lock = read_json(path)
    # Runtime sources and immediate numerical executors. The original verifier
    # remains responsible for the full frozen code/data inventory.
    required = {HERE / name for name in ('PLAN.json', 'run_trial.py', 'verify_runs.py',
                                         'restore_prepared.py', 'source_inventory.py')}
    required.update(path for engine in ('stable', 'reviewed') for _, path in implementations(engine).values())
    required.update(HERE / 'pn' / (name + '.py') for name in ('install_pn', 'cyclic_tree', 'resident_cyclic'))
    required.update(ENGINE / name for name in ('graph_runtime.py', 'resident_controller.cu',
                                              'libresident_controller.so', 'fp32_operator.py',
                                              'persistent_weights.py', 'model_weight_writers.py',
                                              'persistent_fp32.cu', 'fast_fp32_coefficient.cu'))
    required.update((ROOT / 'campanas/etapa3_motor_nuevo_20260922/graph_core.py',
                     ROOT / 'campanas/etapa3_motor_nuevo_20260922/pn_execution.py',
                     ROOT / 'motor_nuevo/native_hybrid_20260922/libgraph_control_v2.so',
                     ROOT / 'motor_nuevo/pipeline_review_20260922/runtime_session.py',
                     EVENT_OWNER / 'event_coupling.py', FLYWIRE / 'work/motor13_20260922/block_midpoint.py'))
    checked = {}
    for source in sorted(required):
        key = str(source)
        need(key in lock, 'Runtime source absent from frozen SOURCES: ' + key)
        exact(sha(source), lock[key], 'Runtime source changed: ' + key)
        checked[key] = lock[key]
    return lock, checked


def compare(stable, reviewed, ms):
    need(ms in (100, 2000), 'Only complete 100/2000ms pairs are admitted')
    stable, reviewed = Path(stable).resolve(), Path(reviewed).resolve()
    need(stable != reviewed, 'Two distinct run folders required')
    lock, checked = frozen_sources()
    plan_digest = lock[str(HERE / 'PLAN.json')]
    summaries, artifacts = {}, {}
    for folder, engine in ((stable, 'stable'), (reviewed, 'reviewed')):
        result = read_json(folder / 'RESULT.json')
        exact(result.get('status'), 'COMPLETE', engine + ': incomplete run')
        need(not result.get('error') and not result.get('cleanup_errors'), engine + ': runtime/cleanup error')
        exact(result.get('engine'), engine, 'Wrong run label')
        exact(result.get('requested_ms'), ms, 'Wrong requested duration')
        exact(result.get('completed_ms'), ms, 'Wrong completed duration')
        exact(result.get('sources_sha256'), FROZEN_SOURCES_SHA256, 'Run has wrong frozen SOURCES')
        exact(result.get('plan_sha256'), plan_digest, 'Run has wrong frozen PLAN')
        exact(result.get('runner_sha256'), lock[str(HERE / 'run_trial.py')], 'Wrong frozen runner')
        exact(result.get('restore_sha256'), lock[str(HERE / 'restore_prepared.py')], 'Wrong frozen restorer')
        events = read_json(folder / 'EVENTS.json')
        summaries[engine] = validate_runtime(result, events, engine, ms, lock)
        expected_imports = {str(path) for _, path in implementations(engine).values()}
        expected_imports.update(str(HERE / 'pn' / (name + '.py')) for name in ('cyclic_tree', 'resident_cyclic'))
        for name in ('EXECUTED_SOURCES_SETUP.json', 'EXECUTED_SOURCES_FINAL.json'):
            inventory = read_json(folder / name)
            need(isinstance(inventory, dict) and expected_imports <= set(inventory),
                 engine + ': required runtime owner absent from ' + name)
            need(set(inventory) <= set(lock), engine + ': source inventory escapes frozen SOURCES')
            need(all(digest == lock[path] for path, digest in inventory.items()), engine + ': imported source hash differs')
        for name in ('RESULT.json', 'EVENTS.json', 'EXECUTED_SOURCES_SETUP.json', 'EXECUTED_SOURCES_FINAL.json'):
            artifacts[str(folder / name)] = sha(folder / name)
    exact(summaries['stable']['PN_model_identity'], summaries['reviewed']['PN_model_identity'], 'Pair PN identities differ')
    return {'status': 'PASS_RUNTIME_CONTRACT', 'ms': ms, 'runs': summaries,
            'scope': 'Executor identity and existing runtime contracts only; use alongside frozen verify_runs.py. No new neural, event-equivalence, timing or scientific threshold.',
            'limitation': 'Cross-checks frozen callable-source reports and event accounting, not independent attestation of every GPU instruction or neural accuracy.',
            'provenance': {'supplement_sha256': sha(__file__), 'sources_sha256': FROZEN_SOURCES_SHA256,
                           'plan_sha256': plan_digest, 'runtime_sources_verified': checked}, 'artifacts': artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable', type=Path, required=True)
    parser.add_argument('--reviewed', type=Path, required=True)
    parser.add_argument('--ms', type=int, choices=(100, 2000), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'Supplement output must be new')
    try:
        report = compare(args.stable, args.reviewed, args.ms)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {'status': 'INVALID_RUNTIME_CONTRACT', 'ms': args.ms,
                  'error': {'type': type(exc).__name__, 'message': str(exc)},
                  'supplement_sha256': sha(__file__), 'expected_sources_sha256': FROZEN_SOURCES_SHA256}
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'ms', 'error') if key in report}, indent=2))
    return 0 if report['status'] == 'PASS_RUNTIME_CONTRACT' else 2


if __name__ == '__main__':
    raise SystemExit(main())
