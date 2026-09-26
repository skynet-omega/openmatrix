"""Independent CPU verifier of complete review12 pairs; no organism imports.

Functional gates come from PLAN.json. Neural/event differences are reported
without new numerical acceptance thresholds. Timing is engineering evidence.
"""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import math
import zipfile

import numpy as np
from comparison_math import array_error, load_arrays

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARED = ROOT / 'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state'
HISTORICAL_MOTOR = ROOT / 'campanas/etapa45_navigation_wind_20260925_40/motor_wind.py'
NS_PER_MS = 1_000_000
BODY_NS = 25_000
TRACE_SHAPES = {
    'fase': (), 'paso': (), 'CNS_time_ns': (), 'PN_time_ns': (), 'body_time_ns': (),
    'sensores_usados': (3,), 'sensores_pendientes': (3,), 'antenas_mm': (2, 3),
    'concentracion_campo': (2,), 'ORN_q_L': (35,), 'ORN_q_R': (39,),
    'ORN_filters': (4, 74), 'PN_q_legacy': (2,), 'PN_general_transmission': (629,),
    'PN_gamma_nS': (100,), 'PN_additional_nS': (366,), 'DN_q_actual': (4,),
    'DN_q_usada': (4,), 'DN_baseline': (4,), 'command_forward_mm_s': (),
    'command_yaw_rate_rad_s': (), 'qpos': (109,), 'qvel': (108,),
    'position_mm': (3,), 'yaw_delta_deg': (), 'upright': (), 'contact_active': (6,),
    'normal_force_N': (6,), 'contact_force_N': (6, 3),
    'generalized_force_native': (108,), 'energy_motor_J': (),
    'neural_command_raw_rad_s': (), 'motor_filter_applied_rad_s': (),
    'motor_filter_state_rad_s': (), 'wind_torque_native': (),
}
NEURAL_FIELDS = {
    'time_ns', 'cns', 'body_qpos', 'body_qvel', 'pending_sensors',
    'cell_delta', 'cell_gates', 'cell_q', 'cell_counts', 'cell_clipped',
    'cell_last_siz', 'cell_previous_slope', 'cell_trough',
    'axon_q', 'axon_s', 'axon_last_voltage', 'axon_previous_slope',
    'axon_trough', 'axon_counts', 'axon_clipped',
}
PN_CLOCKS = {
    '/base/base/base/base/base/time_ns',
    '/base/base/base/base/base/pn/pn/time_ns',
    '/base/base/base/base/base/pn/calcium/time_ns',
    '/base/base/base/base/base/pn/calcium/chemistry/time_ns',
    '/base/base/base/base/ach/time_ns', '/base/base/base/extra_output/time_ns',
    '/base/base/general_output/time_ns', '/base/apl_receptor/time_ns',
    '/receptors/GABA/time_ns', '/receptors/Glu/time_ns',
}
INITIAL_ARRAYS = {'session': 574, 'prosthesis': 5, 'published': 1,
                  'effective_operator': 8, 'boundary': 0}


def need(ok, message):
    if not ok:
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
    def nonfinite(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=nonfinite)


def finite(array, label):
    need(array.dtype.kind in 'biufcUS', 'Unsupported array dtype: ' + label)
    if array.dtype.kind in 'biufc':
        need(np.isfinite(array).all(), 'Nonfinite array: ' + label)


def protocol(plan=None):
    """Read literal historical constants without importing MuJoCo or motor code."""
    tree = ast.parse(HISTORICAL_MOTOR.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MotorWind')
    wanted = {'WIND_FIRST_STEP', 'WIND_LAST_STEP', 'WIND_TORQUE_NATIVE', 'TAU_S', 'THRESHOLD_RAD_S'}
    values = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted:
                values[key] = ast.literal_eval(node.value)
    need(set(values) == wanted, 'Historical motor constants incomplete')
    need(values['WIND_FIRST_STEP'] == 1001 and values['WIND_LAST_STEP'] == 1020,
         'Historical wind interval changed')
    need(values['WIND_TORQUE_NATIVE'] == -0.004672697857153467,
         'Historical literal wind torque changed')
    if plan is not None:
        expected = {'trial_continuous': True, 'body_dt_s': BODY_NS * 1e-9,
                    'wind_first_step': values['WIND_FIRST_STEP'], 'wind_last_step': values['WIND_LAST_STEP'],
                    'world_torque_native': [0., 0., values['WIND_TORQUE_NATIVE']],
                    'expected_nonzero_wind_substeps': 800,
                    'mapping': 'current_pose_scratch_kinematics_comPos'}
        need(plan.get('physics') == expected, 'PLAN physical protocol differs from literal historical protocol')
    return values


def validate_traces(z, ms, origin, label):
    need(set(z) == set(TRACE_SHAPES), label + ': expected all 35 trace fields')
    for key, shape in TRACE_SHAPES.items():
        a = z[key]
        need(a.shape == (ms,) + shape, label + ': trace length/layout ' + key)
        finite(a, label + '/' + key)
        kind = 'U' if key == 'fase' else ('b' if key == 'contact_active' else
                 ('i' if key in ('paso', 'CNS_time_ns', 'PN_time_ns', 'body_time_ns') else 'f'))
        need(a.dtype.kind == kind, label + ': trace dtype ' + key)
    need(np.all(z['fase'] == 'ensayo'), label + ': wrong trace phase')
    need(np.array_equal(z['paso'], np.arange(1, ms + 1)), label + ': trace steps')
    clock = origin + np.arange(1, ms + 1, dtype=np.int64) * NS_PER_MS
    for key in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
        need(np.array_equal(z[key], clock), label + ': stale/wrong clock ' + key)
    need(np.array_equal(z['sensores_usados'][1:], z['sensores_pendientes'][:-1]), label + ': sensor lag')
    need(np.array_equal(z['DN_q_usada'][1:], z['DN_q_actual'][:-1]), label + ': DN lag')
    need(np.all(z['DN_baseline'] == z['DN_baseline'][0]), label + ': moving DN baseline')
    need(np.array_equal(z['position_mm'], z['qpos'][:, :3] * 10.), label + ': position units/qpos')


def saved_tree(prefix):
    """Decode only the small PN/publication/auxiliary trees; reject orphan arrays."""
    prefix = Path(prefix)
    metadata = read_json(prefix.with_suffix('.json'))
    arrays = load_arrays(prefix.with_suffix('.npz'))
    used = set()
    out = {}
    def walk(value, path):
        if isinstance(value, dict) and set(value) == {'__array__'}:
            name = value['__array__']
            need(isinstance(name, str) and name in arrays and name not in used, 'Bad array reference ' + path)
            used.add(name)
            finite(arrays[name], str(prefix) + path)
            out[path] = arrays[name]
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + '/' + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + '/' + str(index))
        else:
            if isinstance(value, float):
                need(math.isfinite(value), 'Nonfinite scalar ' + path)
            out[path] = value
    walk(metadata, '')
    need(used == set(arrays), 'Unreferenced arrays in ' + str(prefix))
    return out


def tree_difference(a, b):
    need(set(a) == set(b), 'Saved tree field schema differs')
    out = {}
    for path in a:
        x, y = a[path], b[path]
        if isinstance(x, np.ndarray):
            need(isinstance(y, np.ndarray), 'Array/scalar tree mismatch ' + path)
            out[path] = array_error(x, y)
        else:
            need(type(x) is type(y), 'Saved scalar type differs ' + path)
            out[path] = {'exact': x == y}
            if type(x) in (int, float):
                delta = abs(x - y)
                need(math.isfinite(delta), 'Nonfinite difference ' + path)
                out[path].update(max_abs=delta, changed_values=int(x != y), values=1)
    return out


def exact_tree(a, b, label):
    difference = tree_difference(a, b)
    need(all(item['exact'] for item in difference.values()), label)
    return difference


def tree_clocks(tree, required):
    clocks = {key: value for key, value in tree.items() if key.endswith('/time_ns')}
    need(set(clocks) == required, 'Missing/extra saved state clocks')
    need(all(type(value) is int for value in clocks.values()), 'Integer saved clocks required')
    return clocks


def validate_clock_progress(initial, current, elapsed_ns, required, label):
    first = tree_clocks(initial, required)
    last = tree_clocks(current, required)
    need(all(last[key] == value + elapsed_ns for key, value in first.items()), label + ': stale local state clock')
    return {key: {'initial_ns': value, 'final_ns': last[key], 'advance_ns': elapsed_ns}
            for key, value in first.items()}


def require_final_files(folder, ms):
    for name in ('pn_state', 'published'):
        for suffix in ('.json', '.npz'):
            need((folder / 'final_state' / (name + suffix)).is_file(), 'Missing final_state/' + name + suffix)
            need((folder / ('state_' + str(ms) + 'ms') / (name + suffix)).is_file(), 'Missing final observation tree')
    if ms == 2000:
        need((folder / 'scientific_final/MANIFEST.json').is_file(), 'Missing scientific final state')


def validate_neural(z, traces, ms, label):
    need(set(z) == NEURAL_FIELDS, label + ': incomplete neural fields')
    steps = [step for step in (0, 1, 20, 100, 1000, 1020, 1500, 2000) if step <= ms]
    clock = z['time_ns']
    need(clock.dtype.kind == 'i' and clock.shape == (len(steps),), label + ': neural timestamps')
    origin = int(clock[0])
    need(np.array_equal(clock, origin + np.asarray(steps) * NS_PER_MS), label + ': missing/stale neural sample')
    for key, a in z.items():
        shape = (() if key == 'time_ns' else (359373,) if key == 'cns' else
                 (109,) if key == 'body_qpos' else (108,) if key == 'body_qvel' else
                 (3,) if key == 'pending_sensors' else (1557, 17) if key == 'cell_delta' else
                 (1557, 17, 4) if key == 'cell_gates' else (1557, 12) if key.startswith('axon_') else (1557,))
        need(a.shape == (len(steps),) + shape, label + ': neural length/layout ' + key)
        kind = 'i' if key == 'time_ns' or key.endswith(('_counts', '_clipped')) else 'f'
        need(a.dtype.kind == kind, label + ': neural dtype ' + key)
        finite(a, label + '/neural/' + key)
    need(z['cns'].shape == (len(steps), 359373), label + ': CNS layout')
    for index, step in enumerate(steps):
        if step:
            for field, trace in [('body_qpos', 'qpos'), ('body_qvel', 'qvel'), ('pending_sensors', 'sensores_pendientes')]:
                need(np.array_equal(z[field][index], traces[trace][step-1]), label + ': neural/trace observation differs ' + field)
    need(np.array_equal(z['pending_sensors'][0], traces['sensores_usados'][0]), label + ': first sensory input differs')
    return origin, steps


def validate_initial(proof, checkpoint):
    need(proof.get('method') == 'Direct recursive key/length/type/shape/dtype/value comparison; no structural hash',
         'Initial proof is not a direct tree comparison')
    trees = proof.get('trees', {})
    need(set(trees) == set(INITIAL_ARRAYS), 'Missing initial scientific owner')
    for name, count in INITIAL_ARRAYS.items():
        record = trees[name]
        need(record.get('exact') is True and record.get('different_paths') == [], 'Initial tree differs: ' + name)
        need(record.get('arrays') == count, 'Initial proof array coverage differs: ' + name)
        for key in ('array_values', 'scalar_leaves', 'containers'):
            need(type(record.get(key)) is int and record[key] >= 0, 'Initial proof coverage missing')
    need(sum(record['arrays'] for record in trees.values()) == 588 and proof.get('exact') is True,
         'Initial 588-array direct proof incomplete')
    need(proof.get('checkpoint_manifest_sha256') == sha(checkpoint/'MANIFEST.json'), 'Initial checkpoint identity changed')


def validate_wind(folder, traces, ms, origin, constants=None):
    constants = protocol() if constants is None else constants
    torque = constants['WIND_TORQUE_NATIVE']
    first, last = constants['WIND_FIRST_STEP'], constants['WIND_LAST_STEP']
    motor = read_json(folder/'MOTOR.json')
    count = 0 if ms == 100 else 800
    expected_trace = np.where((traces['paso'] >= first) & (traces['paso'] <= last), torque, 0.)
    need(np.array_equal(traces['wind_torque_native'], expected_trace), 'Wrong per-tick wind torque')
    expected = {'trial_steps': ms, 'body_calls': 40*ms, 'wind_substeps': count,
                'expected_wind_substeps': 800, 'nonzero_substeps': count,
                'wind_first_step': first, 'wind_last_step': last,
                'wind_torque_native': torque, 'wind_torque_Nm': torque*1e-7,
                'tau_s': constants['TAU_S'], 'threshold_rad_s': constants['THRESHOLD_RAD_S'],
                'output_rad_s': math.radians(5.0),
                'mapping': 'current_pose_scratch_kinematics_comPos',
                'physical_restart_during_trial': False}
    for key, value in expected.items():
        need(type(motor.get(key)) is type(value) and motor[key] == value, 'Wrong motor/wind audit ' + key)
    need(type(motor.get('wind_generalized_peak_native')) in (int, float) and
         math.isfinite(motor['wind_generalized_peak_native']), 'Nonfinite/missing wind audit peak')
    path = folder/'wind_substeps.npz'
    if count == 0:
        need(motor['wind_generalized_peak_native'] == 0., 'Unexpected wind before step1001')
        if path.exists():
            data = load_arrays(path)
            need(data and all(a.ndim and a.shape[0] == 0 for a in data.values()), 'Unexpected wind calls in100ms')
        return {'count': 0, 'audit': motor}, None
    need(path.is_file(), 'Missing physical wind log')
    z = load_arrays(path)
    shapes = {'trial_step': (800,), 'body_call': (800,), 'body_time_s': (800,),
              'world_torque': (800, 3), 'generalized': (800, 108), 'qpos': (800, 109)}
    need(set(z) == set(shapes), 'Wind log field schema')
    for key, shape in shapes.items():
        need(z[key].shape == shape, 'Wind log length/layout ' + key)
        finite(z[key], 'wind/' + key)
        need(z[key].dtype.kind == ('i' if key in ('trial_step', 'body_call') else 'f'), 'Wind log dtype ' + key)
    need(np.array_equal(z['trial_step'], np.repeat(np.arange(1001, 1021), 40)), 'Wrong wind trial steps')
    calls = np.arange(40001, 40801)
    need(np.array_equal(z['body_call'], calls), 'Wrong wind body calls')
    need(np.array_equal(z['world_torque'], np.tile([0., 0., torque], (800, 1))), 'Wrong literal world torque')
    need(np.all(np.any(z['generalized'] != 0., axis=1)), 'Wind mapped to zero on a scheduled call')
    need(motor['wind_generalized_peak_native'] == float(np.max(np.abs(z['generalized']))), 'Wind audit peak disagrees with log')
    expected_seconds = (origin + (calls-1)*BODY_NS) * 1e-9
    # Allow only a derived accumulated floating-point clock roundoff bound,
    # not a fitted timing tolerance; body clocks themselves are checked in ns.
    roundoff = (origin//BODY_NS + calls) * np.spacing(np.maximum(1., np.abs(expected_seconds)))
    need(np.all(np.diff(z['body_time_s']) > 0.) and
         np.all(np.abs(z['body_time_s'] - expected_seconds) <= roundoff), 'Wind substep physical time disagrees')
    return {'count': count, 'audit': motor, 'max_body_clock_roundoff_s': float(np.max(np.abs(z['body_time_s']-expected_seconds)))}, z


def event_difference(left, right, ms):
    datasets = [read_json(left), read_json(right)]
    need(all(isinstance(data, list) and len(data) == 16*ms for data in datasets), 'Incomplete event blocks')
    origin = datasets[0][0]['start_elapsed_ns']
    result = {kind: {'stable_count': 0, 'reviewed_count': 0, 'identity_different_blocks': 0,
                    'ordered_identity_equal_blocks': 0, 'payload_compared_events': 0,
                    'payload_exact_events': 0, 'excluded_stable_events': 0, 'excluded_reviewed_events': 0,
                    'max_time_difference_s': 0., 'max_jump_difference': 0., 'max_post_q_difference': 0.,
                    'post_q_presence_differences': 0}
              for kind in ('committed', 'predictor')}
    totals = {kind: [collections.Counter(), collections.Counter()] for kind in result}
    first_difference = None
    for index, pair in enumerate(zip(*datasets)):
        kind = 'predictor' if index % 2 == 0 else 'committed'
        duration = 62500 if kind == 'predictor' else 125000
        start = origin + (index//2)*125000
        signatures = []
        for side, block in enumerate(pair):
            need(set(block) == {'block', 'start_elapsed_ns', 'duration_ns', 'events'}, 'Event block schema')
            need(block['block'] == index and block['start_elapsed_ns'] == start and
                 block['duration_ns'] == duration, 'Event block clock/identity')
            need(isinstance(block['events'], list), 'Event list required')
            for event in block['events']:
                need(set(event) == {'row', 'neuron_id', 'producer', 'time_s', 'jump', 'post_q'}, 'Event payload schema')
                need(type(event['row']) is int and event['row'] >= 0 and type(event['neuron_id']) is int and
                     event['producer'] in ('gamma_cuda', 'nongamma_lif'), 'Invalid event identity')
                need(type(event['time_s']) in (float, int) and math.isfinite(event['time_s']) and
                     0 <= event['time_s'] <= duration*1e-9, 'Invalid event time')
                need(type(event['jump']) in (float, int) and math.isfinite(event['jump']) and
                     (event['post_q'] is None or (type(event['post_q']) in (float, int) and math.isfinite(event['post_q']))),
                     'Nonfinite event payload')
            signature = [(event['row'], event['neuron_id'], event['producer']) for event in block['events']]
            signatures.append(signature)
            totals[kind][side].update(signature)
            result[kind]['stable_count' if side == 0 else 'reviewed_count'] += len(signature)
        row = result[kind]
        if collections.Counter(signatures[0]) != collections.Counter(signatures[1]):
            row['identity_different_blocks'] += 1
            if first_difference is None:
                first_difference = {'kind': kind, 'block': index, 'relative_start_ns': start-origin,
                                    'stable': pair[0], 'reviewed': pair[1]}
        if signatures[0] != signatures[1]:
            row['excluded_stable_events'] += len(signatures[0])
            row['excluded_reviewed_events'] += len(signatures[1])
            continue
        row['ordered_identity_equal_blocks'] += 1
        for a, b in zip(pair[0]['events'], pair[1]['events']):
            row['payload_compared_events'] += 1
            row['payload_exact_events'] += int(a == b)
            row['max_time_difference_s'] = max(row['max_time_difference_s'], abs(a['time_s']-b['time_s']))
            row['max_jump_difference'] = max(row['max_jump_difference'], abs(a['jump']-b['jump']))
            if a['post_q'] is None or b['post_q'] is None:
                row['post_q_presence_differences'] += int(a['post_q'] != b['post_q'])
            else:
                row['max_post_q_difference'] = max(row['max_post_q_difference'], abs(a['post_q']-b['post_q']))
    for kind, (a, b) in totals.items():
        result[kind].update(same_total_identity_counts=a == b,
                            unmatched_stable_total=sum((a-b).values()), unmatched_reviewed_total=sum((b-a).values()))
    return {'by_context': result, 'first_identity_difference': first_difference,
            'scope': 'Payload compared only in blocks with identical ordered identities; exclusions counted. No event tolerance or new admission gate.'}


def verify_sources(lock):
    need(isinstance(lock, dict) and bool(lock), 'Empty source lock')
    for path, digest in lock.items():
        need(Path(path).is_file() and sha(path) == digest, 'Source changed: ' + path)


def verify_scientific(folder, clock, final_published):
    """Hash every scientific artifact and scan all numeric array owners serially."""
    folder = folder/'scientific_final'
    manifest = read_json(folder/'MANIFEST.json')
    need(manifest.get('schema') == 'review12_scientific_boundary_v1' and manifest.get('time_ns') == clock,
         'Scientific snapshot clock/schema')
    expected = {stem+suffix for stem in ('session', 'prosthesis', 'published', 'effective_operator', 'motor')
                for suffix in ('.json', '.npz')} | {'boundary.json', 'GAUSSIAN_INTERVALS.json'}
    need(set(manifest.get('files', {})) == expected, 'Incomplete scientific snapshot owners')
    need({path.name for path in folder.iterdir() if path.is_file()} == expected | {'MANIFEST.json'}, 'Scientific snapshot file set differs')
    counts = {}
    for name, digest in manifest['files'].items():
        path = folder/name
        need(sha(path) == digest, 'Scientific snapshot hash differs: ' + name)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    finite(z[key], 'scientific/' + name + '/' + key)
                counts[name] = len(z.files)
    exact_tree(saved_tree(folder/'published'), final_published, 'Scientific publication differs from final_state')
    motor = saved_tree(folder/'motor')
    need(motor['/trial_step'] == 2000 and motor['/body_calls'] == 80000 and motor['/wind_substeps'] == 800,
         'Scientific motor counters differ')
    # Check all small scalar trees too; the enormous historical session JSON
    # is content-addressed above, never used as a substitute for live PN output.
    saved_tree(folder/'effective_operator')
    read_json(folder/'boundary.json')
    read_json(folder/'GAUSSIAN_INTERVALS.json')
    return {'manifest_sha256': sha(folder/'MANIFEST.json'), 'array_counts': counts,
            'published_exact_to_final_state': True, 'restart_tested': manifest.get('restart_tested'),
            'scope': 'All scientific files hash-checked, all arrays finite; small auxiliary trees checked. Full historical session/prosthesis scalar diff is not a neural-admission metric.'}


def compare(stable, reviewed, ms):
    need(ms in (100, 2000), 'Only complete100ms or2000ms pairs are admitted')
    stable, reviewed = Path(stable).resolve(), Path(reviewed).resolve()
    need(stable != reviewed, 'Pair must contain two runs')
    plan = read_json(HERE/'PLAN.json')
    limits = plan['comparison_limits_engineering_only']
    constants = protocol(plan)
    lock = read_json(HERE/'SOURCES.json')
    verify_sources(lock)
    plan_digest, lock_digest = sha(HERE/'PLAN.json'), sha(HERE/'SOURCES.json')
    runs = []
    for folder, engine in ((stable, 'stable'), (reviewed, 'reviewed')):
        result = read_json(folder/'RESULT.json')
        need(result.get('status') == 'COMPLETE' and not result.get('cleanup_errors') and not result.get('error'), 'Run incomplete or failed')
        need(result.get('engine') == engine and result.get('requested_ms') == result.get('completed_ms') == ms,
             'Pair engine/duration mismatch')
        need(result.get('parameters_changed') is False, 'Run reports changed biological/numerical parameters')
        need(result.get('plan_sha256') == plan_digest and result.get('sources_sha256') == lock_digest, 'Pair provenance differs from frozen live sources/plan')
        need(result.get('runner_sha256') == sha(HERE/'run_trial.py') and result.get('restore_sha256') == sha(HERE/'restore_prepared.py'), 'Runner/restorer identity differs')
        need(Path(result.get('source_checkpoint', '')).resolve() == PREPARED.resolve(), 'Wrong prepared checkpoint')
        need(result.get('source_trace_sha256') == sha(PREPARED.parent/'traces.npz'), 'Prepared trace identity differs')
        initial = read_json(folder/'INITIAL.json')
        validate_initial(initial, PREPARED)
        need(result.get('initial') == initial, 'RESULT and direct initial proof disagree')
        inventories = [read_json(folder/name) for name in ('EXECUTED_SOURCES_SETUP.json', 'EXECUTED_SOURCES_FINAL.json')]
        for inventory in inventories:
            need(bool(inventory) and set(inventory) <= set(lock), 'Executed source missing from lock')
            need(all(lock[path] == digest for path, digest in inventory.items()), 'Executed source content changed')
        need(set(inventories[0]) <= set(inventories[1]), 'Final imported-source evidence lost modules')
        require_final_files(folder, ms)
        traces = load_arrays(folder/'traces.npz')
        neural = load_arrays(folder/'neural_states.npz')
        need('time_ns' in neural and neural['time_ns'].ndim == 1 and neural['time_ns'].size > 0, 'Missing initial neural clock')
        origin = int(neural['time_ns'][0])
        validate_traces(traces, ms, origin, engine)
        origin, steps = validate_neural(neural, traces, ms, engine)
        trees = {}
        for step in [value for value in steps if value != 1]:
            trees[step] = {name: saved_tree(folder/f'state_{step}ms'/name) for name in ('pn_state', 'published')}
            need(trees[step]['published']['/time_ns'] == origin + step*NS_PER_MS, 'Published CNS clock differs')
        final = {name: saved_tree(folder/'final_state'/name) for name in ('pn_state', 'published')}
        for name in final:
            exact_tree(final[name], trees[ms][name], 'Final tree differs from final observation: ' + name)
        clocks = {}
        for step in trees:
            clocks[str(step)] = validate_clock_progress(trees[0]['pn_state'], trees[step]['pn_state'], step*NS_PER_MS, PN_CLOCKS, engine)
        validate_clock_progress(trees[0]['published'], final['published'], ms*NS_PER_MS, {'/time_ns'}, engine)
        wind, wind_arrays = validate_wind(folder, traces, ms, origin, constants)
        scientific = verify_scientific(folder, origin+ms*NS_PER_MS, final['published']) if ms == 2000 else None
        for key in ('advance_total_s', 'wall_total_s'):
            need(type(result.get(key)) in (float, int) and math.isfinite(result[key]) and result[key] > 0., 'Invalid timing ' + key)
        need(result['wall_total_s'] >= result['advance_total_s'], 'Timing counters impossible')
        wall_limit = (plan['short_pair']['wall_s_each_max'] if ms == 100 else
                      plan['confirmation']['stable_wall_s_max' if engine == 'stable' else 'candidate_wall_s_max'])
        need(result['wall_total_s'] <= wall_limit, engine + ': RESULT process time exceeds PLAN budget')
        rss = result.get('resources_peak_rss_bytes')
        need(type(rss) is int and 0 < rss < plan['resources']['RAM_GiB_max'] * 1024**3,
             engine + ': missing/exceeded RESULT peak RAM budget')
        pools = result.get('thread_pools')
        need(isinstance(pools, list) and any(pool.get('internal_api') == 'openblas' for pool in pools),
             'Missing OpenBLAS thread evidence')
        need(all(pool.get('num_threads') == plan['resources']['OPENBLAS_NUM_THREADS']
                 for pool in pools if pool.get('internal_api') == 'openblas'), 'OpenBLAS thread budget differs')
        runs.append(dict(result=result, initial=initial, traces=traces, neural=neural, origin=origin,
                         steps=steps, trees=trees, final=final, clocks=clocks, wind=wind, wind_arrays=wind_arrays, scientific=scientific))
    a, b = runs
    if ms == 100:
        need(a['result']['wall_total_s'] + b['result']['wall_total_s'] <= plan['short_pair']['wall_s_total_max'],
             'Short pair process time exceeds total PLAN budget')
    need(a['result']['parameters'] == b['result']['parameters'], 'Numerical/biological parameters differ')
    need(a['initial'] == b['initial'], 'Initial proof coverage differs across pair')
    need(a['origin'] == b['origin'] and a['steps'] == b['steps'], 'Pair initial clocks/sampling differ')
    fields = {key: array_error(a['traces'][key], b['traces'][key]) for key in TRACE_SHAPES}
    need(fields['DN_baseline']['exact'], 'Pair DN baseline differs')
    need(np.array_equal(a['traces']['DN_q_usada'][0], b['traces']['DN_q_usada'][0]), 'Initial DN command differs')
    initial_neural = {key: array_error(a['neural'][key][0], b['neural'][key][0]) for key in NEURAL_FIELDS}
    need(all(value['exact'] for value in initial_neural.values()), 'Initial neural/body state differs across pair')
    for name in ('pn_state', 'published'):
        exact_tree(a['trees'][0][name], b['trees'][0][name], 'Initial saved tree differs: ' + name)
    gates = {'release_bound': all(fields[key]['max_abs'] <= limits['release_max_abs']
                                  for key in ('ORN_q_L', 'ORN_q_R', 'DN_q_actual', 'PN_general_transmission')),
             'position_bound': fields['position_mm']['max_abs'] <= limits['position_max_abs_mm'],
             'yaw_bound': fields['yaw_delta_deg']['max_abs'] <= limits['yaw_max_abs_deg'],
             'same_applied_yaw_each_tick': fields['motor_filter_applied_rad_s']['exact'] and fields['command_yaw_rate_rad_s']['exact'],
             'same_contacts_each_tick': fields['contact_active']['exact'],
             'forward_command_bound': fields['command_forward_mm_s']['max_abs'] <= limits['forward_command_max_abs_mm_s']}
    wind_pair = {'stable': a['wind'], 'reviewed': b['wind']}
    if ms == 2000:
        wa, wb = a['wind_arrays'], b['wind_arrays']
        equal_pose = np.all(wa['qpos'] == wb['qpos'], axis=1)
        conditional = array_error(wa['generalized'][equal_pose], wb['generalized'][equal_pose])
        wind_pair.update(equal_pose_substeps=int(equal_pose.sum()), different_pose_substeps=int((~equal_pose).sum()),
                         generalized_where_qpos_equal=conditional,
                         generalized_all_report_only=array_error(wa['generalized'], wb['generalized']),
                         qpos_difference=array_error(wa['qpos'], wb['qpos']))
        gates['same_wind_mapping_where_qpos_equal'] = conditional['exact']
    timing = {'stable_advance_s': a['result']['advance_total_s'], 'reviewed_advance_s': b['result']['advance_total_s'],
              'stable_process_s': a['result']['wall_total_s'], 'reviewed_process_s': b['result']['wall_total_s'],
              'advance_speedup': a['result']['advance_total_s']/b['result']['advance_total_s'],
              'process_speedup': a['result']['wall_total_s']/b['result']['wall_total_s'],
              'advance_fraction_saved': 1-b['result']['advance_total_s']/a['result']['advance_total_s'],
              'process_fraction_saved': 1-b['result']['wall_total_s']/a['result']['wall_total_s'],
              'scope': 'Paired wall measurements; speedup is engineering evidence, not scientific or biological admission.'}
    performance = {}
    if ms == 100:
        performance['minimum_advance_fraction_saved'] = timing['advance_fraction_saved'] >= plan['short_pair']['minimum_advance_fraction_saved']
        if plan['short_pair'].get('process_no_regression'):
            performance['process_no_regression'] = timing['process_fraction_saved'] >= 0.
    events = event_difference(stable/'EVENTS.json', reviewed/'EVENTS.json', ms)
    neural = {str(step): {key: array_error(a['neural'][key][index], b['neural'][key][index]) for key in NEURAL_FIELDS}
              for index, step in enumerate(a['steps'])}
    trees = {name: {str(step): tree_difference(a['trees'][step][name], b['trees'][step][name])
                    for step in a['trees']} for name in ('pn_state', 'published')}
    return {'status': 'PASS' if all(gates.values()) and all(performance.values()) else 'FAIL',
            'ms': ms, 'functional_status': 'PASS' if all(gates.values()) else 'FAIL',
            'gates': gates, 'performance_gates': performance, 'limits': limits,
            'fields': fields, 'forward_command_exact': fields['command_forward_mm_s']['exact'],
            'neural': neural, 'saved_trees': trees,
            'final_trees': {name: tree_difference(a['final'][name], b['final'][name]) for name in a['final']},
            'local_state_clocks': {'stable': a['clocks'], 'reviewed': b['clocks']},
            'events': events, 'wind': wind_pair, 'timing': timing,
            'scientific_snapshots': {'stable': a['scientific'], 'reviewed': b['scientific']},
            'initial_evidence': {'direct_arrays': 588, 'initial_neural': initial_neural,
                                 'limitation': 'Direct live/source tree evidence validated; independently saved initial neural, PN and publication values compared. Full live initial session was not serialized by the runner.'},
            'provenance': {'plan_sha256': plan_digest, 'sources_sha256': lock_digest,
                           'verifier_sha256': sha(__file__), 'frozen_files_verified': len(lock)},
            'artifacts': {str(folder/name): sha(folder/name) for folder in (stable, reviewed)
                          for name in ('RESULT.json', 'INITIAL.json', 'traces.npz', 'neural_states.npz', 'EVENTS.json', 'MOTOR.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable', '--control', dest='stable', type=Path, required=True)
    parser.add_argument('--reviewed', '--candidate', dest='reviewed', type=Path, required=True)
    parser.add_argument('--ms', type=int, choices=(100, 2000), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'Verification output must be new')
    try:
        report = compare(args.stable, args.reviewed, args.ms)
    except (ValueError, OSError, KeyError, TypeError, StopIteration, zipfile.BadZipFile) as exc:
        report = {'status': 'INVALID', 'ms': args.ms,
                  'error': {'type': type(exc).__name__, 'message': str(exc)}, 'verifier_sha256': sha(__file__)}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'ms', 'gates', 'performance_gates', 'timing', 'error') if key in report}, indent=2))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
