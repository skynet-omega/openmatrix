"""One real-organism checkpoint/restart discriminator for the neural candidate.

Prepare and load run in separate Python processes.  This does not modify the
historical organism or the stable motor, and does not call a synthetic model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EPOCH = ROOT / 'motor_nuevo/epoch_cost_20260923'
STAGE2 = ROOT / 'motor_nuevo/neurocore_blocks_20260925_03'
sys.path[:0] = [str(OLD / 'work/motor14_20260922'),
                str(OLD / 'work/motor13_20260922'),
                str(ROOT / 'motor_nuevo/pipeline_review_20260922')]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def equal_state(a, b, path='state'):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return path + '.keys'
        for key in sorted(a):
            failure = equal_state(a[key], b[key], path + '.' + str(key))
            if failure:
                return failure
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path + '.length'
        for i, (x, y) in enumerate(zip(a, b)):
            failure = equal_state(x, y, path + '[' + str(i) + ']')
            if failure:
                return failure
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.dtype != y.dtype or x.shape != y.shape:
            return path + '.layout'
        allow_nan = x.dtype.kind in 'fc'
        if not np.array_equal(x, y, equal_nan=allow_nan):
            mismatch = np.argwhere(~(x == y))
            at = tuple(int(i) for i in mismatch[0]) if len(mismatch) else ()
            return (path + '[' + ','.join(map(str, at)) + ']=' +
                    repr(x[at].item()) + ' vs ' + repr(y[at].item()))
        return None
    return None if type(a) is type(b) and a == b else path


def physical_events(audit):
    # Session-local block ordinal is diagnostic, not part of the organism.
    return [{k: v for k, v in entry.items() if k != 'block'} for entry in audit]


def restore_field(world, metadata):
    from static_field import StaticLateralField
    field = StaticLateralField.__new__(StaticLateralField)
    field.base, field.world, field.arm = world.boundary, world, metadata['arm']
    field.center = np.asarray(metadata['center_mm'], dtype=float)
    field.axis = np.asarray(metadata['odor_axis'], dtype=float)
    field.origin_ns = int(metadata['installed_ns'])
    field.onset_ms = float(metadata['first_ON_after_install_ms'])
    if field.metadata() != metadata:
        raise ValueError('External driver metadata changed during restoration')
    world.boundary = field


def prepare(out, reference_continuous=None):
    import cupy as cp
    from motor_runtime import load
    from runtime_session import RuntimeSession
    from session_io import write_state

    obj = session = undo = None
    result = {'phase': 'control' if reference_continuous else 'prepare',
              'status': 'STARTED', 'field': 'odor_left',
              'candidate': str(STAGE2), 'completed_ms': 0}
    try:
        obj, diagnostic, plan, Field, field, _ = load(out / 'preparation_inputs')
        brain = obj.core.hybrid
        if plan['checkpoint'] != 'work/stage234_settling_extension_20260915/settled_700ms':
            raise ValueError('Unexpected preparation')
        sys.path.insert(0, str(EPOCH))
        for name in ('event_waveform', 'event_ports', 'event_coupling',
                     'native_cell', 'device_cell'):
            if name in sys.modules:
                raise RuntimeError('Unexpected owner already imported: ' + name)
        import event_waveform, event_ports, event_coupling, native_cell, device_cell
        for module in (event_waveform, event_ports, event_coupling,
                       native_cell, device_cell):
            if Path(module.__file__).resolve().parent != EPOCH:
                raise RuntimeError('Wrong event owner: ' + module.__name__)
        sys.path.insert(0, str(STAGE2))
        from real_model import install
        undo = install()
        session = RuntimeSession(brain, 'causal_cuda')
        diagnostic.instalar_campo(obj, Field, field, 'odor_left', 0.)
        start_ns = int(brain.time_ns)
        t = time.perf_counter()
        obj.step()
        cp.cuda.runtime.deviceSynchronize()
        result['advance_wall_s'] = time.perf_counter() - t
        if int(brain.time_ns) != start_ns + 1_000_000:
            raise ValueError('Organism did not advance one millisecond')
        result['completed_ms'] = 1
        result['time_ns'] = int(brain.time_ns)
        result['pending_sensors_sha256'] = digest(obj.core.pending_sensors)
        result['cns_sha256'] = digest(brain.state)
        result['field_metadata'] = obj.core.world.boundary.metadata()
        result['boundary_class'] = type(obj.core.world.boundary).__name__
        if reference_continuous is None:
            t = time.perf_counter()
            obj.save(out / 'checkpoint')
            result['checkpoint_wall_s'] = time.perf_counter() - t
            result['checkpoint_bytes'] = sum(p.stat().st_size for p in
                                             (out / 'checkpoint').rglob('*') if p.is_file())
            if (digest(brain.state) != result['cns_sha256'] or
                    digest(obj.core.pending_sensors) != result['pending_sensors_sha256']):
                raise ValueError('Saving changed the running organism')
        prior_events = len(session.events.audit)
        obj.step()
        cp.cuda.runtime.deviceSynchronize()
        if int(brain.time_ns) != start_ns + 2_000_000:
            raise ValueError('Continuous branch did not advance a second millisecond')
        final = out / 'continuous_2ms'
        final.mkdir()
        write_state(final / 'core', obj.core.state_dict())
        write_state(final / 'prosthesis', obj.state())
        write_state(final / 'published', {'rates': obj.core.brain.rates,
                    'time_ns': obj.core.brain.time_ns,
                    'rng': obj.core.brain.rng.bit_generator.state})
        save_json(final / 'events.json', physical_events(session.events.audit[prior_events:]))
        result['continuation_events'] = len(session.events.audit) - prior_events
        if reference_continuous is not None:
            from session_io import read_state
            result['component_differences'] = {
                name: equal_state(read_state(final / name),
                                  read_state(reference_continuous / name), name)
                for name in ('core', 'prosthesis', 'published')}
            result['component_differences']['events'] = equal_state(
                json.loads((final / 'events.json').read_text()),
                json.loads((reference_continuous / 'events.json').read_text()),
                'events')
            result['status'] = ('CONTROL_EXACT' if all(v is None for v in
                                 result['component_differences'].values())
                                else 'CONTROL_DIVERGED')
        else:
            result['status'] = 'CHECKPOINT_WRITTEN'
    except BaseException as exc:
        result.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc())
    finally:
        if session is not None:
            try:
                result['runtime'] = session.report()
                session.close()
            except BaseException as exc:
                result['session_close_error'] = repr(exc)
        if undo is not None:
            undo()
        if obj is not None:
            obj.close()
        save_json(out / 'PREPARE.json', result)
    return result['status'] in ('CHECKPOINT_WRITTEN', 'CONTROL_EXACT')


def reload(out, recover_driver):
    result = {'phase': 'reload', 'status': 'STARTED',
              'test_only_external_driver_recovery': recover_driver}
    obj = session = undo = None
    try:
        from motor_runtime import load  # Establishes the historical import path.
        sys.path.insert(0, str(OLD / 'work/stage4_antennal_contact_adapter_20260915'))
        from antennal_runtime import AntennalContactRuntime
        from antennal_world import AntennalWorld
        expected = json.loads((out / 'PREPARE.json').read_text())
        if expected['status'] != 'CHECKPOINT_WRITTEN':
            raise ValueError('No successful preparation to reload')
        if recover_driver:
            sys.path.insert(0, str(OLD / 'work/stage3_static_lateral_field_20260916'))
            old_bind = AntennalWorld.bind
            def bind_with_driver(world, body):
                old_bind(world, body)
                restore_field(world, expected['field_metadata'])
            AntennalWorld.bind = bind_with_driver
            try:
                obj = AntennalContactRuntime.load(out / 'checkpoint')
            finally:
                AntennalWorld.bind = old_bind
        else:
            obj = AntennalContactRuntime.load(out / 'checkpoint')
        result['time_ns'] = int(obj.core.hybrid.time_ns)
        result['pending_sensors_sha256'] = digest(obj.core.pending_sensors)
        result['cns_sha256'] = digest(obj.core.hybrid.state)
        result['boundary_class'] = type(obj.core.world.boundary).__name__
        result['clock_equal'] = result['time_ns'] == expected['time_ns']
        result['state_equal'] = result['cns_sha256'] == expected['cns_sha256']
        result['sensors_equal'] = (result['pending_sensors_sha256'] ==
                                   expected['pending_sensors_sha256'])
        result['driver_equal'] = result['boundary_class'] == expected['boundary_class']
        from session_io import read_state
        result['prestep_core_difference'] = equal_state(
            obj.core.state_dict(), read_state(out / 'checkpoint/core_carrier/session'),
            'prestep_core')
        result['prestep_prosthesis_difference'] = equal_state(
            obj.state(), read_state(out / 'checkpoint/prosthesis'),
            'prestep_prosthesis')
        result['driver_metadata_equal'] = (
            obj.core.world.boundary.metadata() == expected['field_metadata']
            if hasattr(obj.core.world.boundary, 'metadata') else False)
        if not all(result[k] for k in ('clock_equal', 'state_equal',
                                     'sensors_equal', 'driver_equal')):
            result['status'] = 'INCOMPLETE_RESTORE'
        elif recover_driver:
            import cupy as cp
            from runtime_session import RuntimeSession
            from session_io import read_state, write_state
            brain = obj.core.hybrid
            sys.path.insert(0, str(EPOCH))
            import event_waveform, event_ports, event_coupling, native_cell, device_cell
            for module in (event_waveform, event_ports, event_coupling,
                           native_cell, device_cell):
                if Path(module.__file__).resolve().parent != EPOCH:
                    raise RuntimeError('Wrong event owner: ' + module.__name__)
            sys.path.insert(0, str(STAGE2))
            from real_model import install
            undo = install()
            session = RuntimeSession(brain, 'causal_cuda')
            obj.step()
            cp.cuda.runtime.deviceSynchronize()
            final = out / 'continuous_2ms'
            state_checks = {
                'core': obj.core.state_dict(),
                'prosthesis': obj.state(),
                'published': {'rates': obj.core.brain.rates,
                              'time_ns': obj.core.brain.time_ns,
                              'rng': obj.core.brain.rng.bit_generator.state},
            }
            resumed = out / 'resumed_2ms'
            resumed.mkdir()
            for name, current in state_checks.items():
                write_state(resumed / name, current)
            save_json(resumed / 'events.json',
                      physical_events(session.events.audit))
            reference_core = read_state(final / 'core')
            result['core_field_differences'] = {
                key: equal_state(state_checks['core'][key], reference_core[key],
                                 'core.' + key)
                for key in sorted(state_checks['core'])}
            result['hybrid_field_differences'] = {
                key: equal_state(state_checks['core']['hybrid'][key],
                                 reference_core['hybrid'][key],
                                 'core.hybrid.' + key)
                for key in sorted(state_checks['core']['hybrid'])}
            result['component_differences'] = {}
            for name, current in state_checks.items():
                failure = equal_state(current, read_state(final / name), name)
                result['component_differences'][name] = failure
            reference_events = json.loads((final / 'events.json').read_text())
            result['component_differences']['events'] = equal_state(
                physical_events(session.events.audit), reference_events, 'events')
            result['status'] = ('CONTINUATION_EXACT' if all(v is None for v in
                                 result['component_differences'].values())
                                else 'CONTINUATION_DIVERGED')
        else:
            result['status'] = 'FULL_PRESTEP_RESTORE'
    except BaseException as exc:
        result.update(status='LOAD_FAILED', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.close()
        if undo is not None:
            undo()
        if obj is not None:
            obj.close()
        save_json(out / 'RELOAD.json', result)
    return result['status'] in ('FULL_PRESTEP_RESTORE', 'CONTINUATION_EXACT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'reload'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-limit', type=int, default=300)
    parser.add_argument('--test-recover-driver', action='store_true')
    parser.add_argument('--reference-continuous', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        args.out.mkdir(parents=True, exist_ok=False)
    elif not args.out.is_dir():
        raise FileNotFoundError(args.out)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Wall limit')))
    signal.alarm(args.wall_limit)
    try:
        passed = (prepare(args.out, args.reference_continuous) if args.phase == 'prepare' else
                  reload(args.out, args.test_recover_driver))
    finally:
        signal.alarm(0)
    print(json.dumps({'phase': args.phase, 'passed': passed, 'out': str(args.out)}), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
