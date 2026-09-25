"""Trace the first differing physical LIF operator in continuous vs restart.

The wrapper only copies scalars at the named neuron. Its output is verified
against the previously saved, uninstrumented real-organism continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from check_restart import (EPOCH, HERE, OLD, ROOT, STAGE2, digest, equal_state,
                           physical_events, save_json)

ROW = 37405


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def run(mode: str, out: Path, reference: Path):
    import cupy as cp
    from motor_runtime import load as load_prepared
    from runtime_session import RuntimeSession
    from session_io import read_state

    obj = session = undo = None
    source = None
    trace = []
    result = {'mode': mode, 'operator': 'event_coupling.lif_record',
              'target_row': ROW, 'status': 'STARTED'}
    try:
        if mode == 'continuous':
            obj, diagnostic, plan, Field, field, _ = load_prepared(out / 'preparation_inputs')
        else:
            from static_lateral_checkpoint import load as load_versioned
            obj = load_versioned(HERE / 'run_10_versioned_driver/checkpoint_v1')
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
        if mode == 'continuous':
            diagnostic.instalar_campo(obj, Field, field, 'odor_left', 0.)
            obj.step()
            cp.cuda.runtime.deviceSynchronize()
        prep = json.loads((HERE / 'run_10_versioned_driver/PREPARE.json').read_text())
        result['prestep'] = {
            'time_ns': int(brain.time_ns),
            'cns_sha256': digest(brain.state),
            'pending_sensors_sha256': digest(obj.core.pending_sensors),
            'next_step_ns': int(brain.next_step_ns),
            'body_integration_sha256': digest(obj.body.integration_state()),
        }
        if (result['prestep']['time_ns'] != prep['time_ns'] or
                result['prestep']['cns_sha256'] != prep['cns_sha256'] or
                result['prestep']['pending_sensors_sha256'] != prep['pending_sensors_sha256']):
            raise ValueError('Physical prestep does not match the saved checkpoint')
        engine = session.events
        target = np.flatnonzero(engine.rows[engine.other] == ROW)
        if len(target) != 1:
            raise ValueError('Target LIF row is not unique in the event owner')
        idx = int(target[0])
        source = event_coupling.lif_record

        def record(v, refractory, counts, q, vinf, rate, reset,
                   threshold, caps, tau, dt, *args):
            before = {'voltage': finite(v[idx]), 'refractory': finite(refractory[idx]),
                      'count': int(counts[idx]), 'q': finite(q[idx]),
                      'vinf': finite(vinf[idx]), 'rate': finite(rate[idx]),
                      'reset': finite(reset[idx]), 'threshold': finite(threshold[idx]),
                      'cap': finite(caps[idx]), 'filter_tau': finite(tau[idx])}
            outputs = source(v, refractory, counts, q, vinf, rate, reset,
                             threshold, caps, tau, dt, *args)
            clipped, times, jumps, posts = outputs
            trace.append({'call': len(trace), 'start_elapsed_ns': int(engine.start_elapsed),
                          'brain_time_ns': int(brain.time_ns), 'duration_s': finite(dt),
                          'before': before,
                          'after': {'voltage': finite(v[idx]),
                                    'refractory': finite(refractory[idx]),
                                    'count': int(counts[idx]), 'q': finite(q[idx])},
                          'event_times_s': [finite(x) for x in times[idx]],
                          'event_jumps': [finite(x) for x in jumps[idx]],
                          'event_posts': [finite(x) for x in posts[idx]]})
            return outputs

        event_coupling.lif_record = record
        try:
            obj.step()
            cp.cuda.runtime.deviceSynchronize()
        finally:
            event_coupling.lif_record = source
            source = None
        current = {
            'core': obj.core.state_dict(),
            'prosthesis': obj.state(),
            'published': {'rates': obj.core.brain.rates,
                          'time_ns': obj.core.brain.time_ns,
                          'rng': obj.core.brain.rng.bit_generator.state},
        }
        result['component_differences'] = {
            name: equal_state(value, read_state(reference / name), name)
            for name, value in current.items()}
        result['component_differences']['events'] = equal_state(
            physical_events(session.events.audit[-16:]),
            json.loads((reference / 'events.json').read_text()), 'events')
        result['calls'] = len(trace)
        result['trace'] = trace
        result['status'] = ('TRACE_EXACT' if all(v is None for v in
                            result['component_differences'].values())
                            else 'TRACE_CHANGED_EXECUTION')
    except BaseException as exc:
        import traceback
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if source is not None:
            event_coupling.lif_record = source
        if session is not None:
            session.close()
        if undo is not None:
            undo()
        if obj is not None:
            obj.close()
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'TRACE.json', result)
    return result['status'] == 'TRACE_EXACT'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('continuous', 'restart'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference = HERE / 'run_10_versioned_driver' / (
        'continuous_2ms' if args.mode == 'continuous' else 'resumed_2ms')
    passed = run(args.mode, args.out, reference)
    print(json.dumps({'mode': args.mode, 'passed': passed, 'out': str(args.out)}))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
