"""Check committed state at the restart boundary after executor installation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import traceback

from check_restart import EPOCH, HERE, OLD, STAGE2, digest, equal_state, save_json


def main():
    from motor_runtime import load as _historical_path
    from static_lateral_checkpoint import load
    import cupy as cp

    source = HERE / 'run_10_versioned_driver/checkpoint_v1'
    out = HERE / 'after_install_02'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    obj = session = undo = None
    result = {'status': 'STARTED', 'checkpoint': str(source)}
    try:
        obj = load(source)
        brain = obj.core.hybrid

        def snapshot():
            cp.cuda.runtime.deviceSynchronize()
            return copy.deepcopy({
                'core': obj.core.state_dict(),
                'prosthesis': obj.state(),
                'published': {'rates': obj.core.brain.rates,
                              'time_ns': obj.core.brain.time_ns,
                              'rng': obj.core.brain.rng.bit_generator.state},
                'driver': obj.core.world.boundary.metadata(),
            })

        before = snapshot()
        control = snapshot()
        result['reader_difference'] = equal_state(before, control, 'reader')
        del control
        result['prestep'] = {
            'cns_sha256': digest(brain.state),
            'pending_sensors_sha256': digest(obj.core.pending_sensors),
            'next_step_ns': int(brain.next_step_ns),
        }
        sys.path.insert(0, str(EPOCH))
        import event_waveform, event_ports, event_coupling, native_cell, device_cell
        for module in (event_waveform, event_ports, event_coupling, native_cell, device_cell):
            if Path(module.__file__).resolve().parent != EPOCH:
                raise RuntimeError('Wrong event owner: ' + module.__name__)
        sys.path.insert(0, str(STAGE2))
        from real_model import install
        from runtime_session import RuntimeSession
        undo = install()
        after_model = snapshot()
        result['real_model_install_difference'] = equal_state(
            before, after_model, 'real_model_install')
        del before
        session = RuntimeSession(brain, 'causal_cuda')
        after_session = snapshot()
        result['runtime_session_install_difference'] = equal_state(
            after_model, after_session, 'runtime_session_install')
        del after_model, after_session
        result['after'] = {
            'cns_sha256': digest(brain.state),
            'pending_sensors_sha256': digest(obj.core.pending_sensors),
            'next_step_ns': int(brain.next_step_ns),
        }
        result['status'] = ('EXACT' if all(result[key] is None for key in
                            ('reader_difference', 'real_model_install_difference',
                             'runtime_session_install_difference'))
                            else 'INSTALL_CHANGED_STATE')
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.close()
        if undo is not None:
            undo()
        if obj is not None:
            obj.close()
        save_json(out / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': str(out / 'RESULT.json')}))
    return 0 if result['status'] == 'EXACT' else 2


if __name__ == '__main__':
    raise SystemExit(main())
