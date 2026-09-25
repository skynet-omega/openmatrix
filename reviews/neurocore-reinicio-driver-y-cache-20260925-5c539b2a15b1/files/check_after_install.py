"""Check committed state at the restart boundary after executor installation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

from check_restart import EPOCH, HERE, OLD, STAGE2, digest, equal_state, save_json


def main():
    from motor_runtime import load as _historical_path
    from static_lateral_checkpoint import load
    from session_io import read_state
    import cupy as cp

    source = HERE / 'run_10_versioned_driver/checkpoint_v1'
    out = HERE / 'after_install_01'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    obj = session = undo = None
    result = {'status': 'STARTED', 'checkpoint': str(source)}
    try:
        obj = load(source)
        brain = obj.core.hybrid
        prior_core, prior_prosthesis = obj.core.state_dict(), obj.state()
        expected_core = read_state(source / 'organism/core_carrier/session')
        expected_prosthesis = read_state(source / 'organism/prosthesis')
        result['before'] = {
            'core_difference': equal_state(prior_core, expected_core, 'core'),
            'prosthesis_difference': equal_state(prior_prosthesis, expected_prosthesis, 'prosthesis'),
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
        session = RuntimeSession(brain, 'causal_cuda')
        cp.cuda.runtime.deviceSynchronize()
        current_core, current_prosthesis = obj.core.state_dict(), obj.state()
        result['after'] = {
            'core_difference_from_checkpoint': equal_state(current_core, expected_core, 'core'),
            'prosthesis_difference_from_checkpoint': equal_state(current_prosthesis, expected_prosthesis, 'prosthesis'),
            'core_difference_from_preinstall': equal_state(current_core, prior_core, 'core'),
            'prosthesis_difference_from_preinstall': equal_state(current_prosthesis, prior_prosthesis, 'prosthesis'),
            'cns_sha256': digest(brain.state),
            'pending_sensors_sha256': digest(obj.core.pending_sensors),
            'next_step_ns': int(brain.next_step_ns),
        }
        result['status'] = ('EXACT' if all(value is None for section in
                            (result['before'], result['after'])
                            for key, value in section.items() if 'difference' in key)
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
