"""Preserve scientific owners at a committed boundary, including motor memory.

The trial is continuous. These saved owners are evidence for future recovery;
no claim of tested generic restart is made by writing them.
"""
import hashlib
import json
from pathlib import Path


def snapshot(folder, obj, motor, auditor):
    from session_io import write_state
    from operator_state import OperatorState, LEGACY_BINDINGS
    h = obj.core.hybrid
    if (getattr(obj.core, 'failed', False) or getattr(h, '_native_rebuild_required', False)
            or getattr(h, '_operator_restore_invalid', False)):
        raise RuntimeError('Invalidated owner cannot publish a scientific snapshot')
    folder = Path(folder)
    temporary = folder.with_name(folder.name + '.partial')
    temporary.mkdir(exist_ok=False)
    write_state(temporary/'session', obj.core.state_dict())
    write_state(temporary/'prosthesis', obj.state())
    write_state(temporary/'published', dict(rates=obj.core.brain.rates,
        time_ns=obj.core.brain.time_ns, rng=obj.core.brain.rng.bit_generator.state))
    write_state(temporary/'effective_operator', OperatorState(h, LEGACY_BINDINGS).state_dict())
    fields = ('alpha','filtered','raw','applied','forward','trial_step','body_calls',
              'substeps_per_ms','wind_substeps','wind_active','wind_generalized_peak_native')
    write_state(temporary/'motor', {key:getattr(motor,key) for key in fields})
    (temporary/'boundary.json').write_text(json.dumps(obj.core.world.boundary.metadata(), indent=2)+'\n')
    auditor.export(temporary/'GAUSSIAN_INTERVALS.json')
    record = dict(schema='review12_scientific_boundary_v1', time_ns=int(h.time_ns),
                  phase='after_committed_organism_step_before_next_sensor_consumption',
                  restart_tested=False, files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(temporary.iterdir()) if p.is_file()})
    (temporary/'MANIFEST.json').write_text(json.dumps(record, indent=2)+'\n')
    if folder.exists():
        raise FileExistsError(folder)
    temporary.rename(folder)
