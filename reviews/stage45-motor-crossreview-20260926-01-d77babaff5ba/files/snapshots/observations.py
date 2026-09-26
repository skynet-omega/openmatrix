"""Read committed boundaries and publish each evidence block once.

This module never advances the organism or consumes a sensor/RNG value.
Atomic directory rename identifies a closed block; .partial is not evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

MOTOR_FIELDS = ('alpha', 'filtered', 'raw', 'applied', 'forward', 'trial_step',
                'body_calls', 'substeps_per_ms', 'wind_substeps', 'wind_active',
                'wind_generalized_peak_native')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def digest(value):
    """Typed content hash, independent of dict insertion order."""
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, np.ndarray):
            need(not v.dtype.hasobject, 'Object state is not auditable')
            a = np.ascontiguousarray(v)
            if a.dtype.kind in 'fc':
                need(np.isfinite(a).all(), 'Nonfinite array in witnessed state')
            h.update(b'array' + a.dtype.str.encode() + str(a.shape).encode())
            h.update(a.tobytes())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b'{')
            for k in sorted(v):
                add(k)
                add(v[k])
            h.update(b'}')
        elif isinstance(v, (list, tuple)):
            h.update(b'[')
            for item in v:
                add(item)
            h.update(b']')
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode() + b';')
    add(value)
    return h.hexdigest()


def neural(obj, cp):
    # Reuse the reviewed owner mapping; the CNS vector mixes representations.
    from run_trial import collect
    state = collect(obj, cp)
    brain = obj.core.brain
    plastic = obj.core.plasticity
    need(plastic.enabled is False, 'Plasticity was enabled during baseline')
    state.update(published_output=brain.rates.copy(),
                 published_time_ns=np.asarray(brain.time_ns, dtype=np.int64),
                 plasticity_eligibility=plastic.eligibility.copy(),
                 plasticity_factors=plastic.factors.copy(),
                 plasticity_time_ns=np.asarray(plastic.time_ns, dtype=np.int64),
                 plasticity_update_count=np.asarray(plastic.update_count, dtype=np.int64))
    for key, v in state.items():
        if v.dtype.kind in 'fc':
            need(np.isfinite(v).all(), 'Nonfinite observation: ' + key)
    return state


def witness(obj, motor, session, cp):
    from operator_state import OperatorState, LEGACY_BINDINGS
    # Hash each copy separately to bound observer memory. No live state is edited.
    return dict(session=digest(obj.core.state_dict()), prosthesis=digest(obj.state()),
        observed=digest(neural(obj, cp)), rng=digest(obj.core.brain.rng.bit_generator.state),
        stored_weights=digest(obj.core.brain.W.data),
        effective_operator=digest(OperatorState(obj.core.hybrid, LEGACY_BINDINGS).state_dict()),
        motor=digest({k: getattr(motor, k) for k in MOTOR_FIELDS}),
        event_records=len(session.events.audit))


def finish_directory(temporary, final, metadata):
    need(not final.exists(), 'Refusing to overwrite evidence: ' + str(final))
    files = sorted(p for p in temporary.rglob('*') if p.is_file())
    metadata['files'] = {str(p.relative_to(temporary)): sha(p) for p in files}
    save(temporary/'MANIFEST.json', metadata)
    # Persist file contents before publishing the directory. This does not claim
    # protection against failure of the host/storage device itself.
    for p in files:
        with p.open('rb') as f:
            os.fsync(f.fileno())
    temporary.rename(final)
    fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def primary(folder, ms, obj, motor, auditor, session, cp):
    from full_snapshot import snapshot
    from session_io import write_state
    folder = Path(folder)
    temporary = folder.with_name(folder.name + '.partial')
    temporary.mkdir(parents=True, exist_ok=False)
    cp.cuda.runtime.deviceSynchronize()
    before = witness(obj, motor, session, cp)
    snapshot(temporary/'scientific', obj, motor, auditor)
    np.save(temporary/'stored_weights.npy', obj.core.brain.W.data, allow_pickle=False)
    write_state(temporary/'plasticity', obj.core.plasticity.state_dict())
    np.savez_compressed(temporary/'neural.npz', **neural(obj, cp))
    after = witness(obj, motor, session, cp)
    need(before == after, 'Snapshot changed an owner: ' + str([k for k in before if before[k] != after[k]]))
    finish_directory(temporary, folder, dict(schema='motor13_primary_v1', elapsed_ms=ms,
        time_ns=int(obj.core.hybrid.time_ns), read_only_witness=before,
        restart_tested=False, plasticity_enabled=False))


def block(folder, first, last, rows, state, events, event_start, event_end, intervals, wind_rows):
    folder = Path(folder)
    temporary = folder.with_name(folder.name + '.partial')
    temporary.mkdir(parents=True, exist_ok=False)
    need(len(rows) == last-first+1 and all(int(r['paso']) == first+i for i, r in enumerate(rows)),
         'Observation block has a gap or duplicate')
    need(len(intervals) == last-first+1, 'Incomplete input intervals')
    need(event_start == 16*(first-1) and event_end == 16*last, 'Event coverage differs from block')
    need(int(state['time_ns']) == int(rows[-1]['CNS_time_ns']), 'Neural/trace endpoints differ')
    np.savez_compressed(temporary/'traces.npz', **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
    np.savez_compressed(temporary/'neural.npz', **state)
    save(temporary/'events.json', events)
    save(temporary/'inputs.json', intervals)
    if wind_rows:
        need(all(first <= r['trial_step'] <= last for r in wind_rows), 'Wind record outside block')
        np.savez_compressed(temporary/'wind_substeps.npz', **{
            k: np.asarray([r[k] for r in wind_rows]) for k in wind_rows[0]})
    need(len(events) == event_end-event_start, 'Event cursor mismatch')
    finish_directory(temporary, folder, dict(schema='motor13_block_v1', first_ms=first,
        last_ms=last, event_start=event_start, event_end=event_end,
        time_ns=int(state['time_ns']), wind_substeps=len(wind_rows), phase='committed_boundary'))


def verify_directory(folder):
    folder = Path(folder)
    need(not folder.name.endswith('.partial'), 'Uncommitted directory')
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    need(actual == set(manifest['files']) | {'MANIFEST.json'}, 'Manifest file set mismatch')
    for name, checksum in manifest['files'].items():
        need(sha(folder/name) == checksum, 'Evidence changed: ' + str(folder/name))
    return manifest
