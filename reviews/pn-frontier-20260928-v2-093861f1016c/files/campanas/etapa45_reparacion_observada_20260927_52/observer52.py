"""Read-only committed-output panel and exact scientific-owner witnesses."""
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Panel:
    def __init__(self, node_ids, enabled, orn_ids):
        with np.load(H / 'PANEL.npz', allow_pickle=False) as z:
            self.ids = z['ids'].copy()
            self.rows = z['canonical_rows'].copy()
        need(np.array_equal(node_ids[self.rows], self.ids), 'Panel identity')
        need(len(self.ids) == 2757, 'Panel population size')
        self.enabled = enabled
        self.q, self.time_ns = [], []
        self.orn_ids = np.array(orn_ids, np.int64)
        self.orn_rows = np.searchsorted(node_ids, self.orn_ids)
        need(np.array_equal(node_ids[self.orn_rows], self.orn_ids), 'ORN panel identity')
        self.orn_q = []

    def record(self, committed_release, time_ns):
        # The caller already requested this release for the historical16.
        # No extra release, coefficient, RHS, state mutation or GPU operation.
        if self.enabled:
            values = committed_release[self.rows].copy()
            need(np.isfinite(values).all(), 'Nonfinite panel output')
            self.q.append(values)
            orn = committed_release[self.orn_rows].copy()
            need(np.isfinite(orn).all(), 'Nonfinite ORN output')
            self.orn_q.append(orn)
            self.time_ns.append(int(time_ns))

    def flush(self, path):
        if self.enabled:
            np.savez_compressed(path, ids=self.ids, rows=self.rows,
                                time_ns=np.array(self.time_ns, np.int64),
                                q=np.asarray(self.q), ORN_ids=self.orn_ids,
                                ORN_rows=self.orn_rows, ORN_q=np.asarray(self.orn_q))


def witness(run, air):
    """Hash separately to avoid retaining multiple full state copies.

    Uses the same scientific owners as the qualified48 restoration, plus
    the new air and candidate owners. Wall time/performance counters are
    not scientific state. Candidate raw-range infinity sentinels are
    represented by masks and finite values, never silently dropped.
    """
    import cupy as cp
    from observations import digest
    from operator_state import OperatorState, LEGACY_BINDINGS
    from resume49 import interval_state, motor_state, boundary_state
    cp.cuda.runtime.deviceSynchronize()
    obj = run.obj
    out = dict(session=digest(obj.core.state_dict()),
               prosthesis=digest(obj.state()),
               published=digest(dict(rates=obj.core.brain.rates,
                   time_ns=obj.core.brain.time_ns,
                   rng=obj.core.brain.rng.bit_generator.state)),
               effective_operator=digest(OperatorState(obj.core.hybrid, LEGACY_BINDINGS).state_dict()),
               stored_operator=digest(dict(weights=obj.core.brain.W.data)),
               input_owner=digest(run.stimulus.state()),
               interval_owner=digest(interval_state(run.auditor)),
               motor_owner=digest(motor_state(run.motor)),
               boundary=digest(boundary_state(run)),
               event_audit=digest(run.session.events.audit),
               air_owner=digest(dict(field_world=air.field,
                   origin_rotation=air.origin_rotation, origin=air.origin,
                   prefix=air.prefix, setting=air.setting, device=air.device.get(),
                   last=air.last, seen=air.seen.get(), calls=air.calls.get())))
    c = run.observer.candidate
    if c is not None:
        ranges = {}
        for name, source in [('low', c.low), ('high', c.high)]:
            v = source.get()
            need(not np.isnan(v).any(), 'NaN candidate range')
            ranges[name] = dict(finite=np.where(np.isfinite(v), v, 0.),
                               positive_inf=np.isposinf(v), negative_inf=np.isneginf(v))
        out['candidate_owner'] = digest(dict(q0=c.q0.get(), S=c.scales.get(),
                                            gE0=c.ge0.get(), ranges=ranges))
    return out
