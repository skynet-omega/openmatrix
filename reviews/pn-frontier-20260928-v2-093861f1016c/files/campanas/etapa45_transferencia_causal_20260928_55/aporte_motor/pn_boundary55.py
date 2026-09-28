"""Narrow ALPN q/common-filter intervention proposed for campaign 55.

No imports from the organism and no GPU work on import. The live entry point
requires an existing runtime and has NOT yet passed a live CNS no-op. This is
not a transplant of fine PN10208, PN->KC receptors, or the entire PN system.
"""
import hashlib
import numpy as np

PN_IDS_SHA256 = 'df813ecbdf482fc234a18577c748ce95b037cfee8e5fda8834c0d02a5af4b4c1'
SCHEMA = 'campaign55_canonical_ALPN_q_common_filter_only_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def selection(node_ids, pn_ids, transmission_start, state_size):
    """Reuse the exact 686 anatomical identities selected in campaign 51."""
    nodes, ids = np.asarray(node_ids), np.asarray(pn_ids)
    need(nodes.ndim == ids.ndim == 1 and nodes.dtype.kind in 'iu'
         and ids.dtype.kind in 'iu', 'Integer anatomical identities required')
    need(len(nodes) == 166700 and np.all(nodes[1:] > nodes[:-1]), 'Canonical graph identity/order')
    need(len(ids) == 686 and ids_hash(ids) == PN_IDS_SHA256, 'Not the frozen ALPN population from 51')
    rows = np.searchsorted(nodes, ids)
    need(np.all(rows < len(nodes)) and np.array_equal(nodes[rows], ids), 'ALPN missing from recipient')
    need(type(transmission_start) is int and transmission_start == 177758,
         'Common-filter layout differs from 48/51/54')
    selected = np.r_[rows, transmission_start + rows]
    need(np.max(selected) < state_size and len(np.unique(selected)) == 1372, 'Invalid selected slots')
    return rows, selected


def prepare_payload(node_ids, pn_ids, transmission_start, state, time_ns):
    """Read a decoded donor at its actual committed clock; never shift time."""
    state = np.asarray(state)
    need(state.ndim == 1 and state.dtype == np.float64, 'Expected FP64 canonical host state')
    _, selected = selection(node_ids, pn_ids, transmission_start, len(state))
    values = state[selected].copy()
    need(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), 'Invalid donor q/filter values')
    need(type(time_ns) is int and time_ns >= 0, 'Explicit donor clock required')
    return dict(schema=SCHEMA, time_ns=time_ns, pn_ids=np.asarray(pn_ids).copy(),
                node_ids_sha256=ids_hash(node_ids), state_size=len(state),
                selected_indices=selected, values=values)


def checked_values(payload, node_ids, transmission_start, state_size, time_ns):
    need(payload.get('schema') == SCHEMA, 'Wrong intervention scope')
    need(type(payload.get('time_ns')) is int and payload['time_ns'] == time_ns, 'Donor/recipient clock mismatch')
    need(payload['state_size'] == state_size and payload['node_ids_sha256'] == ids_hash(node_ids),
         'Donor/recipient graph or state layout mismatch')
    rows, selected = selection(node_ids, payload['pn_ids'], transmission_start, state_size)
    need(np.array_equal(selected, payload['selected_indices']), 'Altered patch support')
    values = np.asarray(payload['values'])
    need(values.dtype == np.float64 and values.shape == (1372,) and np.isfinite(values).all()
         and np.all((values >= 0) & (values <= 1)), 'Invalid patch values')
    return rows, selected, values.copy()


def apply_between_committed_ms(run, payload):
    """PROPOSED live API: caller must own the stopped outer step loop.

    Apply before the next stimulus.consume()/obj.step(), after a whole ms.
    The runner must compare an actual continuation against untouched control
    before authorizing crossed donors. No DN, fine-PN, receptor, clock,
    event history, body or operator state is written by this function.
    """
    h, session = run.obj.core.hybrid, run.session
    need(session.brain is h and not session.closed, 'Missing matching live runtime')
    need(session.events.active is None, 'Cannot patch an active event interval')
    need(not getattr(h, '_edge_buffer_active', False)
         and not getattr(h, '_general_buffer_active', False), 'Active temporary operator writer')
    need(not getattr(h, '_native_rebuild_required', False)
         and not getattr(h, '_operator_restore_invalid', False)
         and not run.obj.core.failed, 'Invalidated runtime')
    need(h.time_ns == run.obj.core.time_ns == h.brain.time_ns == run.obj.core.world.time_ns,
         'Not a committed outer boundary')
    need(isinstance(h.state, np.ndarray) and h.state.dtype == np.float64, 'Unexpected host owner')
    rows, selected, values = checked_values(payload, h.brain.node_ids,
        int(h.transmission_start), len(h.state), int(h.time_ns))
    need(not np.isin(h.brain.node_ids[rows], run.obj.dn_ids).any(), 'Motor reader selected')
    run.stimulus.cp.cuda.runtime.deviceSynchronize()
    old_state, old_rates = h.state.copy(), h.brain.rates.copy()
    outside = np.ones(len(h.state), dtype=bool); outside[selected] = False
    other_rates = np.ones(h.brain.n_neurons, dtype=bool); other_rates[rows] = False
    try:
        h.state[selected] = values
        h.publish_rates()
        need(np.array_equal(h.state[selected], values), 'Patch not installed')
        need(np.array_equal(h.state[outside], old_state[outside]), 'Nonselected neural state changed')
        need(np.array_equal(h.brain.rates[other_rates], old_rates[other_rates]), 'Nonselected publication changed')
        run.stimulus.cp.cuda.runtime.deviceSynchronize()
    except BaseException:
        h.state[:] = old_state
        h.brain.rates[:] = old_rates
        run.obj.core.failed = True
        raise
    return dict(schema=SCHEMA, time_ns=int(h.time_ns), canonical_ALPN_count=len(rows),
                selected_slots=len(selected), changed_slots=int(np.count_nonzero(old_state[selected] != values)),
                nonselected_state_exact=True, nonselected_published_rates_exact=True,
                live_continuation_qualified=False)
