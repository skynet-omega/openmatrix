"""Stamp explicit pair conductances at PN contacts, conserving local current.

An EM count or signed CNS weight is never converted to nS here. The caller
supplies each pair's total g and receptor reversal, and a declared allocation
over its contacts. Co-located receptors retain sum(g) and sum(g*E), including
shunting conductance when inward and outward currents cancel.

This is a stateless electrical interface. Receptor kinetics, source feedback
and their restart state belong to the caller, not to this anatomical map.
"""
import hashlib
import json
import numpy as np


def _frozen(value, dtype):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _stage(spec):
    if not isinstance(spec, dict) or set(spec) != {'nodes', 'conductance_nS', 'reversal_mV'}:
        raise ValueError('Explicit nodes, conductance_nS and reversal_mV required')
    nodes = np.asarray(spec['nodes'])
    g = np.asarray(spec['conductance_nS'])
    E = np.asarray(spec['reversal_mV'])
    if (nodes.ndim != 1 or nodes.dtype.kind not in 'iu' or not len(nodes)
            or np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes)
            or g.shape != nodes.shape or g.dtype.kind not in 'fiu'
            or not np.isfinite(g).all() or np.any(g < 0)
            or E.dtype.kind not in 'fiu' or E.shape not in [(), nodes.shape]
            or not np.isfinite(E).all()):
        raise ValueError('Unique nonnegative nodes, nonnegative finite g and finite E required')
    return nodes, g.astype(float, copy=False), np.broadcast_to(E, g.shape).astype(float, copy=False)


def combine_conductance_stages(specs):
    """Exact affine-current combination; never subtract inhibitory from excitatory g.

    The electrical solver still checks physical membrane support and size.
    Zero-g nodes are retained to keep anatomical support stable across time;
    their effective E is defined as zero because their current is zero.
    """
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError('At least one explicit conductance stage required')
    rows = [_stage(spec) for spec in specs]
    nodes = np.unique(np.concatenate([row[0] for row in rows]))
    total = np.zeros(len(nodes)); moment = np.zeros(len(nodes))
    with np.errstate(over='ignore', invalid='ignore'):
        for raw, g, E in rows:
            slots = np.searchsorted(nodes, raw)
            total[slots] += g
            moment[slots] += g * E
        reversal = np.divide(moment, total, out=np.zeros_like(total), where=total > 0)
    if not np.isfinite(total).all() or not np.isfinite(moment).all() or not np.isfinite(reversal).all():
        raise ValueError('Combined conductance or reversal moment overflowed')
    return dict(nodes=nodes, conductance_nS=total, reversal_mV=reversal)


class LocalPairConductance:
    """Complete named routes with explicit contact fractions, not fitted weights."""
    def __init__(self, source_ids, contact_pre_ids, contact_nodes, contact_fractions,
                 *, full_size, provenance):
        ids = np.asarray(source_ids); pre = np.asarray(contact_pre_ids)
        nodes = np.asarray(contact_nodes); fractions = np.asarray(contact_fractions)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                or np.any(ids <= 0) or np.any(ids[1:] <= ids[:-1])
                or pre.ndim != 1 or pre.dtype.kind not in 'iu' or not len(pre)
                or not np.array_equal(np.unique(pre), ids)
                or nodes.shape != pre.shape or nodes.dtype.kind not in 'iu'
                or type(full_size) is not int or full_size <= 0
                or np.any(nodes < 0) or np.any(nodes >= full_size)
                or fractions.shape != pre.shape or fractions.dtype.kind not in 'fiu'
                or not np.isfinite(fractions).all() or np.any(fractions < 0)
                or not isinstance(provenance, str) or not provenance.strip()):
            raise ValueError('Complete anatomy, explicit allocation fractions and provenance required')
        slots = np.searchsorted(ids, pre)
        totals = np.bincount(slots, weights=fractions, minlength=len(ids))
        if not np.allclose(totals, 1., rtol=0, atol=1e-12):
            raise ValueError('Contact fractions must sum to one separately for every pair')
        unique, inverse = np.unique(nodes, return_inverse=True)
        self.source_ids = _frozen(ids, np.int64)
        self.contact_pre_ids = _frozen(pre, np.int64)
        self.contact_nodes = _frozen(nodes, np.int64)
        self.contact_fractions = _frozen(fractions, np.float64)
        self.source_slot = _frozen(slots, np.int64)
        self.nodes = _frozen(unique, np.int64)
        self.node_slot = _frozen(inverse, np.int64)
        self.full_size = full_size
        self.provenance = provenance
        h = hashlib.sha256(json.dumps(dict(schema='PN_local_pair_conductance_v1',
            full_size=full_size, provenance=provenance), sort_keys=True).encode())
        for a in (self.source_ids, self.contact_pre_ids, self.contact_nodes, self.contact_fractions):
            h.update(a.shape[0].to_bytes(8, 'little')); h.update(a.tobytes())
        self.identity = h.hexdigest()

    def sample(self, pair_conductance_nS, pair_reversal_mV):
        """Distribute each explicitly supplied pair total exactly once."""
        g = np.asarray(pair_conductance_nS); E = np.asarray(pair_reversal_mV)
        if (g.shape != self.source_ids.shape or g.dtype.kind not in 'fiu'
                or not np.isfinite(g).all() or np.any(g < 0)
                or E.shape not in [(), self.source_ids.shape] or E.dtype.kind not in 'fiu'
                or not np.isfinite(E).all()):
            raise ValueError('Explicit nonnegative pair g in nS and finite reversal in mV required')
        # Aggregate two moments; averaging E by contact counts would be wrong.
        with np.errstate(over='ignore', invalid='ignore'):
            local_g = g.astype(float)[self.source_slot] * self.contact_fractions
            local_E = np.broadcast_to(E, g.shape).astype(float)[self.source_slot]
            total = np.bincount(self.node_slot, weights=local_g, minlength=len(self.nodes))
            moment = np.bincount(self.node_slot, weights=local_g * local_E, minlength=len(self.nodes))
            reversal = np.divide(moment, total, out=np.zeros_like(total), where=total > 0)
        if not np.isfinite(total).all() or not np.isfinite(moment).all() or not np.isfinite(reversal).all():
            raise ValueError('Local conductance or reversal moment overflowed')
        return dict(nodes=self.nodes, conductance_nS=total, reversal_mV=reversal)
