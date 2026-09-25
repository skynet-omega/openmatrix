"""Conserved spatial membrane model bound to the reusable block executor.

No historical numerical modules are imported. The organism object is only an
input/output adapter; equations and physical event rules live in the model CU.
"""
from pathlib import Path
import hashlib
import threading
import time
import types
import numpy as np
import cupy as cp
from block_executor import BlockExecutor, need

HERE = Path(__file__).resolve().parent
CELL_FIELDS = ('delta', 'gates', 'q', 'counts', 'last_siz', 'previous_slope', 'trough', 'clipped')


class DeviceCell:
    def __init__(self, batch, events):
        need(batch.ports == 17 and batch.backend == 'cuda', 'declared model requires 17 CUDA coordinates')
        self.b, self.events, self.width = batch, events, 8
        self._gate = threading.Lock()
        self.publication_failed = False
        publisher = batch._motor_axonal_callback
        # Compatibility assertion belongs to this bridge, not to the core.
        from axon_gpu import Publisher
        need(type(publisher) is Publisher, 'publisher independence not declared')
        need(publisher.fields['q'].shape[1] == 12, 'unknown axonal observation map')
        self.ts = publisher.wrapper.publisher.synaptic_tau
        source = (HERE/'membrane_model.cu').read_text()
        self.engine = BlockExecutor(source, 'cell_epoch', int(batch.n), self.inputs(publisher))
        with self.engine.stream:
            self.ge = cp.zeros((batch.n, 4)); self.gi = cp.zeros_like(self.ge)
            self.current = cp.zeros_like(batch.delta)
            self.gain = publisher.gain.copy()
            self.et = cp.zeros((batch.n, self.width)); self.ej = cp.zeros_like(self.et)
            self.ep = cp.zeros_like(self.et); self.ec = cp.zeros(batch.n, dtype=cp.int32)
            self.flag = cp.zeros(batch.n, dtype=cp.int32)
            self.observation = cp.ascontiguousarray(batch.obs[1])
            self.coords = cp.arange(5, 17, dtype=cp.int32)
            self.vscratch = [cp.empty_like(batch.delta) for _ in range(3)]
            self.gscratch = [cp.empty_like(batch.gates) for _ in range(3)]
            self.errors = cp.zeros((batch.n, 6)); self.clock = cp.zeros((batch.n, 2), dtype=cp.int64)
        self.report = {'calls': 0, 'accepted': 0, 'rejected': 0, 'build_s': self.engine.build_s,
            'native_wall_s': 0., 'event_capacity_per_cell_per_epoch': self.width, 'compressed': False,
            'scheduler': 'generic_transactional_independent_blocks', 'count_unit': 'cell_trials',
            'device_bytes': self.engine.pool.total_bytes(), 'kernel': self.engine.kernel.attributes,
            'implementation_sources': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ('device_cell.py', 'block_executor.py', 'membrane_model.cu',
                             'dense_warp.cuh', 'independent_blocks.cuh')}}

    def inputs(self, publisher):
        return {**{'cell.'+k: getattr(self.b, k) for k in CELL_FIELDS},
                **{'axon.'+k: v for k, v in publisher.fields.items()}}

    def advance(self, batch, ns, ge, gi, *, current_pA=None, inner_step_ns=25000):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError('membrane adapter already in use')
        try:
            return self._advance(batch, ns, ge, gi, current_pA, inner_step_ns)
        finally:
            self._gate.release()

    def _advance(self, batch, ns, ge, gi, current_pA, inner_step_ns):
        need(batch is self.b, 'wrong model owner')
        if self.publication_failed:
            raise RuntimeError('physical publication invalidated; rebuild session')
        need(type(ns) is int and ns > 0 and type(inner_step_ns) is int and 0 < inner_step_ns <= 25000,
             'invalid physical epoch')
        for array in (ge, gi):
            need(array.shape == (batch.n, 4) and np.isfinite(array).all() and np.all(array >= 0),
                 'invalid held conductance')
        p = batch._motor_axonal_callback
        need(p.elapsed == 0 and self.events.active is not None, 'physical publisher ownership mismatch')
        current = None if current_pA is None else cp.asarray(current_pA)
        if current is not None:
            need(current.shape == self.current.shape and bool(cp.isfinite(current).all()), 'invalid current')
        origin = batch.elapsed_ns
        started = time.perf_counter()

        def bind(state, counts, maximum_error, status):
            self.ge.set(np.ascontiguousarray(ge)); self.gi.set(np.ascontiguousarray(gi))
            self.ec.fill(0); self.et.fill(0); self.ej.fill(0); self.ep.fill(0); self.flag.fill(0)
            cp.copyto(self.gain, p.gain)
            if current is None:
                self.current.fill(0)
            else:
                cp.copyto(self.current, current)
            s = {k: state['cell.'+k] for k in CELL_FIELDS}
            a = {k: state['axon.'+k] for k in p.fields}
            return (np.int32(batch.n), np.int64(ns), np.int64(inner_step_ns), np.float64(batch.rest),
                s['delta'], s['gates'], self.ge, self.gi, self.current, batch.C, batch.G,
                batch.chanG, batch.chanb, batch.shuntG, batch.shuntb, batch.ena,
                *self.vscratch, *self.gscratch, self.errors, self.clock, self.observation,
                batch.caps, batch.tau, s['q'], s['last_siz'], s['previous_slope'], s['trough'],
                s['counts'], s['clipped'], self.ec, self.et, self.ej, self.ep, self.flag,
                self.coords, np.float64(self.ts), self.gain, a['q'], a['s'], a['last_voltage'],
                a['previous_slope'], a['trough'], a['counts'], a['clipped'], counts, maximum_error, status)

        state, counts, maximum_error = self.engine.run(self.inputs(p), bind)
        self.report['native_wall_s'] += time.perf_counter()-started
        try:
            with self.engine.stream:
                count = self.ec.get(); et = self.et.get(); ej = self.ej.get(); ep = self.ep.get()
            need(np.all((count >= 0) & (count <= self.width)), 'invalid event count')
            rr, cc = np.where(np.arange(self.width)[None, :] < count[:, None])
            if len(rr):
                order = np.argsort(et[rr, cc], kind='stable'); rr, cc = rr[order], cc[order]
                self.events.active.add(et[rr, cc]+(origin-self.events.start_elapsed)*1e-9,
                    self.events.gamma[rr], ej[rr, cc], posts=ep[rr, cc])
            with self.engine.stream:
                for k in CELL_FIELDS:
                    setattr(batch, k, state['cell.'+k])
                for k in p.fields:
                    cp.copyto(p.fields[k], state['axon.'+k])
            self.engine.stream.synchronize()
            batch.elapsed_ns += ns; p.elapsed = ns
        except BaseException:
            self.publication_failed = True; batch._native_publication_invalid = True
            raise
        stats = getattr(batch, '_motor_adaptive_stats', None)
        if stats is None:
            stats = batch._motor_adaptive_stats = {'accepted': 0, 'rejected': 0,
                'min_step_ns': 25000, 'max_estimated_error': 0.}
        accepted = int(counts[:, 0].sum()); rejected = int(counts[:, 1].sum())
        stats['accepted'] += accepted; stats['rejected'] += rejected
        stats['min_step_ns'] = min(stats['min_step_ns'], int(counts[:, 2].min()))
        stats['max_estimated_error'] = max(stats['max_estimated_error'], float(maximum_error.max()))
        self.report['calls'] += 1; self.report['accepted'] += accepted; self.report['rejected'] += rejected
        self.report['last_accepted_quantiles'] = np.quantile(counts[:, 0], [0, .5, .9, .99, 1]).tolist()
        return batch.host(batch.q)

    def close(self):
        self.engine.close()


def install(brain, events):
    batch = brain._spatial_batch
    while hasattr(batch, 'base'):
        batch = batch.base
    original = batch.advance
    holder = types.SimpleNamespace(core=None, report={'initialized': False})
    def advance(self, *args, **kwargs):
        if holder.core is None:
            holder.core = DeviceCell(self, events); holder.report = holder.core.report
        return holder.core.advance(self, *args, **kwargs)
    batch.advance = types.MethodType(advance, batch)
    def restore():
        batch.advance = original
        if holder.core is not None:
            holder.core.close()
    return holder, restore
