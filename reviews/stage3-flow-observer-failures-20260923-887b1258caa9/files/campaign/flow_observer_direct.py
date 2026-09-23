"""Read-only observer of coefficients consumed by the accepted CNS stage.

The actual target/rate returned to EventCoupling is captured, with the exact
GPU state supplied at that call. Signed edge terms are reconstructed from that
stage state and checked against the independently captured target. The target
is a rate-model coefficient, not a membrane current. For clipped/saturated
targets, the raw input cannot be independently recovered by inverse tanh.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np


TYPES = ('DM1_lPN', 'MBON32', 'LAL170', 'LAL171', 'DNa02')
TARGET_TOL = 1e-9
RATE_TOL = 1e-12


def effective_pn_release(transmission, slots, zfast, zslow, fast_area, slow_area):
    """Reconstruct the declared filtered ORN release on selected PN edges."""
    result = np.asarray(transmission, dtype=np.float64).copy()
    selected = slots >= 0
    if np.any(selected):
        j = slots[selected]
        result[selected] = (fast_area*zfast[j]+slow_area*zslow[j])/(fast_area+slow_area)
    return result


def accepted_midpoint_flag():
    """Inspect this frozen block-midpoint closure without changing its code."""
    from prosthetic_olfactory_brain import GpuProstheticOlfactoryBrain
    fn = GpuProstheticOlfactoryBrain.advance
    if fn.__module__ != 'block_midpoint' or 'active' not in fn.__code__.co_freevars:
        raise RuntimeError('Unexpected block-midpoint owner; no accepted-step observer')
    active = inspect.getclosurevars(fn).nonlocals['active']
    if not isinstance(active, dict):
        raise RuntimeError('Unknown accepted-step flag')
    return active


class FlowObserver:
    def __init__(self, hybrid, metadata, output: Path):
        self.h = hybrid
        self.out = output
        self.phase = None
        self.ms = None
        self.records = []
        self.terms = {}
        self._graph_stage = None
        self._build_calls = 0
        self.max_target_error = 0.0
        self.max_rate_error = 0.0
        self.max_raw_inversion_error = 0.0
        self.censored = 0
        b = hybrid.brain
        table = metadata.set_index('bodyId').reindex(b.node_ids)
        self.types = table['type'].fillna('UNANNOTATED').astype(str).to_numpy()
        self.rows = np.flatnonzero(np.isin(self.types, TYPES))
        selected_types = {self.types[r] for r in self.rows}
        if selected_types != set(TYPES) or len(self.rows) != 10:
            raise ValueError('Expected five bilateral focal types')
        self.by_row = {}
        pn_rows = np.asarray(getattr(hybrid, '_pn_rows', []),dtype=np.int64)
        for row in self.rows:
            if hybrid.visual_mask[row]:
                raise ValueError('Focal receiver is not ordinary nonvisual rate mode')
            for name, specialized in (
                ('ORN terminal', getattr(hybrid, '_orn_rows', [])),
                ('dynamic receptor', hybrid._dynamic_cache['rows']),
                ('PVLP adaptation', getattr(hybrid, '_pvlp_rows', [])),
            ):
                if row in specialized:
                    raise ValueError(f'Focal receiver requires {name} audit')
            start, stop = b.W.indptr[row:row+2]
            edges = np.arange(start, stop, dtype=np.int64)
            pre = b.W.indices[edges].astype(np.int64)
            key = str(int(b.node_ids[row]))
            pn_special = bool(row in pn_rows)
            if pn_special and self.types[row] != 'DM1_lPN':
                raise ValueError('Unexpected prosthetic PN target')
            self.by_row[int(row)] = dict(key=key, edges=edges, pre=pre,
                                         operator='prosthetic_orn_pn' if pn_special else 'ordinary_rate')
            self.terms[key] = []
        self.out.mkdir(parents=True, exist_ok=False)
        self._log = (self.out/'FLOW.jsonl').open('x', encoding='utf-8', buffering=1)
        meta = {
            'schema':'stage3_direct_accepted_coefficient_v1',
            'time_scope':'Last coefficient callback captured by NativeGraph.trial. CUDA graph replay reuses the same buffers; after a successful advance they hold the final accepted trial stage. Its target/rate were returned to the CNS integrator. This is one sample per organism millisecond, not integrated flow.',
            'units':'Inherited signed weighted rate-model input. No claim of ionic current pA or receptor-specific inhibitory current.',
            'target_tolerance':TARGET_TOL,'rate_tolerance':RATE_TOL,
            'rows':{},
        }
        for row, entry in self.by_row.items():
            pre = entry['pre'];key = entry['key']
            meta['rows'][key] = dict(post_row=row,post_type=self.types[row],operator=entry['operator'],
                post_instance=str(table.iloc[row]['instance']),post_soma_side=str(table.iloc[row]['somaSide']),
                edge_positions=entry['edges'].tolist(),pre_rows=pre.tolist(),
                pre_ids=b.node_ids[pre].tolist(),pre_types=self.types[pre].tolist(),
                pre_soma_sides=table.iloc[pre]['somaSide'].fillna('unknown').astype(str).tolist(),
                pre_root_sides=table.iloc[pre]['rootSide'].fillna('unknown').astype(str).tolist(),
                caps=hybrid.caps[pre].tolist())
        (self.out/'FLOW_METADATA.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    def set_interval(self, phase: str, ms: int) -> None:
        self.phase, self.ms = phase, ms

    def capture(self, state, target, rate, drive, light) -> None:
        """Remember arrays built into the graph; replay itself calls no Python."""
        if self.phase is None:
            raise RuntimeError('Coefficient tap outside declared interval')
        self._graph_stage = (state, target, rate, drive, light)
        self._build_calls += 1

    def sample_last(self, adapter) -> None:
        import cupy as cp
        if (self.phase is None or self.ms is None or adapter.core is None or
                self._graph_stage is None or self._build_calls != 18):
            raise RuntimeError(f'No captured six-coefficient CUDA graph stage; build calls={self._build_calls}')
        h = self.h
        state, target, rate, dg, _light = self._graph_stage
        got_target = cp.asnumpy(target[self.rows])
        got_rate = cp.asnumpy(rate[self.rows])
        weights = {row: cp.asnumpy(h.cuda['weights'][entry['edges']])
                   for row, entry in self.by_row.items()}
        drives = cp.asnumpy(dg[self.rows])
        source_release = {row: cp.asnumpy(state[h.transmission_start+entry['pre']])
                          for row, entry in self.by_row.items()}
        q = cp.asnumpy(state[self.rows])
        pn_filter_start = h.orn_pn_synaptic_manifest['parent_state_size']
        pn_filter_count = len(h._orn_rows)
        pn_state = cp.asnumpy(state[pn_filter_start:pn_filter_start+4*pn_filter_count]).reshape(4,-1)
        _, _, pn_zfast, pn_zslow = pn_state
        from prosthetic_olfactory_brain import FAST_AREA, SLOW_AREA
        record = dict(phase=self.phase,ms=self.ms,time_ns=int(h.time_ns),
                      graph_build_coefficient_calls=self._build_calls,
                      graph_clock_s=cp.asnumpy(adapter.core.clock).tolist(),
                      source_time_ns_after_step=int(h._online_source.time_ns),rows={})
        for j, row in enumerate(self.rows):
            entry = self.by_row[int(row)]
            pre = entry['pre']
            release = source_release[int(row)]
            if entry['operator'] == 'prosthetic_orn_pn' and h.orn_synaptic_enabled:
                slot_index = int(np.searchsorted(h._pn_rows,row))
                if slot_index >= len(h._pn_rows) or h._pn_rows[slot_index] != row:
                    raise RuntimeError('PN filter row mapping changed')
                offset = h._pn_local_ptr[slot_index]
                slots = h._pn_slots[offset:offset+len(pre)]
                release = effective_pn_release(release,slots,pn_zfast,pn_zslow,FAST_AREA,SLOW_AREA)
            edge_terms = weights[int(row)] * (release * h.caps[pre])
            if not h.visual_output_connected:
                edge_terms = np.where(h.visual_mask[pre], 0., edge_terms)
            signed_net = float(np.sum(edge_terms,dtype=np.float64))
            predicted = float(max(0., np.tanh(h.rate_gain[row] * (signed_net + drives[j] - h.rate_theta[row]))))
            rate_expected = float(1./h.tau[row])
            target_error = abs(predicted-float(got_target[j]))
            rate_error = abs(rate_expected-float(got_rate[j]))
            if 0. < got_target[j] < 1.-1e-12:
                consumed_net = float(np.arctanh(got_target[j])/h.rate_gain[row]-drives[j]+h.rate_theta[row])
                raw_error = abs(consumed_net-signed_net)
                self.max_raw_inversion_error = max(self.max_raw_inversion_error,raw_error)
                invertible = True
            else:
                consumed_net = None
                raw_error = None
                invertible = False
                self.censored += 1
            self.max_target_error = max(self.max_target_error,target_error)
            self.max_rate_error = max(self.max_rate_error,rate_error)
            if (not np.isfinite(edge_terms).all() or target_error > TARGET_TOL or
                    rate_error > RATE_TOL or (invertible and raw_error > TARGET_TOL)):
                raise ValueError(f'Unreconstructed effective coefficient for {entry["key"]} at {h.time_ns}: target {target_error}, rate {rate_error}')
            self.terms[entry['key']].append(edge_terms.copy())
            record['rows'][entry['key']] = dict(signed_net_reconstructed=signed_net,
                signed_net_consumed_inverse=consumed_net,raw_inversion_error=raw_error,
                inverse_identifiable=invertible,drive=float(drives[j]),
                theta=float(h.rate_theta[row]),target=float(got_target[j]),rate=float(got_rate[j]),
                operator=entry['operator'],target_reconstruction_error=target_error,q_stage=float(q[j]))
        self.records.append(record)
        self._log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')

    def save(self) -> None:
        if not self.records:
            return
        arrays = {'time_ns':np.asarray([r['time_ns'] for r in self.records],dtype=np.int64),
                  'ms':np.asarray([r['ms'] for r in self.records],dtype=np.int32),
                  'phase':np.asarray([r['phase'] for r in self.records])}
        for key, values in self.terms.items():
            arrays['terms_'+key] = np.asarray(values,dtype=np.float64)
        np.savez_compressed(self.out/'FLOW_TERMS.npz',**arrays)
        report = {'samples':len(self.records),'max_target_error':self.max_target_error,
                  'max_rate_error':self.max_rate_error,
                  'max_raw_inversion_error':self.max_raw_inversion_error,
                  'censored_target_samples':self.censored,
                  'scope':'Actual accepted-stage target/rate plus stage-state signed-term reconstruction; no causal intervention or integrated current.'}
        (self.out/'FLOW_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    def close(self) -> None:
        self._log.close()


def install(session, observer: FlowObserver):
    accepted_midpoint_flag()
    h = session.brain
    old = h.coefficients_gpu
    def tapped(state, drive, light):
        target, rate = old(state, drive, light)
        observer.capture(state, target, rate, drive, light)
        return target, rate
    h.coefficients_gpu = tapped
    def undo():
        del h.coefficients_gpu
    return undo
