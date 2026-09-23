"""Bounded native observer. Python runs only at epoch boundaries and final save."""
from pathlib import Path
import json
import numpy as np
import cupy as cp

ENABLED = False
SELECTED = np.array([603, 683, 1529, 97], dtype=np.int32)
IDS = np.array([76431, 81004, 544736, 45199], dtype=np.int64)
TRACERS = []
CAPACITY = 8192
MAX_CALLS = 4


def physical_source():
    here = Path(__file__).resolve().parent
    return (here/'kc_trace.cuh').read_text() + '\n' + (here/'physical_events.cu').read_text()


class Trace:
    def __init__(self, core, module):
        self.core = core
        self.calls = 0
        self.inputs = []
        self.module = module
        self.enabled = ENABLED
        self.begin_kernel = module.get_function('kd_begin')
        channels = len(SELECTED)*13
        self.values = cp.zeros((channels, CAPACITY, 10), dtype=cp.float64)
        self.meta = cp.zeros((channels, CAPACITY, 4), dtype=cp.int64)
        self.counts = cp.zeros(channels, dtype=cp.int32)
        self.overflow = cp.zeros(1, dtype=cp.int32)
        lookup = np.full(core.b.n, -1, dtype=np.int32)
        if core.b.n != 1557:
            raise ValueError('KC diagnostic identity map changed')
        lookup[SELECTED] = np.arange(len(SELECTED), dtype=np.int32)
        self.lookup = cp.asarray(lookup)
        module.get_function('kd_bind')((1,), (1,), (self.values, self.meta, self.counts,
            self.lookup, self.overflow, np.int32(CAPACITY)))
        TRACERS.append(self)

    def begin(self, ns):
        core = self.core
        active = self.enabled and self.calls < MAX_CALLS
        self.begin_kernel((1,), (1,), (np.int32(active), np.int32(self.calls),
                                      np.int64(core.b.elapsed_ns)))
        if active:
            # Inputs to one whole physical epoch, selected rows only.
            record = {'origin_ns': np.array(core.b.elapsed_ns, dtype=np.int64),
                      'duration_ns': np.array(ns, dtype=np.int64),
                      'rest': np.array(core.b.rest), 'ts': np.array(core.ts),
                      'selected_local': SELECTED, 'selected_ids': IDS}
            for name, array in core.state.items():
                record['state_'+name] = cp.asnumpy(array[SELECTED])
            for name, array in core.ax.items():
                record['ax_'+name] = cp.asnumpy(array[SELECTED])
            for name in ('ge', 'gi', 'current', 'gain'):
                record[name] = cp.asnumpy(getattr(core, name)[SELECTED])
            for name in ('caps', 'tau'):
                record[name] = cp.asnumpy(getattr(core.b, name)[SELECTED])
            for name in ('C', 'G', 'chanG', 'chanb', 'shuntG', 'shuntb', 'ena'):
                record[name] = cp.asnumpy(getattr(core.b, name))
            record['observation'] = cp.asnumpy(core.observation)
            self.inputs.append(record)
        self.calls += 1

    def save(self, root):
        root = Path(root)
        root.mkdir(exist_ok=False)
        self.core.stream.synchronize()
        if int(self.overflow.get()[0]):
            raise RuntimeError('KC sideband overflow; capture invalid')
        counts = self.counts.get()
        take = int(counts.max(initial=0))
        np.savez_compressed(root/'trace.npz', values=self.values[:, :take].get(),
            meta=self.meta[:, :take].get(), counts=counts,
            selected_local=SELECTED, selected_ids=IDS)
        for call, record in enumerate(self.inputs):
            np.savez_compressed(root/f'input_{call:02d}.npz', **record)
        (root/'metadata.json').write_text(json.dumps({'enabled': self.enabled,
            'calls_seen': self.calls, 'calls_captured': len(self.inputs),
            'capacity_per_channel': CAPACITY, 'overflow': False,
            'columns': ['voltage', 'previous_voltage', 'increment', 'previous_increment',
                        'previous_trough', 'post_q', 'post_s', 'peak', 'event', 'event_count'],
            'meta_columns': ['call_index', 'absolute_sample_ns', 'accepted_h_ns', 'split'],
            'phase': 'even call=predictor discarded; odd call=accepted physical epoch'},
            indent=2)+'\n')


def save_all(root):
    for i, tracer in enumerate(TRACERS):
        tracer.save(Path(root)/f'kc_native_{i}')
