"""Persistent FP32 mirror with model-declared mutable positions.

The source remains authoritative and may contain temporary stage weights.
Refresh is enqueued immediately before the consumer on the same CUDA stream.
This module knows no neuron types, biological owners or integration method.
"""
from pathlib import Path
import cupy as cp
import numpy as np


class PersistentFP32:
    def __init__(self, source, positions, *, audit=False):
        if source.dtype != cp.float64 or source.ndim != 1 or not source.flags.c_contiguous:
            raise ValueError('Contiguous FP64 source required')
        raw = np.asarray(positions)
        if raw.dtype.kind not in 'iu' or raw.ndim != 1:
            raise ValueError('Integer position vector required')
        selected = np.unique(raw.astype(np.int64))
        if selected.size and (selected[0] < 0 or selected[-1] >= source.size):
            raise ValueError('Mutable position outside source')
        self.source = source
        self.source_ptr = source.data.ptr
        self.shape = source.shape
        self.positions = cp.asarray(selected)
        self.values = source.astype(cp.float32)
        self.count = int(selected.size)
        self.audit_enabled = bool(audit)
        self.failures = cp.zeros(1, dtype=cp.uint64)
        self.audit_uses = cp.zeros(1, dtype=cp.uint64)
        module = cp.RawModule(
            code=Path(__file__).with_name('persistent_fp32.cu').read_text(),
            options=('--std=c++11', '--fmad=false'),
        )
        self.refresh_kernel = module.get_function('refresh_indexed')
        self.audit_kernel = module.get_function('audit_mirror')
        self.full_refreshes = 0
        self.invalid = False
        self.refresh_at_boundary = False

    def validate_source(self, source):
        if (self.invalid or source.data.ptr != self.source_ptr or source.shape != self.shape
                or source.dtype != cp.float64):
            raise RuntimeError('Weight storage changed: rebuild the captured operator')

    def refresh(self, source):
        self.validate_source(source)
        if self.count:
            self.refresh_kernel(((self.count + 255)//256,), (256,),
                (np.int64(self.count), self.positions, source, self.values))
        if self.audit_enabled:
            self.audit_kernel(((source.size + 255)//256,), (256,),
                (np.int64(source.size), source, self.values, self.failures, self.audit_uses))

    def refresh_all(self):
        # Explicit out-of-graph mutation (e.g. plasticity). Caller orders streams.
        cp.copyto(self.values, self.source, casting='unsafe')
        self.full_refreshes += 1

    def report(self):
        return {'weights': int(self.source.size), 'mutable_positions': self.count,
                'mutable_fraction': self.count/self.source.size,
                'full_refreshes_after_initialization': self.full_refreshes,
                'audit_enabled': self.audit_enabled,
                'audit_mismatches': int(self.failures.get()[0]),
                'audited_evaluations': int(self.audit_uses.get()[0])}
