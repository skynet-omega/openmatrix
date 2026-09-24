"""One captured, generic target/rate oracle over the real prepared operator.

Construction executes model Python; each subsequent query launches one CUDA
graph. The oracle owns scratch/state only and never publishes a physical state.
The caller must retain the same operator/version/ports throughout a replay.
"""
import numpy as np
import cupy as cp
from gpu_coefficient_buffers import CoefficientBuffers


class EffectiveOracle:
    def __init__(self, adapter):
        self.adapter = adapter
        self.core = adapter.core
        self.stream = self.core.stream
        b = adapter.brain
        source = b._online_source
        old_reader = source.general_transmission
        old_buffers = getattr(b, '_coefficient_buffers', None)
        old_statistics = dict(b.statistics)
        self.calls = 0
        try:
            with self.stream:
                self.x = cp.array(self.core.x)
                self.clock = cp.asarray([0., 0.], dtype=cp.float64)
                self.buffers = CoefficientBuffers(self.core.n)
                b._coefficient_buffers = self.buffers
                source.general_transmission = lambda: adapter.pn
                def query():
                    self.z = self.core.project(self.x, self.clock, 0.)
                    a, r = self.core.coefficient(self.z)
                    self.a, self.r = a.copy(), r.copy()
                query()
                self.stream.synchronize()
                query()
                self.stream.synchronize()
                self.stream.begin_capture()
                try:
                    query()
                finally:
                    self.graph = self.stream.end_capture()
            self.stream.synchronize()
        finally:
            source.general_transmission = old_reader
            if old_buffers is None:
                del b._coefficient_buffers
            else:
                b._coefficient_buffers = old_buffers
            b.statistics.clear()
            b.statistics.update(old_statistics)

    def query(self, x, t):
        """Returned arrays are scratch; copy before issuing another query."""
        with self.stream:
            if x.shape != self.x.shape or x.dtype != cp.float64:
                raise ValueError('Oracle state layout')
            cp.copyto(self.x, x)
            self.clock.set(np.asarray([t, 0.], dtype=np.float64))
            self.graph.launch(self.stream)
        self.calls += 1
        return self.z, self.a, self.r
