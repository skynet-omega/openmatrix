"""Transactional CUDA execution for independent model-defined state blocks.

The model supplies its kernel, arrays and argument binding. This executor owns
private epoch storage, serialization and all-block success checks. A failed
kernel never changes caller state. Returned arrays have independent ownership.
"""
from pathlib import Path
import hashlib
import threading
import time
import numpy as np
import cupy as cp


def need(ok, message):
    if not ok:
        raise ValueError(message)


class BlockExecutor:
    def __init__(self, source, kernel_name, count, state):
        need(type(count) is int and 0 < count < 2**31, 'invalid block count')
        need(isinstance(state, dict) and bool(state), 'state mapping required')
        self.count = count
        self._gate = threading.Lock()
        self.closed = False
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.pool = cp.cuda.MemoryPool()
        self.layout = {}
        for name, value in state.items():
            need(isinstance(name, str) and isinstance(value, cp.ndarray), 'named GPU state required')
            need(value.ndim > 0 and value.shape[0] == count and value.flags.c_contiguous,
                 'contiguous state with one leading block dimension required')
            need(value.dtype.kind in 'fiub', 'unsupported state dtype')
            self.layout[name] = (value.shape, value.dtype)
        cp.cuda.get_current_stream().synchronize()
        before = time.perf_counter()
        with cp.cuda.using_allocator(self.pool.malloc), self.stream:
            self.private = {name: cp.empty_like(value) for name, value in state.items()}
            self.status = cp.empty(count, dtype=cp.int32)
            self.statistics = cp.empty((count, 3), dtype=cp.int64)
            self.error = cp.empty(count, dtype=cp.float64)
            dependency_bytes=b''.join((Path(__file__).parent/name).read_bytes()
                for name in ('dense_warp.cuh','independent_blocks.cuh'))
            compiled_source='// core SHA256 '+hashlib.sha256(dependency_bytes).hexdigest()+'\n'+source
            self.kernel = cp.RawKernel(compiled_source, kernel_name,
                options=('--std=c++17', '--fmad=false', '--prec-div=true', '-I'+str(Path(__file__).resolve().parent)))
            self.kernel.compile()
        self.build_s = time.perf_counter()-before
        self.last_failure = None

    def run(self, state, bind_arguments):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError('block executor already in use')
        try:
            need(not self.closed, 'block executor is closed')
            need(set(state) == set(self.layout), 'state fields changed')
            for name, value in state.items():
                shape, dtype = self.layout[name]
                need(isinstance(value, cp.ndarray) and value.shape == shape and
                     value.dtype == dtype and value.flags.c_contiguous, 'state layout changed: '+name)
            cp.cuda.get_current_stream().synchronize()
            with cp.cuda.using_allocator(self.pool.malloc), self.stream:
                for name, value in state.items():
                    cp.copyto(self.private[name], value)
                self.status.fill(0)
                self.statistics.fill(0)
                self.error.fill(0)
                args = bind_arguments(self.private, self.statistics, self.error, self.status)
                self.kernel((self.count,), (32,), args)
                status = self.status.get()
                counts = self.statistics.get()
                error = self.error.get()
                if np.any(status) or not np.isfinite(error).all() or np.any(error < 0):
                    self.last_failure = {'status': status.tolist(), 'finite_error': bool(np.isfinite(error).all())}
                    raise RuntimeError('block epoch failed before publication: '+str(self.last_failure))
                need(np.all(counts >= 0), 'negative adaptive statistics')
                # Allocate the complete result before exposing any field.
                published = {name: value.copy() for name, value in self.private.items()}
            self.stream.synchronize()
            return published, counts, error
        finally:
            self._gate.release()

    def close(self):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError('block executor already in use')
        try:
            self.stream.synchronize()
            self.closed = True
        finally:
            self._gate.release()
