"""Private FP64 assembly frames. Public outputs retain independent ownership."""
from contextlib import contextmanager
import threading
import cupy as cp


class CoefficientFrame:
    def __init__(self, size):
        self.size = size
        self.arrays = {k: cp.empty(size, dtype=cp.float64) for k in ('target', 'rate')}
        self.used = {}

    def reset(self):
        self.used.clear()

    def base(self, slot, size):
        if slot in self.used or not 0 < size <= self.size:
            raise RuntimeError('Invalid coefficient base layout')
        self.used[slot] = size
        return self.arrays[slot][:size]

    def append(self, slot, prefix, *parts):
        a = self.arrays[slot]
        end = self.used[slot]
        if prefix.data.ptr != a.data.ptr or prefix.ndim != 1 or prefix.size != end:
            raise RuntimeError('Coefficient prefix is not this frame')
        for part in parts:
            if part.ndim != 1 or part.dtype != cp.float64 or end + part.size > self.size:
                raise RuntimeError('Coefficient suffix layout changed')
            if a.data.ptr <= part.data.ptr < a.data.ptr + a.nbytes:
                raise RuntimeError('Coefficient suffix aliases its destination')
            a[end:end + part.size] = part
            end += part.size
        self.used[slot] = end
        return a[:end]


class CoefficientBuffers:
    def __init__(self, size):
        self.size = size
        self.stream = cp.cuda.get_current_stream().ptr
        self.thread = threading.get_ident()
        self.frames = []
        self.depth = 0

    @contextmanager
    def acquire(self, state):
        if (state.dtype != cp.float64 or state.ndim != 1 or state.size != self.size
                or threading.get_ident() != self.thread
                or cp.cuda.get_current_stream().ptr != self.stream):
            raise RuntimeError('Coefficient buffer state, thread or stream changed')
        depth = self.depth
        if depth == len(self.frames):
            self.frames.append(CoefficientFrame(self.size))
        frame = self.frames[depth]
        frame.reset()
        self.depth += 1
        try:
            yield frame
        finally:
            self.depth -= 1
            if self.depth != depth:
                raise RuntimeError('Coefficient frame lifetime mismatch')
