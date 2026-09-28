"""One external-input transformation and a read-only FP32 kernel witness."""
from pathlib import Path
import sys
import types
import numpy as np

H = Path(__file__).resolve().parent
PARENT = H.parent / 'etapa45_reparacion_observada_20260927_52'
sys.path.insert(0, str(PARENT))
from pilot52_owners import AirOwner as ParentAirOwner


def need(ok, message):
    if not ok:
        raise ValueError(message)


def normalize(raw, total):
    raw = np.asarray(raw, dtype=np.float64)
    need(raw.shape == (335,) and np.isfinite(raw).all() and np.all(raw >= 0), 'Invalid JO input')
    need(np.isfinite(total) and total > 0 and raw.sum() > 0, 'Undefined JO normalization')
    scaled = raw * (total / raw.sum())
    post = scaled.astype(np.float32)
    need(np.isfinite(post).all() and np.array_equal(raw > 0, post > 0), 'JO support changed')
    need(abs(post.astype(np.float64).sum() - total) <= 1e-7 * total, 'Post-FP32 total outside contract')
    return post.astype(np.float64)


def installer(brain, stimulus):
    import cupy as cp
    import observer
    import fp32_operator
    ob, restore = observer.install(brain, stimulus)
    old = fp32_operator.FastCSR
    ob.candidate = None
    with np.load(PARENT / 'donors/JO_anatomy_arrays.npz', allow_pickle=False) as z:
        rows = cp.asarray(z['source_rows'], dtype=cp.int64)
    expected = cp.zeros(335, cp.float32)
    seen = cp.zeros(335, cp.float32)
    counts = cp.zeros(2, cp.uint64)
    probe = cp.RawKernel(r'''extern "C" __global__ void observe_jo(
        const long long* rows, const float* drive, const float* expected,
        float* seen, unsigned long long* counts) {
        int j=threadIdx.x+blockIdx.x*blockDim.x;
        if(j<335) {
            float v=drive[rows[j]]; seen[j]=v;
            if(v!=expected[j]) atomicAdd(counts+1,1ULL);
            if(j==0) atomicAdd(counts,1ULL);
        }
    }''', 'observe_jo')
    ob.jo_probe = dict(expected=expected, seen=seen, counts=counts)

    class ObservedInput(old):
        def __init__(self, b, **kw):
            super().__init__(b, **kw)
            original = self.kernel

            def capture(grid, block, args):
                # args[10] is the actual FP32 drive after FastCSR's copy.
                probe((3,), (128,), (rows, args[10], expected, seen, counts))
                return original(grid, block, args)

            self.kernel = capture

    fp32_operator.FastCSR = ObservedInput

    def undo():
        fp32_operator.FastCSR = old
        restore()

    return ob, undo


class AirOwner(ParentAirOwner):
    def __init__(self, run, setting, prefix, *, matched):
        super().__init__(run, setting, prefix)
        self.matched = matched
        initial_v = self.velocity_mm_s()
        self.reference_totals = [float(self.api.encode(
            air_velocity_world_mm_s=self.origin_rotation @ np.array([0., 100. * side, 0.]),
            body_velocity_world_mm_s=initial_v,
            body_to_world=self.origin_rotation).sum()) for side in (1, -1)]
        self.total = min(self.reference_totals)
        self.raw_history, self.post_history, self.call_history = [], [], []
        parent_consume = run.stimulus.consume

        def consume(owner, k):
            returned = parent_consume(k)
            raw = self.last.copy()
            if matched and k - self.origin > prefix:
                self.last = normalize(raw, self.total)
                full = np.zeros(run.obj.core.brain.n_neurons, np.float64)
                full[self.api.rows] = self.last
                self.device.set(full)
            probe = run.observer.jo_probe
            probe['expected'].set(self.last.astype(np.float32))
            probe['counts'].fill(0)
            self.raw_history.append(raw)
            self.cp.cuda.get_current_stream().synchronize()
            return returned

        run.stimulus.consume = types.MethodType(consume, run.stimulus)

    def check(self):
        super().check()
        probe = self.run.observer.jo_probe
        counts = probe['counts'].get()
        need(counts[0] > 0 and counts[1] == 0, 'FP32 kernel input differs on one or more evaluations')
        post = probe['seen'].get()
        need(np.array_equal(post, self.last.astype(np.float32)), 'Last consumed input mismatch')
        if self.matched and self.run.stimulus.k - self.origin > self.prefix:
            need(abs(post.astype(np.float64).sum()-self.total) <= 1e-7*self.total, 'Consumed total outside contract')
        self.post_history.append(post.copy())
        self.call_history.append(counts.copy())
