"""Pure source-port projection and fixed-epoch CSR partition; no model IDs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_npz(stem):
    stem = Path(stem)
    data = json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as z:
        def decode(x):
            if isinstance(x, dict) and set(x) == {'__array__'}:
                return z[x['__array__']].copy()
            if isinstance(x, dict):
                return {k: decode(v) for k, v in x.items()}
            return x
        return decode(data)


def convolution(t, tq, ts):
    """Exact exponential convolution with the parent's stable close-tau series."""
    t, tq = np.asarray(t), np.asarray(tq)
    z, den = t * (1. / ts - 1. / tq), 1. - ts / tq
    positive = np.exp(-t / tq) * (-np.expm1(-np.maximum(z, 0.)))
    negative = np.exp(-t / ts) * np.expm1(np.minimum(z, 0.))
    ordinary = np.where(z >= 0., positive, negative) / np.where(den == 0., 1., den)
    series = t / ts * np.exp(-t / ts) * (1. + z/2. + z*z/6. + z*z*z/24.)
    return np.where(np.abs(z) < 1e-5, series, ordinary)


@dataclass(frozen=True)
class Ports:
    rows: np.ndarray
    q: np.ndarray
    s: np.ndarray
    tau: np.ndarray
    ts: float
    ptr: np.ndarray
    times: np.ndarray
    jumps: np.ndarray
    sets: np.ndarray
    posts: np.ndarray
    duration: float
    version: str


def prepare_ports(data, n_sources):
    rows = np.array(data['rows'], dtype=np.int32, copy=True)
    q, s, tau = (np.array(data[k], dtype=np.float64, copy=True) for k in ('q', 's', 'tau'))
    ts, duration = float(data['ts']), int(data['duration_ns']) * 1e-9
    n = len(rows)
    need(rows.ndim == 1 and len(np.unique(rows)) == n, 'Duplicate/invalid source port')
    need(np.all((rows >= 0) & (rows < n_sources)), 'Port source outside graph')
    need(all(x.shape == (n,) and np.isfinite(x).all() for x in (q, s, tau)), 'Invalid q/s/tau')
    need(np.all(tau > 0.) and np.isfinite(ts) and ts > 0. and duration > 0., 'Invalid tau/duration')
    times, jumps, posts = (np.array(data[k], dtype=np.float64, copy=True) for k in ('times', 'jumps', 'posts'))
    erows = np.array(data['event_rows'], dtype=np.int32, copy=True)
    sets = np.array(data['setop'], dtype=np.bool_, copy=True)
    ne = len(times)
    need(all(x.shape == (ne,) for x in (times, jumps, posts, erows, sets)), 'Event layout')
    need(np.isfinite(times).all() and np.isfinite(jumps).all() and np.isfinite(posts[sets]).all(), 'Nonfinite event')
    need(np.all((times >= 0.) & (times <= duration)), 'Event outside epoch')
    need(np.all((erows >= 0) & (erows < n)), 'Event source outside port layout')
    # Original ordinal is an explicit tie-breaker for repeated same-time SET/ADD.
    order = np.lexsort((np.arange(ne), times, erows))
    ptr = np.r_[0, np.cumsum(np.bincount(erows, minlength=n), dtype=np.int64)]
    arrays = [rows, q, s, tau, ptr, times[order], jumps[order], sets[order], posts[order]]
    version = array_digest(*arrays, np.array([ts, duration]))
    for a in arrays:
        a.flags.writeable = False
    return Ports(*arrays[:4], ts, *arrays[4:], duration, version)


def project_cpu(ports, t):
    need(np.isfinite(t) and 0. <= t <= ports.duration, 'Query outside epoch')
    q = ports.q * np.exp(-t / ports.tau)
    s = ports.s * np.exp(-t / ports.ts) + ports.q * convolution(t, ports.tau, ports.ts)
    for i in np.flatnonzero(np.diff(ports.ptr)):
        qi, si, prev = float(ports.q[i]), float(ports.s[i]), 0.
        for k in range(ports.ptr[i], ports.ptr[i + 1]):
            mark = float(ports.times[k])
            if mark > t:
                break
            dt = mark - prev
            si = si * np.exp(-dt / ports.ts) + qi * convolution(dt, ports.tau[i], ports.ts)
            qi *= np.exp(-dt / ports.tau[i])
            qi = float(ports.posts[k]) if ports.sets[k] else qi + float(ports.jumps[k])
            prev = mark
        dt = t - prev
        q[i] = qi * np.exp(-dt / ports.tau[i])
        s[i] = si * np.exp(-dt / ports.ts) + qi * convolution(dt, ports.tau[i], ports.ts)
    need(np.isfinite(q).all() and np.isfinite(s).all(), 'Nonfinite projection')
    return q, s


@dataclass
class Partition:
    port_ptr: np.ndarray
    port_sources: np.ndarray
    port_weights: np.ndarray
    port_positions: np.ndarray
    continuous_ptr: np.ndarray
    continuous_positions: np.ndarray
    active_rows: np.ndarray
    weight_version: str


def partition_csr(ptr, indices, weights, ports, n_sources):
    n = len(ptr) - 1
    need(ptr.dtype == np.int64 and indices.dtype == np.int32 and weights.dtype == np.float64, 'CSR precision/layout')
    need(ptr[0] == 0 and ptr[-1] == len(indices) == len(weights) and np.all(np.diff(ptr) >= 0), 'CSR offsets')
    need(np.all((indices >= 0) & (indices < n_sources)) and np.isfinite(weights).all(), 'CSR domain')
    lookup = np.full(n_sources, -1, dtype=np.int32)
    lookup[ports.rows] = np.arange(len(ports.rows), dtype=np.int32)
    mask = lookup[indices] >= 0
    pp = np.flatnonzero(mask).astype(np.int64, copy=False)
    cp = np.flatnonzero(~mask).astype(np.int64, copy=False)
    # Search in positions preserves empty rows and original within-row edge order.
    pptr = np.searchsorted(pp, ptr).astype(np.int64)
    cptr = ptr - pptr
    need(len(pp) + len(cp) == len(indices), 'Partition edge count')
    return Partition(pptr, indices[pp], weights[pp], pp, cptr, cp,
                     np.flatnonzero(np.diff(pptr)).astype(np.int32),
                     array_digest(ptr, indices, weights))


class EpochGuard:
    """Events and effective W are immutable within one prepared epoch."""
    def __init__(self, weight_version, event_version):
        self.weight_version, self.event_version = weight_version, event_version
        self.maximum_queried_time, self.valid = -np.inf, True

    def query(self, time, weight_version, event_version):
        need(self.valid, 'Epoch invalidated: rebuild required')
        need(weight_version == self.weight_version, 'Wrong effective W version')
        need(event_version == self.event_version, 'Wrong event version: rebuild required')
        need(np.isfinite(time) and time >= 0., 'Invalid query time')
        self.maximum_queried_time = max(self.maximum_queried_time, time)

    def insert_event(self, time):
        self.valid = False
        if time <= self.maximum_queried_time:
            raise ValueError('Late event invalidated epoch: rebuild and replay affected queries')
        raise ValueError('New event invalidated epoch: rebuild required')


def current_cpu(ptr, indices, weights, release, caps, visual, scale, connected):
    """Independent sequential receptor oracle: [signed/positive, negative]."""
    out = np.zeros((len(ptr) - 1, 2), dtype=np.float64)
    for row in range(len(out)):
        for e in range(ptr[row], ptr[row+1]):
            source = indices[e]
            if visual[row]:
                value = (weights[e] * scale) * release[source]
                out[row, 0 if value >= 0. else 1] += abs(value)
            elif connected or not visual[source]:
                out[row, 0] += weights[e] * (release[source] * caps[source])
    return out
