"""Small CUDA fixture for the event-cut proposal policy; not an organism test."""
from pathlib import Path
import sys
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph


def run(library, boundaries):
    def coefficient(z):
        return cp.full_like(z, .4), cp.full_like(z, 100.)
    graph = NativeGraph(np.array([.2], dtype=np.float64), coefficient,
                        rtol=1e-5, atol=1e-7, norm_size=1,
                        native_library=library)
    try:
        nxt, counts, maxerr = graph.advance(100_000, 100_000, 100, 100_000,
                                             boundaries=np.array(boundaries))
        return float(graph.x.get()[0]), counts, maxerr, nxt
    finally:
        graph.close()


if __name__ == '__main__':
    binary = HERE/'event_step_binary_v2'
    reference = binary/'libbaseline.so'
    candidate = binary/'libcandidate.so'
    exact = .4 + (.2-.4)*np.exp(-100.*100_000e-9)
    dense = [0.5e-6, 50e-6]
    r = run(reference, dense)
    c = run(candidate, dense)
    no_r = run(reference, [])
    no_c = run(candidate, [])
    terminal_r = run(reference, [100e-6])
    terminal_c = run(candidate, [100e-6])
    if not (c[1][0] < r[1][0] and r[1][1] == c[1][1] == 0):
        raise ValueError(f'No event-cut step reduction or unexpected reject: {r}, {c}')
    if not (abs(r[0]-exact) < 1e-7 and abs(c[0]-exact) < 1e-7):
        raise ValueError(f'Analytic endpoint drift: {r}, {c}, {exact}')
    if no_r != no_c:
        raise ValueError(f'No-event policy changed: {no_r}, {no_c}')
    if terminal_r != terminal_c:
        raise ValueError(f'Epoch-end policy changed: {terminal_r}, {terminal_c}')
    print({'reference_trials':r[1][0], 'candidate_trials':c[1][0],
           'reference_endpoint':r[0], 'candidate_endpoint':c[0],
           'analytic_endpoint':exact, 'no_event_equal':True,
           'epoch_end_equal':True})
