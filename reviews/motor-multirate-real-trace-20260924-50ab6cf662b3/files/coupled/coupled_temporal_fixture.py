"""Three-cell waveform-relaxation fixture with a timestamped late SET event.

Bounded diagnostic only: no organism, no claim of certified global error.
Budget: one CPU run, <= 5 s, eight 125-us blocks, <= 4 Picard sweeps/block.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


W = ((0.14, -0.08, 0.05), (0.11, 0.08, -0.04), (-0.03, 0.12, 0.10))
EVENT_COLUMN = (2.0, -0.4, 0.3)
TAU_X = (80e-6, 110e-6, 90e-6)
TAU_S = 180e-6
TAU_Q_EVENT = 20e-6
TAU_S_EVENT = 20e-6
EVENT_TIME = 0.9437e-3
EVENT_SET_Q = 0.9
H = 1e-3
BLOCK = 125e-6
N = 3
Y0 = (0.23, 0.31, 0.19, 0.23, 0.31, 0.19)


class Counter:
    def __init__(self):
        self.products = 0

    def mv(self, x):
        self.products += 1
        return tuple(sum(W[i][j] * x[j] for j in range(N)) for i in range(N))


def event_s(t: float, *, active: bool = True) -> float:
    if not active or t < EVENT_TIME:
        return 0.0
    u = t - EVENT_TIME
    # q(SET)=0.9. The equal-tau limit of q'=-q/tq, s'=(q-s)/ts.
    if TAU_Q_EVENT != TAU_S_EVENT:
        raise RuntimeError("This fixture freezes the equal-time-constant case")
    return EVENT_SET_Q * (u / TAU_S_EVENT) * math.exp(-u / TAU_S_EVENT)


def target(i: int, current: float) -> float:
    return min(1.0, max(0.0, 0.20 + current))


def g_interpolated(t: float, t0: float, t1: float, nodes):
    h = t1 - t0
    f = max(0.0, min(1.0, (t - t0) / h))
    if f <= 0.5:
        a, b, alpha = nodes[0], nodes[1], 2.0 * f
    else:
        a, b, alpha = nodes[1], nodes[2], 2.0 * f - 1.0
    return tuple(a[i] + alpha * (b[i] - a[i]) for i in range(N))


def deriv(t: float, y, current, *, active: bool):
    ee = event_s(t, active=active)
    x, s = y[:N], y[N:]
    return tuple((target(i, current[i] + EVENT_COLUMN[i] * ee) - x[i]) / TAU_X[i] for i in range(N)) + tuple((x[i] - s[i]) / TAU_S for i in range(N))


def rk4_step(t, y, h, current_at, *, active: bool):
    k1 = deriv(t, y, current_at(t, y), active=active)
    z = tuple(y[i] + 0.5 * h * k1[i] for i in range(2 * N))
    k2 = deriv(t + 0.5 * h, z, current_at(t + 0.5 * h, z), active=active)
    z = tuple(y[i] + 0.5 * h * k2[i] for i in range(2 * N))
    k3 = deriv(t + 0.5 * h, z, current_at(t + 0.5 * h, z), active=active)
    z = tuple(y[i] + h * k3[i] for i in range(2 * N))
    k4 = deriv(t + h, z, current_at(t + h, z), active=active)
    return tuple(y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(2 * N))


def integrate_block(t0, t1, y0, nodes, *, active: bool):
    # The event and audit times are explicit grid points, even when not aligned
    # with the base 2.5-us grid. No W product occurs in this local integration.
    times = [t0 + (t1 - t0) * k / 50 for k in range(51)]
    times.extend((t0 + 0.25 * (t1 - t0), t0 + 0.5 * (t1 - t0), t0 + 0.75 * (t1 - t0)))
    if t0 < EVENT_TIME < t1:
        times.extend((EVENT_TIME, 0.5 * (EVENT_TIME + t1)))
    times = sorted(set(times))
    y = y0
    recorded = {t0: y0}
    current_at = lambda t, unused: g_interpolated(t, t0, t1, nodes)
    for left, right in zip(times, times[1:]):
        y = rk4_step(left, y, right - left, current_at, active=active)
        recorded[right] = y
    return recorded


def reference(*, active: bool):
    counter = Counter()
    times = [H * k / 1000 for k in range(1001)]
    if active:
        times.append(EVENT_TIME)
    times = sorted(set(times))
    y = Y0
    for left, right in zip(times, times[1:]):
        y = rk4_step(left, y, right - left, lambda t, z: counter.mv(z[N:]), active=active)
    return y, counter.products


def candidate(*, active: bool):
    counter = Counter()
    y = Y0
    total_sweeps = 0
    max_current_residual = 0.0
    max_fixed_point_change = 0.0
    event_post_audited = False
    for block in range(8):
        t0, t1 = block * BLOCK, (block + 1) * BLOCK
        g0 = counter.mv(y[N:])
        nodes = (g0, g0, g0)
        for sweep in range(1, 5):
            states = integrate_block(t0, t1, y, nodes, active=active)
            mid = t0 + 0.5 * BLOCK
            next_nodes = (g0, counter.mv(states[mid][N:]), counter.mv(states[t1][N:]))
            change = max(abs(next_nodes[k][i] - nodes[k][i]) for k in range(3) for i in range(N))
            nodes = next_nodes
            total_sweeps += 1
            max_fixed_point_change = max(max_fixed_point_change, change)
            if change < 1e-8:
                break
        else:
            raise RuntimeError("Four waveform sweeps failed to converge")
        # Final local solve uses the converged current waveform. This performs
        # no W product and includes the exact timestamped event waveform.
        states = integrate_block(t0, t1, y, nodes, active=active)
        probes = [t0 + 0.25 * BLOCK, t0 + 0.75 * BLOCK]
        if active and t0 < EVENT_TIME < t1:
            probes.append(0.5 * (EVENT_TIME + t1))
            event_post_audited = True
        for t in probes:
            true_g = counter.mv(states[t][N:])
            fitted_g = g_interpolated(t, t0, t1, nodes)
            max_current_residual = max(max_current_residual, max(abs(true_g[i] - fitted_g[i]) for i in range(N)))
        y = states[t1]
    return dict(final=y, products=counter.products, sweeps=total_sweeps,
                max_sampled_current_residual=max_current_residual,
                max_fixed_point_change=max_fixed_point_change,
                event_post_audited=event_post_audited)


def main():
    ref, ref_products = reference(active=True)
    without_event, _ = reference(active=False)
    approx = candidate(active=True)
    delta = max(abs(approx['final'][i] - ref[i]) for i in range(2 * N))
    late_effect = max(abs(ref[i] - without_event[i]) for i in range(2 * N))
    result = dict(schema='coupled_temporal_fixture_v1', reference_products=ref_products,
                  candidate=approx, max_final_abs_error=delta,
                  late_event_effect_max=late_effect,
                  event_time_s=EVENT_TIME, duration_s=H,
                  sampled_residual_is_not_certification=True,
                  checks=dict(event_visible=late_effect > 0.01,
                              candidate_tracks_reference=delta < 0.002,
                              sampled_current_defect=approx['max_sampled_current_residual'] < 0.001,
                              event_post_audited=approx['event_post_audited']))
    result['fixture_pass'] = all(result['checks'].values())
    path = Path(__file__).with_name('COUPLED_TEMPORAL_FIXTURE_RESULT.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'candidate'}, sort_keys=True))
    if not result['fixture_pass']:
        raise RuntimeError('Fixture did not meet its frozen diagnostic checks')


if __name__ == '__main__':
    main()
