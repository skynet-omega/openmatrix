"""Unit-explicit numerical primitives; no anatomical or behavioral fitting.

KC is an event LIF reduction, not a channel-resolved action potential. APL is a
passive coupled compartment reduction; its unidentifiable AHP is not simulated.
"""
import math
import numpy as np
import numba


@numba.njit(cache=True, fastmath=False)
def lif_events(voltage, refractory_left, counts, q, vinf, rate, reset,
               threshold, caps, filter_tau, dt, refractory=.0022):
    """Exact crossings for held conductances; exponential spike-rate filter.

    V is mV, rate is /s, all clocks are seconds. Each spike contributes
    1/(cap_Hz*tau_s) to q. The declared inherited ceiling clips only q;
    every event remains in counts and the number of clipping events is returned.
    """
    clipped = 0
    for j in range(len(voltage)):
        t = 0.; v = voltage[j]; left = refractory_left[j]
        value = q[j] * math.exp(-dt/filter_tau[j])
        while t < dt:
            wait = min(left, dt-t)
            t += wait; left -= wait
            if t >= dt: break
            if vinf[j] <= threshold[j]:
                v = vinf[j]+(v-vinf[j])*math.exp(-rate[j]*(dt-t))
                t = dt; break
            crossing = math.log1p((threshold[j]-v)/(vinf[j]-threshold[j]))/rate[j]
            if crossing > dt-t:
                v = vinf[j]+(v-vinf[j])*math.exp(-rate[j]*(dt-t))
                t = dt; break
            t += max(0., crossing); counts[j] += 1
            # At event time, include the decay of all preceding events.
            event_value = value*math.exp((dt-t)/filter_tau[j])+1./(caps[j]*filter_tau[j])
            if event_value > 1.:
                clipped += 1; event_value = 1.
            value = event_value*math.exp(-(dt-t)/filter_tau[j])
            v = reset[j]; left = refractory
        voltage[j] = v; refractory_left[j] = left; q[j] = value
    return clipped


def cascade(q, s, target, tau_q, tau_s, dt):
    """Exact two first-order filters with a constant dimensionless target."""
    eq, es = np.exp(-dt/tau_q), np.exp(-dt/tau_s)
    factor = dt/tau_s*es if tau_q == tau_s else tau_q/(tau_q-tau_s)*(eq-es)
    return target+(q-target)*eq, target+(s-target)*es+(q-target)*factor


def apl_step(voltage, ge, gi, area, rest, rin, tau, coupling_ratio, dt):
    """Backward Euler RC step; mV, nS, nF, seconds.

    Area fractions are an explicit contact-count proxy, not measured area.
    Symmetric coupling G_ij=gL_total*kappa*f_i*f_j conserves current.
    Summed leak/capacitance equal the measured whole-cell proxy; zero-area
    regions receive no invented membrane. No adaptive/AHP current is implied.
    """
    result = voltage.copy()
    for cell in range(len(voltage)):
        active = area[cell] > 0.; f = area[cell, active]
        gl = f/rin; capacitance_nF = gl*tau
        conductance = np.outer(f, f)*coupling_ratio/rin
        laplacian = np.diag(conductance.sum(axis=1))-conductance
        total = gl+ge[cell, active]+gi[cell, active]
        matrix = np.diag(capacitance_nF/dt+total)+laplacian
        rhs = capacitance_nF/dt*voltage[cell, active]+gl*rest-68.*gi[cell, active]
        result[cell, active] = np.linalg.solve(matrix, rhs)
    if not np.isfinite(result).all() or np.any((result < -80.) | (result > 0.)):
        raise FloatingPointError('APL voltage outside passive reversal bounds')
    return result


@numba.njit(cache=True, fastmath=False)
def afferent_conductances(rows, local_ptr, mode, fraction, slot, pn_slot,
                         apl_edge_slot, ptr, idx, weights, transmission, sax,
                         caps, visual, connected, receptor, apl_edge_release,
                         scales, group, pair_route_slot, route_fractions,
                         route_apl, nregions):
    """All existing afferents; separate declared scale for E and I by target.

    Positive/negative weight sign determines E/I as in the inherited model.
    Equal initial scale values are retained priors, not GABA measurements.
    """
    ge = np.zeros(len(rows)); gi = np.zeros(len(rows))
    age = np.zeros((2, nregions)); agi = np.zeros((2, nregions))
    for j in range(len(rows)):
        row = rows[j]
        for e in range(ptr[row], ptr[row+1]):
            pre = idx[e]
            if not connected and visual[pre]: continue
            a = local_ptr[j]+e-ptr[row]; s = transmission[pre]
            if mode[a] == 1: s *= 1.-fraction[a]
            elif mode[a] == 2: s += fraction[a]*(sax[slot[a]]-s)
            if apl_edge_slot[a] >= 0: s = apl_edge_release[apl_edge_slot[a]]
            f = s*caps[pre]
            if pn_slot[a] >= 0: f = (400./.375)*receptor[pn_slot[a]]
            sign = 0 if weights[e] >= 0. else 1
            value = abs(weights[e])*f*scales[group[j], sign]
            if sign == 0: ge[j] += value
            else: gi[j] += value
            if group[j] == 3:
                route = pair_route_slot[a]; cell = route_apl[route]
                for k in range(nregions):
                    contribution = value*route_fractions[route, k]
                    if sign == 0: age[cell, k] += contribution
                    else: agi[cell, k] += contribution
    return ge, gi, age, agi
