"""Experimental WT9 active reduction, not identified MaleCNS physiology.

Four electrical ports retain the passive Galerkin C and G matrices. Original
NaT/NaP/K densities integrated over source areas give native channel totals.
Uniform gating within SIZ and axon is an approximation requiring a separate
spiking comparison. A dendritic shunt here is a port shunt, not a measured APL
synapse distribution. Units: seconds, mV, pA, nS and nF.
"""
import numpy as np
from scipy.special import expit


def channel_rates(voltage_mV):
    v = np.asarray(voltage_mV, dtype=float)
    steady = np.stack((expit(.1121*(v+29.13)), expit(-.2*(v+47)),
                       expit(.2717*(v+48.77)), expit(.0502*(v+12.85))), axis=-1)
    taus_ms = np.stack((.1270+3.434*expit(-(v+45.35)/5.98),
                        .36+np.exp((v+20.65)/-10.47), np.ones_like(v),
                        2.03+1.96*expit(-(v-30.83)/3.12)), axis=-1)
    return steady, taus_ms/1000


class FourPortActiveKC:
    port_names = ('soma', 'dendrites', 'SIZ', 'axon')

    def __init__(self, artifact, channel_totals, rest_mV=-71.25):
        with np.load(artifact, allow_pickle=False) as z:
            self.C = z['C_nF'].copy()
            self.G = z['G_nS'].copy()
        for m in (self.C, self.G):
            if m.shape != (4, 4) or not np.all(np.isfinite(m)):
                raise ValueError('Invalid passive projection matrix')
            np.testing.assert_allclose(m, m.T, atol=1e-10)
            if np.linalg.eigvalsh(m).min() <= 0:
                raise ValueError('Passive projection must be positive definite')
        self.rest = float(rest_mV)
        self.gnat = np.array([0., 0., channel_totals['NaT_SIZ_nS'], channel_totals['NaT_axon_nS']])
        self.gnap = np.array([0., 0., channel_totals['NaP_SIZ_nS'], 0.])
        self.gk = np.array([0., 0., channel_totals['K_SIZ_nS'], channel_totals['K_axon_nS']])
        self.reset()

    def reset(self):
        self.voltage_mV = np.full(4, self.rest)
        self.gates = channel_rates(self.voltage_mV)[0]

    def step(self, dt_s, current_pA=None, shunt_nS=None, shunt_reversal_mV=-68.):
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('Positive finite timestep required')
        current = np.zeros(4) if current_pA is None else np.asarray(current_pA, dtype=float)
        shunt = np.zeros(4) if shunt_nS is None else np.asarray(shunt_nS, dtype=float)
        if current.shape != (4,) or shunt.shape != (4,) or np.any(shunt < 0):
            raise ValueError('Current and nonnegative shunt must have four ports')
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(shunt)):
            raise ValueError('Nonfinite electrical input')
        steady, taus = channel_rates(self.voltage_mV)
        gates = steady+(self.gates-steady)*np.exp(-dt_s/taus)
        m, h, p, n = gates.T
        gna = self.gnat*m**3*h+self.gnap*p
        gk = self.gk*n**4
        capacitance_rate = self.C/dt_s
        lhs = capacitance_rate+self.G+np.diag(gna+gk+shunt)
        rhs = (capacitance_rate@self.voltage_mV+self.G@np.full(4, self.rest)
               +60.*gna-80.*gk+shunt*shunt_reversal_mV+current)
        voltage = np.linalg.solve(lhs, rhs)
        if not np.all(np.isfinite(voltage)):
            raise FloatingPointError('Nonfinite membrane state')
        self.voltage_mV = voltage
        self.gates = gates
        return voltage.copy()
