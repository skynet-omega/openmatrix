"""Float64, error-controlled hybrid dynamics on the SAME canonical CNS graph.

This is an uncalibrated model intervention, not inferred cell physiology.
Visual neurons use graded voltage and saturating release. Other neurons retain
the previous rate equation. Existing signed weights/topology/plasticity remain
owned by AnatomicalRateBrain; float64 matrices are derived computation caches.
No visual feature, velocity, heading, target or motor instruction enters here.
"""
import copy
import numpy as np
import numba
from visual_sparse_kernel import coefficients as sparse_coefficients


def exponential_midpoint(state, h, coefficients):
    """Second-order frozen-affine midpoint; convex when targets are bounded."""
    target, rate = coefficients(state)
    middle = state + (-np.expm1(-.5*h*rate))*(target-state)
    target, rate = coefficients(middle)
    return state + (-np.expm1(-h*rate))*(target-state)


class HybridVisualBrain:
    SCHEMA = "matrix_hybrid_visual_brain_v1"

    def __init__(self, brain, visual_ids, photoreceptor_ids, rtol=1e-5, atol=1e-7):
        self.brain = brain
        self.visual_ids = np.asarray(visual_ids, dtype=np.int64).copy()
        self.photo_ids = np.asarray(photoreceptor_ids, dtype=np.int64).copy()
        self.parameters = dict(rtol=float(rtol), atol=float(atol), maximum_step_ns=1000000,
            minimum_step_ns=100, voltage_low_mv=-80., voltage_high_mv=0., leak_mv=-60.,
            release_low_mv=-65., release_high_mv=-25., excitatory_reversal_mv=0.,
            inhibitory_reversal_mv=-80., conductance_per_stored_weight=1/30.,
            photoconductance_max=1., photo_fast_tau_s=.005, photo_adaptation_tau_s=.25,
            photo_half=.2, adaptation_strength=.8)
        self._build()
        self.state = np.zeros(brain.n_neurons+2*len(self.photo_ids), dtype=np.float64)
        self.state[:brain.n_neurons] = brain.rates.astype(float)/brain.r_max
        # This new voltage variable preserves normalized release except at
        # sub-roundoff magnitudes, which are recorded at model adoption.
        self.state[self.vi] = (15.+40.*self.state[self.vi])/80.
        self.next_step_ns = 250000
        self.statistics = dict(accepted=0, rejected=0, evaluations=0,
                               minimum_accepted_ns=1000000, maximum_accepted_error=0.)
        self.time_ns = brain.time_ns
        self.visual_output_connected = True
        self.publish_rates()

    def _build(self):
        b = self.brain
        p = self.parameters
        if not 0 < p["rtol"] < 1 or not 0 < p["atol"] < 1:
            raise ValueError("Positive finite numerical tolerances required")
        def index(ids):
            indices = np.searchsorted(b.node_ids, ids)
            if ids.ndim != 1 or np.any(np.diff(ids) <= 0) or np.any(indices >= b.n_neurons) or not np.array_equal(b.node_ids[indices], ids):
                raise ValueError("Unknown/duplicate neural identities")
            return indices
        self.vi, self.pi = index(self.visual_ids), index(self.photo_ids)
        self.visual_mask = np.zeros(b.n_neurons, dtype=bool)
        self.visual_mask[self.vi] = True
        if not np.all(self.visual_mask[self.pi]):
            raise ValueError("Every optical input must be a visual neuron")
        self.ri = np.flatnonzero(~self.visual_mask)
        self.photo_local = np.searchsorted(self.vi, self.pi)
        numba.set_num_threads(8)
        self.weights64 = b.W.data.astype(np.float64)
        self.tau = b.tau_s.astype(float)
        self.caps = b.r_max.astype(float)
        self.rate_gain = b.gain.astype(float)/self.caps
        self.rate_theta = b.theta.astype(float)

    def release(self, state=None):
        state = self.state if state is None else state
        release = state[:self.brain.n_neurons].copy()
        # This is a biological-model output nonlinearity, not solver rescue.
        release[self.vi] = np.clip((80.*release[self.vi]-15.)/40., 0., 1.)
        return release

    def publish_rates(self):
        self.brain.rates[:] = (self.release()*self.caps).astype(np.float32)
        self.brain.time_ns = int(self.time_ns)

    def sync_plastic_weights(self, plasticity):
        self.weights64[plasticity.positions] = self.brain.W.data[plasticity.positions].astype(float)

    def _coefficients(self, state, drive, light):
        self.statistics["evaluations"] += 1
        n, m = self.brain.n_neurons, len(self.pi)
        p = self.parameters
        release = self.release(state)
        fast, adaptation = state[n:n+m], state[n+m:]
        photoconductance = p["photoconductance_max"]*fast/(p["photo_half"]+p["adaptation_strength"]*adaptation+fast)
        photo_by_row = np.zeros(n, dtype=np.float64)
        photo_by_row[self.pi] = photoconductance
        target = np.empty_like(state)
        rate = np.empty_like(state)
        # V in [-80,0] mV: E_leak=-60 -> .25, E_exc=0 -> 1,
        # E_inh=-80 -> 0. These are declared, unmeasured hypotheses.
        target[:n], rate[:n] = sparse_coefficients(self.brain.W.indptr, self.brain.W.indices,
            self.weights64, release, self.caps, self.visual_mask, self.tau, self.rate_gain,
            self.rate_theta, drive, photo_by_row, p["conductance_per_stored_weight"], self.visual_output_connected)
        target[n:n+m], rate[n:n+m] = light, 1./p["photo_fast_tau_s"]
        target[n+m:], rate[n+m:] = fast, 1./p["photo_adaptation_tau_s"]
        return target, rate

    def advance(self, dt_ns, drive, light):
        b = self.brain
        drive, light = np.asarray(drive, dtype=float), np.asarray(light, dtype=float)
        if type(dt_ns) is not int or dt_ns <= 0 or b.time_ns != self.time_ns:
            raise ValueError("Invalid neural duration/clock")
        if drive.shape != (b.n_neurons,) or light.shape != (len(self.pi),) or not np.isfinite(drive).all() or not np.isfinite(light).all() or np.any((light < 0) | (light > 1)):
            raise ValueError("Invalid physical sensory input")
        if np.any(drive[self.vi]):
            raise ValueError("Current-mode visual input is not supported; use photoreceptors")
        remaining = dt_ns
        p = self.parameters
        attempts = 0
        coefficients = lambda y: self._coefficients(y, drive, light)
        while remaining:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Numerical work bound exceeded; no rescue or silent continuation")
            h_ns = min(remaining, self.next_step_ns, p["maximum_step_ns"])
            h = h_ns*1e-9
            full = exponential_midpoint(self.state, h, coefficients)
            half = exponential_midpoint(self.state, .5*h, coefficients)
            half = exponential_midpoint(half, .5*h, coefficients)
            scale = p["atol"]+p["rtol"]*np.maximum(np.abs(full), np.abs(half))
            error = float(np.max(np.abs(half-full)/(3.*scale)))
            if not np.isfinite(error) or not np.isfinite(half).all():
                raise FloatingPointError("Nonfinite adaptive neural integration")
            if error <= 1.:
                if np.any(half < 0.) or np.any(half > 1.):
                    raise FloatingPointError("Bound-preserving neural integration left its domain")
                self.state = half
                remaining -= h_ns
                self.statistics["accepted"] += 1
                self.statistics["minimum_accepted_ns"] = min(self.statistics["minimum_accepted_ns"], h_ns)
                self.statistics["maximum_accepted_error"] = max(self.statistics["maximum_accepted_error"], error)
                self.next_step_ns = min(p["maximum_step_ns"], h_ns*2 if error < .1 else h_ns)
            else:
                self.statistics["rejected"] += 1
                if h_ns//2 < p["minimum_step_ns"]:
                    raise FloatingPointError("Requested neural accuracy cannot be attained at allowed step")
                self.next_step_ns = h_ns//2
        self.time_ns += dt_ns
        self.publish_rates()

    def state_dict(self):
        return dict(schema=self.SCHEMA, visual_ids=self.visual_ids.copy(), photo_ids=self.photo_ids.copy(),
                    parameters=copy.deepcopy(self.parameters), state=self.state.copy(),
                    next_step_ns=self.next_step_ns, statistics=copy.deepcopy(self.statistics),
                    time_ns=self.time_ns, visual_output_connected=self.visual_output_connected)

    @classmethod
    def from_state(cls, brain, state):
        expected = {"schema", "visual_ids", "photo_ids", "parameters", "state", "next_step_ns", "statistics", "time_ns", "visual_output_connected"}
        if set(state) != expected or state["schema"] != cls.SCHEMA:
            raise ValueError("Incomplete hybrid neural state")
        obj = cls.__new__(cls)
        obj.brain = brain
        for key in expected - {"schema"}:
            setattr(obj, key, copy.deepcopy(state[key]))
        obj._build()
        if obj.state.dtype != np.float64 or obj.state.shape != (brain.n_neurons+2*len(obj.pi),) or not np.isfinite(obj.state).all() or np.any((obj.state < 0) | (obj.state > 1)):
            raise ValueError("Invalid authoritative float64 neural state")
        if obj.time_ns != brain.time_ns or not np.array_equal((obj.release()*obj.caps).astype(np.float32), brain.rates):
            raise ValueError("Saved float32 view disagrees with authoritative neural state")
        return obj
