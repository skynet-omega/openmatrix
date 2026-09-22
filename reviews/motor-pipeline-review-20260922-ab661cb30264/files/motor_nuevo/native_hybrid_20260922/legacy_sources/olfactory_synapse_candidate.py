"""Uncoupled ORN→PN rate-model bench from Nagel, Hong & Wilson (2015).

Source: https://doi.org/10.1038/nn.3895, Methods, rate model for Figs. 7–8.
For each component, dA/dt = -r*s*A + (1-A)/tau_A and
dg/dt = k*s*A - g/tau_g. Here time is seconds, s is spikes/second and
g is nS. The published effective gains are not per-contact conductances.
Parameters were constrained by DM6/VM2 EPSCs and disinhibited odor responses;
they are an explicit transfer hypothesis, not a measurement in male DM1.
No MATRIX weights, neurons, sensory encoder, or motor interface are changed.
"""
from dataclasses import dataclass
import math
from numbers import Real


STATUS = "BENCH_IMPLEMENTED_NOT_COUPLED_TO_CNS"
SCHEMA = "matrix_olfactory_synapse_candidate_v1"
SOURCE_DOI = "10.1038/nn.3895"


@dataclass(frozen=True)
class ComponentParameters:
    r_per_spike: float
    tau_A_s: float
    tau_g_s: float
    k_ns_per_spike: float

    def as_dict(self):
        return dict(r_per_spike=self.r_per_spike, tau_A_s=self.tau_A_s,
                    tau_g_s=self.tau_g_s, k_ns_per_spike=self.k_ns_per_spike)


FAST = ComponentParameters(.23, 1.006, .0093, 20.)
SLOW = ComponentParameters(.0073, 33.247, .080, 1.8)


def _nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _advance_component(A, g, input_rate_hz, duration_s, parameters):
    """Exact resource/conductance solution for a constant presynaptic rate."""
    rate_A = parameters.r_per_spike * input_rate_hz + 1. / parameters.tau_A_s
    rate_g = 1. / parameters.tau_g_s
    equilibrium_A = (1. / parameters.tau_A_s) / rate_A
    decay_g = math.exp(-rate_g * duration_s)
    difference = abs(rate_A - rate_g)
    if difference == 0.:
        convolution = duration_s * decay_g
    else:
        # Symmetric expression avoids overflow when either decay is faster.
        convolution = (math.exp(-min(rate_A, rate_g) * duration_s)
                       * (-math.expm1(-difference * duration_s)) / difference)
    integrated_g_decay = -math.expm1(-rate_g * duration_s) / rate_g
    # The difference is analytically nonnegative; remove only subtraction
    # roundoff when both integrals round to duration_s at extremely small steps.
    recovered_convolution = max(0., integrated_g_decay - convolution)
    new_g = (g * decay_g + parameters.k_ns_per_spike * (
        (input_rate_hz * convolution) * A
        + (input_rate_hz * equilibrium_A) * recovered_convolution))
    new_A = (A * math.exp(-rate_A * duration_s)
             + equilibrium_A * (-math.expm1(-rate_A * duration_s)))
    return new_A, new_g


@dataclass
class OlfactorySynapseCandidate:
    """Persistent two-component bench; default A=1 means a rested synapse.

    ``advance(rate_hz, duration_s)`` returns total conductance in nS. Unknown
    prehistory must be specified or conditioned; a fresh A=1 is not an inferred
    resource state for a continuing CNS checkpoint.
    """
    A_fast: float = 1.
    A_slow: float = 1.
    g_fast: float = 0.
    g_slow: float = 0.
    elapsed_s: float = 0.

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for name in ("A_fast", "A_slow", "g_fast", "g_slow", "elapsed_s"):
            value = _nonnegative_finite(getattr(self, name), name)
            if name.startswith("A_") and value > 1.:
                raise ValueError(f"{name} must be between zero and one")
        if not math.isfinite(self.conductance_ns):
            raise ValueError("total conductance must be finite")

    @property
    def conductance_ns(self):
        return self.g_fast + self.g_slow

    @staticmethod
    def parameters():
        return dict(fast=FAST.as_dict(), slow=SLOW.as_dict(),
                    input_unit="spikes/second", time_unit="second",
                    conductance_unit="nS", source_doi=SOURCE_DOI,
                    scope="Published effective rate-model gains; no per-contact or CNS current conversion")

    def advance(self, input_rate_hz, duration_s):
        rate = _nonnegative_finite(input_rate_hz, "input_rate_hz")
        duration = _nonnegative_finite(duration_s, "duration_s")
        self._validate()
        new_time = self.elapsed_s + duration
        if not math.isfinite(new_time):
            raise ValueError("elapsed time would overflow")
        if duration == 0.:
            return self.conductance_ns
        fast = _advance_component(self.A_fast, self.g_fast, rate, duration, FAST)
        slow = _advance_component(self.A_slow, self.g_slow, rate, duration, SLOW)
        values = (*fast, *slow)
        if (not all(math.isfinite(value) and value >= 0. for value in values)
                or not math.isfinite(fast[1] + slow[1])):
            raise ValueError("synaptic state overflowed")
        self.A_fast, self.g_fast = fast
        self.A_slow, self.g_slow = slow
        self.elapsed_s = new_time
        return self.conductance_ns

    def state_dict(self):
        self._validate()
        return dict(schema=SCHEMA, status=STATUS, parameters=self.parameters(),
                    A_fast=float(self.A_fast), A_slow=float(self.A_slow),
                    g_fast=float(self.g_fast), g_slow=float(self.g_slow),
                    elapsed_s=float(self.elapsed_s))

    @classmethod
    def from_state_dict(cls, state):
        if state.get("schema") != SCHEMA or state.get("status") != STATUS:
            raise ValueError("Unsupported olfactory synapse bench state")
        if state.get("parameters") != cls.parameters():
            raise ValueError("State parameters differ from this published candidate")
        return cls(**{name: state[name] for name in
                      ("A_fast", "A_slow", "g_fast", "g_slow", "elapsed_s")})
