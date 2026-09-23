"""Isolated, continuous KC gamma axonal-release hypothesis; no CNS coupling.

For externally specified calyx/somatodendritic activity c and local axonal
ACh proxy u (contacts * Hz), this candidate implements

    tau_b db/dt = u/(K + u) * h(x) - b
    tau_x dx/dt = c/(1 + eta*b) - x
    h(x) = 1/(1 + exp((x - x50)/slope)).

The observed compartment and activity dependence motivating these directions
is described in doi:10.1016/j.cub.2022.09.007 and
doi:10.1016/j.cub.2026.01.014. These are NEW phenomenological equations, not
measured receptor kinetics or the published discrete-time simulation.
No numerical parameter, concentration conversion, or calcium observation
model is supplied as biological calibration. x and c are not DeltaF/F.

Only x and b are integrated. c is an independent input, never suppressed by
this open-loop component. A future CNS adapter must identify axonal ROI
fractions and retain calyx/other output terms separately. In particular, this
bench neither removes nor restores any archived recurrent edge.
"""
from dataclasses import dataclass
import math
from numbers import Real


SCHEMA = "matrix_kcgamma_axon_candidate_v1"
STATUS = "BENCH_IMPLEMENTED_NOT_COUPLED_TO_CNS"
SOURCES = ("10.1016/j.cub.2022.09.007", "10.1016/j.cub.2026.01.014")


def _real(value, name, *, positive=False, unit=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an explicit finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0. or (positive and value == 0.) or (unit and value > 1.):
        raise ValueError(f"Invalid range for {name}")
    return value


@dataclass(frozen=True)
class AxonParameters:
    """All six parameters are explicit hypotheses, without biological defaults.

    eta=0 is the ideal receptor-efficacy knockdown control: the axonal
    compartment and its time constant remain. b can still be recorded as a
    latent activation proxy but has zero influence on release.
    """
    tau_b_s: float
    tau_x_s: float
    K_ach_contact_hz: float
    eta: float
    x50: float
    slope: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _real(getattr(self, name), name,
                          positive=name in ("tau_b_s", "tau_x_s", "K_ach_contact_hz", "slope"),
                          unit=name == "x50")
            if name.startswith("tau_") and not math.isfinite(1./value):
                raise ValueError("Reciprocal time constant must be finite")
            object.__setattr__(self, name, value)

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def activity_gate(x, x50, slope):
    """Overflow-safe decreasing sigmoid; activity is a proxy, not voltage."""
    x = _real(x, "x", unit=True)
    x50 = _real(x50, "x50", unit=True)
    slope = _real(slope, "slope", positive=True)
    z = (x-x50)/slope
    if z >= 0.:
        e = math.exp(-z)
        return e/(1.+e)
    return 1./(1.+math.exp(z))


class KcGammaAxonCandidate:
    """Persistent single-unit open-loop bench, with explicit initial x and b.

    ``advance(calyx_activity, local_ach_contact_hz, duration_s)`` holds both
    inputs constant for the requested interval. The adaptive exponential
    midpoint/doubling method integrates x and b together; h acts on receptor
    PRODUCTION, never as a per-step multiplier on the stored b.

    ``activity_dependent=False`` sets h=1 (a voltage-independent hypothesis).
    ``enabled=False`` bypasses the entire new component: output is exactly c,
    x/b are frozen and no extra-state error controller runs. This differs
    from eta=0, whose axonal release still has the tau_x filter. The bypass
    recovers the supplied parent signal, not a graph that this bench lacks.
    Numerical tolerance defaults are solver settings, not biological values.
    """
    def __init__(self, parameters, *, x, b, enabled=True, activity_dependent=True,
                 rtol=1e-7, atol=1e-10, minimum_step_s=1e-10, maximum_step_s=.01):
        if not isinstance(parameters, AxonParameters):
            raise ValueError("Explicit AxonParameters are required")
        if type(enabled) is not bool or type(activity_dependent) is not bool:
            raise ValueError("Component and activity-dependence flags must be booleans")
        self.parameters = parameters
        self.x, self.b = _real(x, "x", unit=True), _real(b, "b", unit=True)
        self.enabled, self.activity_dependent = enabled, activity_dependent
        self.solver = {name: _real(value, name, positive=True) for name, value in
                       (("rtol", rtol), ("atol", atol), ("minimum_step_s", minimum_step_s),
                        ("maximum_step_s", maximum_step_s))}
        if self.solver["minimum_step_s"] > self.solver["maximum_step_s"]:
            raise ValueError("Invalid solver step interval")
        self.next_step_s = self.solver["maximum_step_s"]
        self.elapsed_s = 0.
        self.calyx_activity = 0.
        self.local_ach_contact_hz = 0.
        self.statistics = dict(accepted=0, rejected=0, evaluations=0)

    @property
    def axonal_release(self):
        return self.x if self.enabled else self.calyx_activity

    @property
    def nonaxonal_release(self):
        """Unmodified input, for checking the compartment boundary only."""
        return self.calyx_activity

    def _targets(self, x, b, c, u):
        p = self.parameters
        # Equivalent to u/(K+u), avoiding overflow in K+u.
        occupied = 0. if u == 0. else 1./(1.+p.K_ach_contact_hz/u)
        gate = activity_gate(x, p.x50, p.slope) if self.activity_dependent else 1.
        return c/(1.+p.eta*b), occupied*gate

    def rhs(self, calyx_activity, local_ach_contact_hz, *, state=None):
        """Continuous derivative (dx/dt, db/dt), in inverse seconds."""
        c = _real(calyx_activity, "calyx_activity", unit=True)
        u = _real(local_ach_contact_hz, "local_ach_contact_hz")
        x, b = (self.x, self.b) if state is None else state
        x, b = _real(x, "x", unit=True), _real(b, "b", unit=True)
        if not self.enabled:
            return 0., 0.
        tx, tb = self._targets(x, b, c, u)
        return (tx-x)/self.parameters.tau_x_s, (tb-b)/self.parameters.tau_b_s

    def _validate(self):
        if type(self.enabled) is not bool or type(self.activity_dependent) is not bool:
            raise ValueError("Invalid component flags")
        for name in ("x", "b", "calyx_activity"):
            _real(getattr(self, name), name, unit=True)
        for name in ("elapsed_s", "local_ach_contact_hz"):
            _real(getattr(self, name), name)
        step = _real(self.next_step_s, "next_step_s", positive=True)
        if step > self.solver["maximum_step_s"]:
            raise ValueError("Adaptive step exceeds its configured maximum")
        if set(self.statistics) != {"accepted", "rejected", "evaluations"} or any(
                type(v) is not int or v < 0 for v in self.statistics.values()):
            raise ValueError("Invalid numerical work record")
        if self.statistics["evaluations"] != 6*(self.statistics["accepted"]+self.statistics["rejected"]):
            raise ValueError("Numerical work record is inconsistent")

    def advance(self, calyx_activity, local_ach_contact_hz, duration_s):
        c = _real(calyx_activity, "calyx_activity", unit=True)
        u = _real(local_ach_contact_hz, "local_ach_contact_hz")
        duration = _real(duration_s, "duration_s")
        self._validate()
        end = self.elapsed_s+duration
        if not math.isfinite(end) or (duration > 0. and end == self.elapsed_s):
            raise ValueError("Elapsed time cannot represent the requested interval")
        if duration == 0.:
            return self.axonal_release
        if not self.enabled:
            self.elapsed_s, self.calyx_activity, self.local_ach_contact_hz = end, c, u
            return c

        p, numeric = self.parameters, self.solver
        state, remaining, next_step = (self.x, self.b), duration, self.next_step_s
        statistics = self.statistics.copy()

        def midpoint(z, h):
            target = self._targets(*z, c, u)
            rates = (1./p.tau_x_s, 1./p.tau_b_s)
            middle = tuple(v+(-math.expm1(-.5*h*r))*(t-v) for v,t,r in zip(z,target,rates))
            target = self._targets(*middle, c, u)
            return tuple(v+(-math.expm1(-h*r))*(t-v) for v,t,r in zip(z,target,rates))

        attempts = 0
        while remaining > 0.:
            attempts += 1
            if attempts > 1000000:
                raise RuntimeError("Numerical work bound exceeded without committing state")
            h = min(remaining, next_step, numeric["maximum_step_s"])
            full = midpoint(state, h)
            half = midpoint(midpoint(state, .5*h), .5*h)
            error = max(abs(a-b)/(3.*(numeric["atol"]+numeric["rtol"]*max(abs(a), abs(b))))
                        for a,b in zip(full,half))
            statistics["evaluations"] += 6
            if not math.isfinite(error) or not all(math.isfinite(v) and 0. <= v <= 1. for v in half):
                raise FloatingPointError("Axonal/receptor integration left its finite unit domain")
            if error <= 1.:
                state = half
                remaining = 0. if h == remaining else remaining-h
                statistics["accepted"] += 1
                next_step = min(numeric["maximum_step_s"], h*2. if error < .1 else h)
            else:
                statistics["rejected"] += 1
                if h*.5 < numeric["minimum_step_s"]:
                    raise FloatingPointError("Requested local axon accuracy unattainable")
                next_step = h*.5
        self.x, self.b = state
        self.elapsed_s, self.next_step_s, self.statistics = end, next_step, statistics
        self.calyx_activity, self.local_ach_contact_hz = c, u
        return self.axonal_release

    @staticmethod
    def scope():
        return dict(status=STATUS, source_dois=list(SOURCES), parameters_are_hypotheses=True,
                    calyx_input="Independent normalized input; never modified by this open-loop bench",
                    ach_unit="contacts * Hz; not concentration", release_unit="normalized proxy; not DeltaF/F",
                    time_unit="second", measured_kinetics=False, calcium_observation_model=None,
                    cns_coupled=False, weights_or_roi_modified=False,
                    receptor_knockdown="eta=0; compartment retained, receptor efficacy absent",
                    disabled="Direct supplied-calyx bypass; new states frozen; no graph restored",
                    activity_dependence="Gate in receptor production; normalized activity proxy, not measured voltage")

    def state_dict(self):
        self._validate()
        return dict(schema=SCHEMA, scope=self.scope(), parameters=self.parameters.as_dict(),
                    enabled=self.enabled, activity_dependent=self.activity_dependent,
                    x=self.x, b=self.b, elapsed_s=self.elapsed_s,
                    calyx_activity=self.calyx_activity, local_ach_contact_hz=self.local_ach_contact_hz,
                    solver=self.solver.copy(), next_step_s=self.next_step_s, statistics=self.statistics.copy())

    @classmethod
    def from_state_dict(cls, state):
        keys = {"schema", "scope", "parameters", "enabled", "activity_dependent", "x", "b",
                "elapsed_s", "calyx_activity", "local_ach_contact_hz", "solver", "next_step_s", "statistics"}
        if not isinstance(state, dict) or set(state) != keys or state["schema"] != SCHEMA or state["scope"] != cls.scope():
            raise ValueError("Unsupported or incomplete local axon bench state")
        if set(state["parameters"]) != set(AxonParameters.__dataclass_fields__) or set(state["solver"]) != {
                "rtol", "atol", "minimum_step_s", "maximum_step_s"}:
            raise ValueError("Incomplete parameter or solver record")
        obj = cls(AxonParameters(**state["parameters"]), x=state["x"], b=state["b"],
                  enabled=state["enabled"], activity_dependent=state["activity_dependent"], **state["solver"])
        for name in ("elapsed_s", "calyx_activity", "local_ach_contact_hz", "next_step_s"):
            setattr(obj, name, state[name])
        obj.statistics = state["statistics"].copy()
        obj._validate()
        return obj
