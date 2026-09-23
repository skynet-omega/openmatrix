"""Reject unsupported electrical edits in the frozen visual runtime.

These seven values describe constants compiled into the existing release and
CPU/CUDA equations. They are not tunable parameters of that runtime. This
contract changes no equation; fitting different values requires a versioned
implementation and a declared state migration.
"""
import math
from numbers import Real


FIXED_VOLTAGE_PARAMETERS = {
    'voltage_low_mv': -80., 'voltage_high_mv': 0., 'leak_mv': -60.,
    'release_low_mv': -65., 'release_high_mv': -25.,
    'excitatory_reversal_mv': 0., 'inhibitory_reversal_mv': -80.,
}


def validate_electrical_parameters(parameters):
    for name, expected in FIXED_VOLTAGE_PARAMETERS.items():
        value = parameters.get(name)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value != expected):
            raise ValueError(
                f'{name}={value!r} is not implemented by this frozen runtime; '
                f'its CPU/CUDA equations require {expected}. '
                'A physiological change requires a new equation version.')


class FrozenVoltageParameters(dict):
    """Normal parameter mapping with explicit rejection of unsupported edits."""

    def __init__(self, values):
        validate_electrical_parameters(values)
        super().__init__(values)

    def __setitem__(self, key, value):
        if key in FIXED_VOLTAGE_PARAMETERS:
            candidate = dict(self)
            candidate[key] = value
            validate_electrical_parameters(candidate)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        candidate = dict(self)
        candidate.update(changes)
        validate_electrical_parameters(candidate)
        super().update(changes)

    def __ior__(self, other):
        self.update(other)
        return self

    def __delitem__(self, key):
        if key in FIXED_VOLTAGE_PARAMETERS:
            raise ValueError(f'Cannot remove compiled electrical constant {key}')
        super().__delitem__(key)

    def pop(self, key, *default):
        if key in FIXED_VOLTAGE_PARAMETERS:
            raise ValueError(f'Cannot remove compiled electrical constant {key}')
        return super().pop(key, *default)

    def popitem(self):
        if not self:
            return super().popitem()
        key = next(reversed(self))
        value = self[key]
        del self[key]
        return key, value

    def clear(self):
        raise ValueError('Cannot clear compiled electrical constants')

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __deepcopy__(self, memo):
        import copy
        result = type(self)(copy.deepcopy(dict(self), memo))
        memo[id(self)] = result
        return result


def guard_visual_session(session):
    """Protect the operational entrypoint without changing archived sources."""
    session.hybrid.parameters = FrozenVoltageParameters(session.hybrid.parameters)
    return session
