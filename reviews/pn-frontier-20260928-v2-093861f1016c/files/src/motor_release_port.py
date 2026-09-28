"""Selective future-input intervention on an existing effective muscle pool.

Signals are normalized model release, not spikes or calibrated motor-unit force.
Keep every input slot: removing a signal never increases other contributors.
"""
import numpy as np

POLICY = 'zero_one_future_release_keep_pool_denominator_v1'


def selective_mean(release, removed_position=None):
    values = np.asarray(release)
    if (values.ndim != 1 or not len(values) or values.dtype.kind != 'f'
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        raise ValueError('Finite normalized floating-point motor release required')
    if removed_position is None:
        return float(values.mean())
    if type(removed_position) is not int or not 0 <= removed_position < len(values):
        raise ValueError('An existing motor input position is required')
    masked = values.copy()
    masked[removed_position] = 0.
    return float(masked.mean())
