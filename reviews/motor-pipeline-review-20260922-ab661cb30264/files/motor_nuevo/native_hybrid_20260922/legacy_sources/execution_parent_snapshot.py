"""Private parent snapshots for one PN/CNS advance; archival API unchanged.

Metadata is copied by the reference serializer once at call entry, never
shared with live manifests. Only the explicit evolving fields are refreshed
between coupling intervals. Frames at different nesting depths own separate
buffers. Restore uses the existing strict loader and route rebinding.
"""
import numpy as np
from projection_parallel_brain import GpuProjectionParallelBrain, KEYS

DYNAMIC = frozenset((
    'state', 'next_step_ns', 'statistics', 'time_ns',
    'held_afferent_rate_hz', 'held_boundary_light',
    'kc_apl_dynamic_state', 'kc_electrical_scales_state',
    'kc_spatial_state', 'kc_axonal_state',
))


def copy_values(destination, source):
    """Refresh owned buffers; reject dtype/layout/schema drift explicitly."""
    if isinstance(source, np.ndarray):
        if not isinstance(destination, np.ndarray) or destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError('Execution snapshot array layout changed')
        np.copyto(destination, source, casting='no')
        return destination
    if isinstance(source, dict):
        if not isinstance(destination, dict) or set(destination) != set(source):
            raise ValueError('Execution snapshot dictionary layout changed')
        for key, value in source.items():
            destination[key] = copy_values(destination[key], value)
        return destination
    if type(source) in (int, float, bool, str, type(None)):
        if type(destination) is not type(source):
            raise ValueError('Execution snapshot scalar type changed')
        return source
    raise TypeError('Unhandled evolving snapshot field: ' + type(source).__name__)


def evolving_state(brain):
    spatial = brain._spatial_batch.state_dict()
    spatial['time_ns'] = brain.time_ns
    return dict(state=brain.state, next_step_ns=brain.next_step_ns,
                statistics=brain.statistics, time_ns=brain.time_ns,
                held_afferent_rate_hz=brain.held_afferent_rate_hz,
                held_boundary_light=brain.held_boundary_light,
                kc_apl_dynamic_state=brain.kc_apl_dynamic_state,
                kc_electrical_scales_state=brain.kc_electrical_scales_state,
                kc_spatial_state=spatial,
                kc_axonal_state=brain._axonal_release.state_dict())


class ParentSnapshotFrames:
    def __init__(self, brain):
        # The reference snapshot is also the exhaustive key/layout contract.
        self.template = GpuProjectionParallelBrain.state_dict(brain)
        if set(self.template) != KEYS or not DYNAMIC < set(self.template):
            raise ValueError('Unsupported parent snapshot schema')
        self.frames = []
        self.depth = 0

    def capture(self, brain):
        import copy
        current = evolving_state(brain)
        if set(current) != DYNAMIC:
            raise ValueError('Incomplete evolving snapshot fields')
        if self.depth == len(self.frames):
            frame = dict(self.template)
            for key in DYNAMIC:
                frame[key] = copy.deepcopy(current[key])
            frame_rates = brain.brain.rates.copy()
            self.frames.append((frame, frame_rates))
        frame, rates = self.frames[self.depth]
        for key in DYNAMIC:
            frame[key] = copy_values(frame[key], current[key])
        np.copyto(rates, brain.brain.rates, casting='no')
        self.depth += 1
        return frame, (brain.brain.time_ns, rates)

    def release(self, frame):
        if self.depth == 0 or self.frames[self.depth - 1][0] is not frame:
            raise RuntimeError('Execution snapshots must close in stack order')
        self.depth -= 1
