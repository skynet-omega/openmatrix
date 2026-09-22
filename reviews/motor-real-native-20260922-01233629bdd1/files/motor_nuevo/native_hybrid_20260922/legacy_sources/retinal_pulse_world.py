"""Fixed physical screen with a finite light pulse, never a neural stimulus.

The default assay is 1 s dark, 1 s bright, 1 s dark on the same frontal
180 x 105 degree, 25 mm spherical screen as ElectrophysiologyEdgeWorld.
The reversed pulse and uniform controls share its aperture and fixed frame.
Screen texture refreshes at 180 Hz; the actual observer pose determines ray
intersections at every sample, including samples between screen refreshes.

This is an explicit three-second exploratory world protocol, not a claimed
reproduction of a published electrophysiological acquisition. Luminance is
normalized, with no calibrated radiometry. No neuronal IDs, target responses,
internal current injection, or motor instructions enter this world.
"""
import copy

import numpy as np

from electrophysiology_edge_world import ElectrophysiologyEdgeWorld


class RetinalPulseWorld(ElectrophysiologyEdgeWorld):
    SCHEMA = 'matrix_retinal_pulse_world_v1'
    CONDITIONS = ('dark_bright_dark', 'bright_dark_bright',
                  'dark_static', 'bright_static')

    def __init__(self, start_ns, center_mm, frame, condition='dark_bright_dark'):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        self.condition = condition
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.pi)
        self.screen_elevation_rad = float(np.deg2rad(105.))
        self.initial_hold_ns = 1_000_000_000
        self.pulse_ns = 1_000_000_000
        self.final_hold_ns = 1_000_000_000
        self.refresh_hz = 180
        self.dark_light = 0.
        self.bright_light = 1.
        self.outside_light = 0.
        self._validate()

    @property
    def duration_ns(self):
        return self.initial_hold_ns + self.pulse_ns + self.final_hold_ns

    def _validate(self):
        if self.condition not in self.CONDITIONS:
            raise ValueError('Unknown retinal pulse condition')
        clocks = (self.start_ns, self.initial_hold_ns, self.pulse_ns,
                  self.final_hold_ns, self.refresh_hz)
        if any(type(value) is not int for value in clocks):
            raise ValueError('Integer nanosecond clocks and refresh rate required')
        if (self.start_ns < 0 or self.initial_hold_ns < 0 or self.pulse_ns <= 0
                or self.final_hold_ns < 0 or self.refresh_hz <= 0):
            raise ValueError('Invalid pulse intervals or refresh rate')
        intervals = (self.initial_hold_ns, self.pulse_ns, self.final_hold_ns)
        if any(ns * self.refresh_hz % 1_000_000_000 for ns in intervals):
            raise ValueError('Pulse intervals must end on refresh boundaries')
        # Reuse the frozen instrument's geometry/luminance validator without
        # retaining a second world or any edge parameters in pulse state.
        geometry = ElectrophysiologyEdgeWorld(
            self.start_ns, self.center_mm, self.frame, 'dark_static')
        for name in ('radius_mm', 'screen_azimuth_rad', 'screen_elevation_rad',
                     'dark_light', 'bright_light', 'outside_light'):
            setattr(geometry, name, getattr(self, name))
        # Its validator also checks that an edge spans its aperture; preserve
        # that unrelated invariant in this temporary validation object.
        geometry.angular_speed_rad_s = geometry.screen_azimuth_rad / (geometry.sweep_ns * 1e-9)
        geometry._validate()

    def pulse_active(self, time_ns):
        frame_index = self.refresh_index(time_ns)
        onset_frame = self.initial_hold_ns * self.refresh_hz // 1_000_000_000
        offset_frame = (self.initial_hold_ns + self.pulse_ns) * self.refresh_hz // 1_000_000_000
        return onset_frame <= frame_index < offset_frame

    def luminance(self, origins, rays, time_ns):
        active = self.pulse_active(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        aperture = ((np.abs(azimuth) <= self.screen_azimuth_rad / 2)
                    & (np.abs(elevation) <= self.screen_elevation_rad / 2))
        if self.condition == 'dark_bright_dark':
            illuminated = active
        elif self.condition == 'bright_dark_bright':
            illuminated = not active
        else:
            illuminated = self.condition == 'bright_static'
        inside = self.bright_light if illuminated else self.dark_light
        return np.where(aperture, inside, self.outside_light)

    def edge_azimuth_rad(self, time_ns):
        self.refresh_index(time_ns)
        raise ValueError('A uniform retinal pulse has no moving edge')

    def state_dict(self):
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or state.get('schema') != cls.SCHEMA:
            raise ValueError('Unknown retinal pulse schema')
        expected = cls(0, np.zeros(3), np.eye(3)).state_dict()
        if set(state) != set(expected):
            raise ValueError('Incomplete or unexpected retinal pulse state')
        obj = cls(state['start_ns'], state['center_mm'], state['frame'], state['condition'])
        obj.__dict__.update(copy.deepcopy({key: value for key, value in state.items() if key != 'schema'}))
        obj.center_mm = np.asarray(obj.center_mm, dtype=np.float64).copy()
        obj.frame = np.asarray(obj.frame, dtype=np.float64).copy()
        obj._validate()
        return obj
