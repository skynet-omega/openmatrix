"""Fixed physical screen for an electrophysiology comparison, not a controller.

The default 180 x 105 degree screen is on a sphere of radius 25 mm. A
30 degree/s edge has a 1 s initial hold, 6 s sweep, and 1 s final hold, with
the screen center crossed at 4 s. These eight seconds are a proposed geometric
reconstruction consistent with a central LED transition at 4 s, NOT a verified
reproduction of the acquisition protocol or a fit to the processed LED trace.

Only screen texture refreshes at 180 Hz. Ray intersections are evaluated from
the current observer pose, so eye/body movement can change the retinal image
between refreshes. The arena center and frame stay at their initial pose.
No cell labels, target neural responses, or motor commands enter this world.
"""
import copy

import numpy as np

from cardinal_grating_world import CardinalGratingWorld


class ElectrophysiologyEdgeWorld(CardinalGratingWorld):
    SCHEMA = 'matrix_electrophysiology_edge_world_v1'
    CONDITIONS = ('on_positive', 'on_negative', 'off_positive', 'off_negative',
                  'dark_static', 'bright_static')

    def __init__(self, start_ns, center_mm, frame, condition):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        self.condition = condition
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.pi)
        self.screen_elevation_rad = float(np.deg2rad(105.))
        self.angular_speed_rad_s = float(np.pi / 6)
        self.initial_hold_ns = 1_000_000_000
        self.sweep_ns = 6_000_000_000
        self.final_hold_ns = 1_000_000_000
        self.refresh_hz = 180
        self.dark_light = 0.
        self.bright_light = 1.
        self.outside_light = 0.
        self._validate()

    @property
    def duration_ns(self):
        return self.initial_hold_ns + self.sweep_ns + self.final_hold_ns

    def _validate(self):
        if self.condition not in self.CONDITIONS:
            raise ValueError('Unknown electrophysiology edge condition')
        clocks = (self.start_ns, self.initial_hold_ns, self.sweep_ns,
                  self.final_hold_ns, self.refresh_hz)
        if any(type(v) is not int for v in clocks):
            raise ValueError('Integer nanosecond clocks and refresh rate required')
        if (self.start_ns < 0 or self.initial_hold_ns < 0 or self.sweep_ns <= 0
                or self.final_hold_ns < 0 or self.refresh_hz <= 0):
            raise ValueError('Invalid edge intervals or refresh rate')
        intervals = (self.initial_hold_ns, self.sweep_ns, self.final_hold_ns)
        if any(ns * self.refresh_hz % 1_000_000_000 for ns in intervals):
            raise ValueError('Edge intervals must end on refresh boundaries')
        if self.center_mm.shape != (3,) or self.frame.shape != (3, 3):
            raise ValueError('Invalid physical arena coordinates')
        if not np.isfinite(self.center_mm).all() or not np.isfinite(self.frame).all():
            raise ValueError('Nonfinite arena coordinates')
        if (not np.allclose(self.frame.T @ self.frame, np.eye(3), atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.det(self.frame), 1., atol=1e-12, rtol=0)):
            raise ValueError('Arena frame must be a proper rotation')
        values = (self.radius_mm, self.screen_azimuth_rad, self.screen_elevation_rad,
                  self.angular_speed_rad_s, self.dark_light, self.bright_light,
                  self.outside_light)
        if not all(np.isscalar(v) and np.isfinite(v) for v in values):
            raise ValueError('Nonfinite or nonscalar screen parameters')
        if (self.radius_mm <= 0 or not 0 < self.screen_azimuth_rad <= np.pi
                or not 0 < self.screen_elevation_rad < np.pi
                or self.angular_speed_rad_s <= 0):
            raise ValueError('Invalid physical screen geometry or speed')
        if not np.isclose(self.angular_speed_rad_s * (self.sweep_ns * 1e-9),
                          self.screen_azimuth_rad, atol=1e-14, rtol=1e-12):
            raise ValueError('Edge speed and sweep duration must span the screen')
        if not 0 <= self.dark_light < self.bright_light <= 1 or self.outside_light != 0:
            raise ValueError('Invalid screen luminances; outside must be dark')

    def refresh_index(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError('Invalid light-world clock')
        # Integer arithmetic is essential at nonintegral-nanosecond refresh
        # boundaries and when the persisted world starts after a long session.
        return (time_ns - self.start_ns) * self.refresh_hz // 1_000_000_000

    def _sweep_frames(self, time_ns):
        frame_index = self.refresh_index(time_ns)
        onset_frame = self.initial_hold_ns * self.refresh_hz // 1_000_000_000
        total_frames = self.sweep_ns * self.refresh_hz // 1_000_000_000
        moving_frames = max(0, min(frame_index - onset_frame, total_frames))
        return moving_frames, total_frames

    def edge_azimuth_rad(self, time_ns):
        moving_frames, total_frames = self._sweep_frames(time_ns)
        if self.condition.endswith('static'):
            raise ValueError('A static screen has no moving edge')
        edge = self.screen_azimuth_rad * (moving_frames / total_frames - .5)
        return -edge if self.condition.endswith('negative') else edge

    def coordinates(self, origins, rays):
        origins = np.asarray(origins, dtype=np.float64)
        rays = np.asarray(rays, dtype=np.float64)
        if origins.ndim == 0 or rays.ndim == 0 or origins.shape[-1] != 3 or rays.shape[-1] != 3:
            raise ValueError('Invalid optical ray shape')
        try:
            np.broadcast_shapes(origins.shape, rays.shape)
        except ValueError as exc:
            raise ValueError('Incompatible optical ray shapes') from exc
        return super().coordinates(origins, rays)

    def luminance(self, origins, rays, time_ns):
        moving_frames, total_frames = self._sweep_frames(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        screen = ((np.abs(azimuth) <= self.screen_azimuth_rad / 2)
                  & (np.abs(elevation) <= self.screen_elevation_rad / 2))
        if self.condition.endswith('static'):
            illuminated = np.full(np.shape(azimuth), self.condition == 'bright_static', dtype=bool)
        else:
            edge = self.edge_azimuth_rad(time_ns)
            # Exact initial/final uniform screens avoid an isolated lit/dark
            # endpoint at the aperture boundary during the holds.
            if moving_frames == 0:
                illuminated = np.zeros(np.shape(azimuth), dtype=bool)
            elif moving_frames == total_frames:
                illuminated = np.ones(np.shape(azimuth), dtype=bool)
            else:
                illuminated = azimuth >= edge if self.condition.endswith('negative') else azimuth <= edge
            if self.condition.startswith('off_'):
                illuminated = ~illuminated
        inside = np.where(illuminated, self.bright_light, self.dark_light)
        return np.where(screen, inside, self.outside_light)

    def state_dict(self):
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or state.get('schema') != cls.SCHEMA:
            raise ValueError('Unknown electrophysiology edge schema')
        expected = cls(0, np.zeros(3), np.eye(3), 'dark_static').state_dict()
        if set(state) != set(expected):
            raise ValueError('Incomplete or unexpected electrophysiology edge state')
        obj = cls(state['start_ns'], state['center_mm'], state['frame'], state['condition'])
        obj.__dict__.update(copy.deepcopy({k: v for k, v in state.items() if k != 'schema'}))
        obj.center_mm = np.asarray(obj.center_mm, dtype=np.float64).copy()
        obj.frame = np.asarray(obj.frame, dtype=np.float64).copy()
        obj._validate()
        return obj
