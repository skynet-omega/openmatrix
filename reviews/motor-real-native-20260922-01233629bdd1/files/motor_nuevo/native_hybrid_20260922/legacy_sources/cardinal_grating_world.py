"""Persisted physical gratings for a situated visual assay, not a controller.

The arena frame is fixed when the assay begins. Subsequent eye/body motion
changes the sampled image. Angular speed is in arena coordinates; azimuth
speed on the surface contracts with elevation. These stimuli are exploratory,
not a reproduction of a physiological recording protocol.
"""
import copy
import numpy as np


class CardinalGratingWorld:
    SCHEMA = 'matrix_cardinal_grating_world_v1'
    CONDITIONS = ('azimuth_positive', 'azimuth_negative', 'azimuth_static',
                  'elevation_positive', 'elevation_negative', 'elevation_static')

    def __init__(self, start_ns, center_mm, frame, condition):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        self.condition = condition
        self.radius_mm = 25.
        self.mean_light = .5
        self.contrast = .8
        self.spatial_period_rad = float(np.pi/6)
        self.angular_speed_rad_s = float(np.pi/3)
        self.initial_phase_rad = 0.
        self.onset_ns = 100_000_000
        self.offset_ns = 1_100_000_000
        self._validate()

    def _validate(self):
        if self.condition not in self.CONDITIONS:
            raise ValueError('Unknown grating condition')
        if any(type(v) is not int for v in (self.start_ns, self.onset_ns, self.offset_ns)):
            raise ValueError('Integer nanosecond clocks required')
        if not (self.start_ns >= 0 and 0 <= self.onset_ns < self.offset_ns):
            raise ValueError('Invalid grating interval')
        if self.center_mm.shape != (3,) or self.frame.shape != (3,3):
            raise ValueError('Invalid physical arena coordinates')
        if not np.isfinite(self.center_mm).all() or not np.isfinite(self.frame).all():
            raise ValueError('Nonfinite arena coordinates')
        if not np.allclose(self.frame.T@self.frame, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(self.frame), 1., atol=1e-12, rtol=0):
            raise ValueError('Arena frame must be a proper rotation')
        values = (self.radius_mm, self.mean_light, self.contrast, self.spatial_period_rad,
                  self.angular_speed_rad_s, self.initial_phase_rad)
        if not np.isfinite(values).all() or self.radius_mm <= 0 or not 0 < self.spatial_period_rad <= 2*np.pi or self.angular_speed_rad_s < 0:
            raise ValueError('Invalid physical grating parameters')
        if not 0 <= self.contrast <= 1 or not 0 <= self.mean_light*(1-self.contrast) <= self.mean_light*(1+self.contrast) <= 1:
            raise ValueError('Luminance outside instrument range')
        cycles = 2*np.pi/self.spatial_period_rad
        if not np.isclose(cycles, round(cycles), rtol=0, atol=1e-10):
            raise ValueError('Spatial period must close at the azimuth seam')

    def phase(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError('Invalid light-world clock')
        elapsed = time_ns-self.start_ns
        moving_s = max(0, min(elapsed, self.offset_ns)-self.onset_ns)*1e-9
        sign = 0 if self.condition.endswith('static') else (-1 if self.condition.endswith('negative') else 1)
        return self.initial_phase_rad + sign*self.angular_speed_rad_s*moving_s

    def coordinates(self, origins, rays):
        rays = np.asarray(rays, dtype=np.float64)
        offset = np.asarray(origins, dtype=np.float64)-self.center_mm
        if not np.isfinite(rays).all() or not np.isfinite(offset).all() or rays.shape[-1] != 3 or offset.shape[-1] != 3:
            raise ValueError('Invalid optical rays')
        if not np.allclose(np.sum(rays*rays, axis=-1), 1., atol=1e-10, rtol=0):
            raise ValueError('Optical rays must be unit vectors')
        if np.any(np.linalg.norm(offset, axis=-1) >= self.radius_mm):
            raise ValueError('Eye left the declared arena; no visual rescue')
        projection = np.sum(offset*rays, axis=-1)
        distance = -projection + np.sqrt(projection**2+self.radius_mm**2-np.sum(offset**2, axis=-1))
        hit = (offset+distance[...,None]*rays)@self.frame
        azimuth = np.arctan2(hit[...,1], hit[...,0])
        elevation = np.arctan2(hit[...,2], np.hypot(hit[...,0],hit[...,1]))
        return azimuth, elevation

    def luminance(self, origins, rays, time_ns):
        phase = self.phase(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        coordinate = azimuth if self.condition.startswith('azimuth') else elevation
        # Same physical contrast envelope for both orientations; smooth poles.
        taper = np.cos(elevation)**2
        return self.mean_light*(1+self.contrast*taper*np.sin(2*np.pi/self.spatial_period_rad*(coordinate-phase)))

    def state_dict(self):
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if state.get('schema') != cls.SCHEMA:
            raise ValueError('Unknown cardinal grating schema')
        obj = cls(state['start_ns'], state['center_mm'], state['frame'], state['condition'])
        if set(state) != set(obj.state_dict()):
            raise ValueError('Incomplete grating state')
        obj.__dict__.update(copy.deepcopy({k:v for k,v in state.items() if k != 'schema'}))
        obj.center_mm = np.asarray(obj.center_mm, dtype=np.float64)
        obj.frame = np.asarray(obj.frame, dtype=np.float64)
        obj._validate()
        return obj
