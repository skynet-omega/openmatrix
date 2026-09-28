"""Fixed optical apparatus for a prosthetic, one-axis sensorimotor assay.

The screen receives geometric rays and time only. Its physical stripe motion
never depends on a neuron, actuator command or measured tracking error.
Normalized luminance is not calibrated biological radiometry.
"""
import copy

import numpy as np

from cardinal_grating_world import CardinalGratingWorld


class CyborgVisualWorld:
    SCHEMA = "matrix_cyborg_visual_world_v1"
    CONDITIONS = ("down", "up", "static")
    coordinates = CardinalGratingWorld.coordinates

    def __init__(self, start_ns, center_mm, frame, condition="down", speed_deg_s=60.):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        self.condition = condition
        self.speed_deg_s = float(speed_deg_s)
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.pi)
        self.screen_elevation_rad = float(np.deg2rad(105.))
        self.mean_light = .5
        self.contrast = .8
        self.period_deg = 30.
        self.motion_onset_ns = 100_000_000
        self.motion_offset_ns = 400_000_000
        self.outside_light = 0.
        self._validate()

    def _validate(self):
        if type(self.start_ns) is not int or self.start_ns < 0:
            raise ValueError("Nonnegative integer screen clock required")
        if self.condition not in self.CONDITIONS:
            raise ValueError("Unknown predefined screen condition")
        if not np.isfinite(self.speed_deg_s) or self.speed_deg_s <= 0:
            raise ValueError("Finite positive physical pattern speed required")
        if (self.center_mm.shape != (3,) or self.frame.shape != (3,3)
            or not np.isfinite(self.center_mm).all() or not np.isfinite(self.frame).all()
            or not np.allclose(self.frame.T@self.frame, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(self.frame),1.,atol=1e-12,rtol=0)):
            raise ValueError("Fixed screen needs a finite center and proper rotation")
        constants = dict(radius_mm=25., screen_azimuth_rad=float(np.pi),
            screen_elevation_rad=float(np.deg2rad(105.)), mean_light=.5,
            contrast=.8, period_deg=30., motion_onset_ns=100_000_000,
            motion_offset_ns=400_000_000, outside_light=0.)
        if any(getattr(self,k) != v for k,v in constants.items()):
            raise ValueError("Unversioned optical protocol change")

    def phase_rad(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError("Cannot observe the apparatus before its start")
        elapsed = time_ns-self.start_ns
        duration = max(0, min(elapsed,self.motion_offset_ns)-self.motion_onset_ns)
        sign = dict(down=-1, up=1, static=0)[self.condition]
        return float(sign*np.deg2rad(self.speed_deg_s)*duration*1e-9)

    def luminance(self, origins, rays, time_ns):
        az, el = self.coordinates(origins,rays)
        visible = ((np.abs(az) <= self.screen_azimuth_rad/2)
                   & (np.abs(el) <= self.screen_elevation_rad/2))
        values = self.mean_light*(1+self.contrast*np.sin(
            (2*np.pi/np.deg2rad(self.period_deg))*(el-self.phase_rad(time_ns))))
        return np.where(visible,values,self.outside_light)

    def state_dict(self):
        self._validate()
        return dict(schema=self.SCHEMA,**copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if not isinstance(state,dict) or state.get("schema") != cls.SCHEMA:
            raise ValueError("Unknown optical apparatus state")
        obj = cls(state["start_ns"],state["center_mm"],state["frame"],
                  state["condition"],state["speed_deg_s"])
        expected = obj.state_dict()
        if set(state) != set(expected) or any(not np.array_equal(state[k],v) for k,v in expected.items()):
            raise ValueError("Incomplete or modified optical apparatus state")
        return obj
