"""A fixed physical screen driven by an explicit causal luminance sequence.

Values are normalized optical luminance, not calibrated photon flux. The screen
uses the existing ray/sphere geometry; it receives no neural IDs or responses.
No interpolation, spectral filtering, clipping or future samples are used.
"""
import copy
import hashlib

import numpy as np

from cardinal_grating_world import CardinalGratingWorld


class MeasuredLuminanceWorld:
    SCHEMA = "matrix_measured_luminance_world_v1"
    CONDITIONS = ("measured", "static")
    coordinates = CardinalGratingWorld.coordinates

    def __init__(self, start_ns, center_mm, frame, sample_elapsed_ns, luminance,
                 condition="measured"):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        clocks = np.asarray(sample_elapsed_ns)
        if clocks.dtype.kind not in "iu" or clocks.dtype.kind == "b":
            raise ValueError("Sample times must be integer nanoseconds")
        if clocks.dtype.kind == "u" and np.any(clocks > np.iinfo(np.int64).max):
            raise ValueError("Sample times exceed int64 range")
        self.sample_elapsed_ns = clocks.astype(np.int64, copy=True)
        self.luminance_samples = np.asarray(luminance, dtype=np.float64).copy()
        self.condition = condition
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.pi)
        self.screen_elevation_rad = float(np.deg2rad(105.))
        self.outside_light = 0.
        self.sequence_sha256 = self.sequence_hash()
        self._validate()

    @property
    def duration_ns(self):
        return int(self.sample_elapsed_ns[-1])

    def sequence_hash(self):
        h = hashlib.sha256()
        for value, dtype in ((self.sample_elapsed_ns, "<i8"), (self.luminance_samples, "<f8")):
            h.update(np.asarray(value, dtype=dtype).tobytes())
        return h.hexdigest()

    def _validate(self):
        if type(self.start_ns) is not int or self.start_ns < 0:
            raise ValueError("Nonnegative integer absolute clock required")
        if self.condition not in self.CONDITIONS:
            raise ValueError("Unknown luminance condition")
        t, v = self.sample_elapsed_ns, self.luminance_samples
        if (t.dtype != np.int64 or t.ndim != 1 or len(t) < 2 or t[0] != 0
                or np.any(t < 0) or np.any(t[1:] <= t[:-1])):
            raise ValueError("Strictly increasing sample times must start at zero")
        if (v.dtype != np.float64 or v.shape != t.shape or not np.isfinite(v).all()
                or np.any(v < 0) or np.any(v > 1)):
            raise ValueError("Finite luminance samples must lie in [0,1]; no clipping")
        if self.sequence_sha256 != self.sequence_hash():
            raise ValueError("Persisted optical sequence hash disagrees")
        if (self.center_mm.shape != (3,) or self.frame.shape != (3, 3)
                or not np.isfinite(self.center_mm).all() or not np.isfinite(self.frame).all()
                or not np.allclose(self.frame.T @ self.frame, np.eye(3), atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.det(self.frame), 1., atol=1e-12, rtol=0)):
            raise ValueError("Finite physical center and proper rotation required")
        values = (self.radius_mm, self.screen_azimuth_rad, self.screen_elevation_rad, self.outside_light)
        if (not all(np.isscalar(x) and np.isfinite(x) for x in values)
                or self.radius_mm <= 0 or not 0 < self.screen_azimuth_rad <= np.pi
                or not 0 < self.screen_elevation_rad < np.pi or self.outside_light != 0.):
            raise ValueError("Invalid fixed screen geometry or exterior luminance")

    def sample_index(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError("Cannot present light before the recorded start")
        elapsed = time_ns - self.start_ns
        if elapsed >= self.duration_ns:
            return len(self.sample_elapsed_ns) - 1
        return int(np.searchsorted(self.sample_elapsed_ns, elapsed, side="right") - 1)

    def screen_luminance(self, time_ns):
        index = self.sample_index(time_ns)
        return float(self.luminance_samples[0 if self.condition == "static" else index])

    def luminance(self, origins, rays, time_ns):
        value = self.screen_luminance(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        aperture = ((np.abs(azimuth) <= self.screen_azimuth_rad / 2)
                    & (np.abs(elevation) <= self.screen_elevation_rad / 2))
        return np.where(aperture, value, self.outside_light)

    def state_dict(self):
        self._validate()
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        expected = {"schema", "start_ns", "center_mm", "frame", "sample_elapsed_ns",
                    "luminance_samples", "condition", "radius_mm", "screen_azimuth_rad",
                    "screen_elevation_rad", "outside_light", "sequence_sha256"}
        if not isinstance(state, dict) or set(state) != expected or state.get("schema") != cls.SCHEMA:
            raise ValueError("Incomplete or unknown measured luminance state")
        obj = cls(state["start_ns"], state["center_mm"], state["frame"],
                  state["sample_elapsed_ns"], state["luminance_samples"], state["condition"])
        for name in ("radius_mm", "screen_azimuth_rad", "screen_elevation_rad", "outside_light", "sequence_sha256"):
            setattr(obj, name, copy.deepcopy(state[name]))
        obj._validate()
        return obj
