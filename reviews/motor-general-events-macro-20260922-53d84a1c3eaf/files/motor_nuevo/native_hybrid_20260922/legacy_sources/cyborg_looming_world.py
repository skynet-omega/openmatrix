"""A lateral dark looming disk on an explicitly reconstructed optical apparatus.

The proper frame has columns (forward, LEFT, up). Positive display azimuth is
toward anatomical RIGHT, derived from eye centers rather than global Y. Only
ray intersections and display time determine luminance; no neural IDs, motion
detector, motor command or desired action enter this component.
"""
import copy
import math

import numpy as np

from cardinal_grating_world import CardinalGratingWorld


def _proper_frame(value):
    frame = np.asarray(value, dtype=np.float64)
    if (frame.shape != (3, 3) or not np.isfinite(frame).all()
            or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0.)
            or not np.isclose(np.linalg.det(frame), 1., atol=1e-10, rtol=0.)):
        raise ValueError("Optical frame must be a finite proper rotation")
    return frame.copy()


class CyborgLoomingWorld:
    SCHEMA = "matrix_cyborg_looming_world_v1"
    CONDITIONS = ("right", "left", "static")

    def __init__(self, start_ns, center_mm, frame, condition="right", lv_s=.060):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = _proper_frame(frame)
        self.condition = condition
        if isinstance(lv_s, (bool, np.bool_)):
            raise ValueError("l/v must be a positive duration in seconds")
        self.lv_s = float(lv_s)
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.deg2rad(198.))
        self.screen_elevation_rad = float(np.deg2rad(72.))
        self.disk_center_azimuth_rad = float(np.deg2rad(54.))
        self.disk_center_elevation_rad = 0.
        self.minimum_diameter_rad = float(np.deg2rad(6.75))
        self.maximum_diameter_rad = float(np.deg2rad(72.))
        self.onset_ns = 100000000
        self.refresh_hz = 180
        self.background_light = 1.
        self.disk_light = 0.
        self.outside_light = 0.
        self.nominal_collision_time_s = self.lv_s / math.tan(self.minimum_diameter_rad / 2.)
        self.growth_duration_s = (self.nominal_collision_time_s
                                  - self.lv_s / math.tan(self.maximum_diameter_rad / 2.))
        self.manifest = self.protocol_manifest()
        self._validate()

    @classmethod
    def from_camera(cls, time_ns, rotation, centers, condition="right", lv_s=.060):
        rotation = _proper_frame(rotation)
        if not isinstance(centers, dict) or set(centers) != {"L", "R"}:
            raise ValueError("Both named physical eye centers are required")
        left, right = (np.asarray(centers[side], dtype=np.float64) for side in ("L", "R"))
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in (left, right)):
            raise ValueError("Finite three-dimensional physical eye centers required")
        forward = rotation[:, 0]
        anatomical_right = right-left
        anatomical_right = anatomical_right - forward * (anatomical_right @ forward)
        norm = float(np.linalg.norm(anatomical_right))
        if norm <= 1e-12:
            raise ValueError("Eye-center separation does not identify lateral right")
        anatomical_right /= norm
        up = np.cross(anatomical_right, forward)
        up /= np.linalg.norm(up)
        if float(up @ rotation[:, 2]) <= 0.:
            raise ValueError("Named eye sides and current optical up are inconsistent")
        frame = np.column_stack((forward, -anatomical_right, up))
        return cls(time_ns, (left+right)*.5, frame, condition, lv_s)

    def protocol_manifest(self):
        return dict(schema="matrix_looming_optical_reconstruction_manifest_v1",
            source_proposal="evidence/cyborg_functional_20260908/transmission_followup.json",
            frame_columns=["forward", "anatomical_left", "up"],
            display_azimuth_positive="anatomical_right = -frame[:,1]",
            source_compatible_nominal_geometry=dict(azimuth_aperture_deg=198., elevation_aperture_deg=72.,
                disk_center_azimuth_deg=[-54., 54.], minimum_diameter_deg=6.75, maximum_diameter_deg=72.,
                slow_lv_s=.060),
            selected_lv_s=self.lv_s,
            engineering_reconstruction=dict(sphere_radius_mm=25., refresh_hz=180, onset_ns=100000000,
                disk_center_elevation_deg=0., normalized_background=1., normalized_disk=0., outside_light=0.,
                growth="diameter = 2 atan((l/v)/(tc-t)); tc chosen from initial diameter; stop at maximum diameter",
                refresh="last available frame, floor((time_ns-start_ns)*180/1e9); no future-frame interpolation",
                initial_frame="Bright aperture in all conditions; disk first appears at onset in right/left",
                final_frame="Maximum disk remains stationary; static condition has no disk",
                angles_defined_at="Arena center; actual eye-ray origins retain physical parallax"),
            metadata_limits=["180 Hz, sphere radius and 100 ms habituation are engineering settings, not recovered display metadata.",
                "Zero center elevation and theoretical angular growth reconstruct missing pattern frames.",
                "Normalized contrast does not establish measured green spectrum, radiance or photon flux.",
                "Both eyes see the same physical scene through their actual rays; field side is not an eye/neuron injection mask."],
            biological_display_replay_claimed=False, neural_or_motor_controller=False)

    def _validate(self):
        if type(self.start_ns) is not int or self.start_ns < 0:
            raise ValueError("Nonnegative integer nanosecond initialization required")
        if self.condition not in self.CONDITIONS:
            raise ValueError("Unknown lateral looming condition")
        if not math.isfinite(self.lv_s) or self.lv_s <= 0.:
            raise ValueError("l/v must be finite and positive")
        if self.center_mm.shape != (3,) or not np.isfinite(self.center_mm).all():
            raise ValueError("Finite three-dimensional arena center required")
        _proper_frame(self.frame)
        constants = dict(radius_mm=25., screen_azimuth_rad=float(np.deg2rad(198.)),
            screen_elevation_rad=float(np.deg2rad(72.)), disk_center_azimuth_rad=float(np.deg2rad(54.)),
            disk_center_elevation_rad=0., minimum_diameter_rad=float(np.deg2rad(6.75)),
            maximum_diameter_rad=float(np.deg2rad(72.)), onset_ns=100000000, refresh_hz=180,
            background_light=1., disk_light=0., outside_light=0.)
        if any(getattr(self, key) != value for key, value in constants.items()):
            raise ValueError("Unversioned looming apparatus parameter change")
        collision = self.lv_s / math.tan(self.minimum_diameter_rad / 2.)
        duration = collision-self.lv_s / math.tan(self.maximum_diameter_rad / 2.)
        if (not math.isfinite(collision) or not math.isfinite(duration) or duration <= 0.
                or self.nominal_collision_time_s != collision or self.growth_duration_s != duration
                or self.manifest != self.protocol_manifest()):
            raise ValueError("Invalid looming dynamics or reconstruction manifest")

    def frame_index(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError("Cannot observe a frame before arena initialization")
        return (time_ns-self.start_ns)*self.refresh_hz // 1000000000

    def displayed_elapsed_s(self, time_ns):
        return self.frame_index(time_ns) / self.refresh_hz

    def diameter_rad(self, time_ns):
        frame = self.frame_index(time_ns)
        onset_frame = self.onset_ns*self.refresh_hz // 1000000000
        if self.condition == "static" or frame < onset_frame:
            return 0.
        growth_time = (frame-onset_frame) / self.refresh_hz
        if growth_time >= self.growth_duration_s:
            return self.maximum_diameter_rad
        return 2.*math.atan(self.lv_s/(self.nominal_collision_time_s-growth_time))

    def coordinates(self, origins, rays):
        left_azimuth, elevation = CardinalGratingWorld.coordinates(self, origins, rays)
        return -left_azimuth, elevation

    def luminance(self, origins, rays, time_ns):
        diameter = self.diameter_rad(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        visible = ((np.abs(azimuth) <= self.screen_azimuth_rad/2.)
                   & (np.abs(elevation) <= self.screen_elevation_rad/2.))
        light = np.where(visible, self.background_light, self.outside_light)
        if diameter > 0.:
            center = (1. if self.condition == "right" else -1.)*self.disk_center_azimuth_rad
            cosine_distance = np.cos(elevation)*np.cos(azimuth-center)
            inside_disk = cosine_distance >= math.cos(diameter/2.)
            light = np.where(visible & inside_disk, self.disk_light, light)
        return light

    def state_dict(self):
        self._validate()
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or state.get("schema") != cls.SCHEMA:
            raise ValueError("Unknown looming-world schema")
        obj = cls(state["start_ns"], state["center_mm"], state["frame"], state["condition"], state["lv_s"])
        expected = obj.state_dict()
        if set(state) != set(expected):
            raise ValueError("Incomplete looming-world state")
        for key, value in expected.items():
            equal = state[key] == value if isinstance(value, dict) else np.array_equal(state[key], value)
            if not equal:
                raise ValueError(f"Modified looming-world state: {key}")
        return obj
