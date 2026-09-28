"""Persisted optical transfer assay on a fixed physical screen.

Default: 1.5 s background, 2 s shared fluctuation, 0.5 s moving grating.
The 180 Hz frame sequence has a zero-mean multisine through 50 Hz. Its
sample-and-hold presentation has spectral images; continuous irradiance is
not strictly band limited. This is an own experiment, not a replay of source
physiology or a controller. No neural labels or desired responses enter it.
"""
import copy
import hashlib
import json

import numpy as np

from electrophysiology_edge_world import ElectrophysiologyEdgeWorld


class VisualTransferWorld(ElectrophysiologyEdgeWorld):
    SCHEMA = 'matrix_visual_transfer_world_v1'
    CONDITIONS = ('transfer', 'static', 'reverse_motion')
    ORIENTATIONS = ('azimuth', 'elevation')

    def __init__(self, start_ns, center_mm, frame, condition='transfer',
                 background=.4, orientation='azimuth', seed=20260908,
                 adaptation_ns=1_500_000_000, fluctuation_ns=2_000_000_000,
                 motion_ns=500_000_000):
        self.start_ns = start_ns
        self.center_mm = np.asarray(center_mm, dtype=np.float64).copy()
        self.frame = np.asarray(frame, dtype=np.float64).copy()
        self.condition, self.orientation = condition, orientation
        self.background, self.seed = float(background), seed
        self.adaptation_ns, self.fluctuation_ns, self.motion_ns = adaptation_ns, fluctuation_ns, motion_ns
        self.radius_mm = 25.
        self.screen_azimuth_rad = float(np.pi)
        self.screen_elevation_rad = float(np.deg2rad(105.))
        self.refresh_hz = 180
        self.bandwidth_hz = 50.
        self.peak_to_peak = .6
        self.motion_amplitude = .3
        self.spatial_period_rad = float(np.pi/6)
        self.angular_speed_rad_s = float(np.pi/3)
        self.initial_phase_rad = 0.
        self.dark_light, self.bright_light, self.outside_light = 0., 1., 0.
        self.sequence_kind = 'zero_mean_random_sign_sine_multisine_v1'
        self._validate_parameters()
        self.fluctuation_samples = self._make_sequence()
        self.sequence_sha256 = self._sequence_hash(self.fluctuation_samples)
        self.sequence_provenance_sha256 = self._sequence_provenance_hash()
        self.frame_elapsed_ns = self._frame_times()
        self._validate()

    @property
    def duration_ns(self):
        return self.adaptation_ns + self.fluctuation_ns + self.motion_ns

    @staticmethod
    def _sequence_hash(samples):
        return hashlib.sha256(np.ascontiguousarray(samples, dtype='<f8').tobytes()).hexdigest()

    def _frame_times(self):
        count = self.duration_ns*self.refresh_hz//1_000_000_000
        return np.asarray([(k*1_000_000_000+self.refresh_hz-1)//self.refresh_hz
                           for k in range(count+1)], dtype=np.int64)

    def _sequence_provenance_hash(self):
        record = {key:getattr(self, key) for key in ('sequence_kind', 'seed', 'refresh_hz',
                  'bandwidth_hz', 'peak_to_peak', 'fluctuation_ns', 'sequence_sha256')}
        return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _validate_parameters(self):
        if self.condition not in self.CONDITIONS or self.orientation not in self.ORIENTATIONS:
            raise ValueError('Unknown transfer condition or grating orientation')
        clocks = (self.start_ns, self.seed, self.adaptation_ns, self.fluctuation_ns,
                  self.motion_ns, self.refresh_hz)
        if any(type(v) is not int for v in clocks) or self.start_ns < 0 or self.seed < 0:
            raise ValueError('Nonnegative integer start/seed and integer intervals required')
        intervals = (self.adaptation_ns, self.fluctuation_ns, self.motion_ns)
        if (any(v <= 0 for v in intervals) or self.refresh_hz != 180 or
                any(v*self.refresh_hz % 1_000_000_000 for v in intervals)):
            raise ValueError('Positive intervals must end on the retained 180 Hz refresh clock')
        values = (self.background, self.bandwidth_hz, self.peak_to_peak, self.motion_amplitude,
                  self.spatial_period_rad, self.angular_speed_rad_s, self.initial_phase_rad)
        if not all(np.isscalar(v) and np.isfinite(v) for v in values):
            raise ValueError('Finite scalar optical parameters required')
        half = self.peak_to_peak/2
        if (self.peak_to_peak <= 0 or self.motion_amplitude <= 0 or
                not 0 <= self.background-max(half, self.motion_amplitude) or
                not self.background+max(half, self.motion_amplitude) <= 1 or
                not 0 < self.bandwidth_hz < self.refresh_hz/2 or
                not 0 < self.spatial_period_rad <= np.pi or self.angular_speed_rad_s <= 0):
            raise ValueError('Requested light would require clipping or has invalid frequencies')
        if self.sequence_kind != 'zero_mean_random_sign_sine_multisine_v1':
            raise ValueError('Unknown persisted sequence construction')
        count = self.fluctuation_ns*self.refresh_hz//1_000_000_000
        if count < 4 or self.bandwidth_hz*self.fluctuation_ns*1e-9 < 1:
            raise ValueError('Insufficient fluctuation duration for a nonzero Fourier bin')
        instrument = ElectrophysiologyEdgeWorld(self.start_ns, self.center_mm, self.frame, 'dark_static')
        for key in ('radius_mm', 'screen_azimuth_rad', 'screen_elevation_rad',
                    'dark_light', 'bright_light', 'outside_light'):
            setattr(instrument, key, getattr(self, key))
        instrument.angular_speed_rad_s = instrument.screen_azimuth_rad/(instrument.sweep_ns*1e-9)
        instrument._validate()

    def _make_sequence(self):
        count = self.fluctuation_ns*self.refresh_hz//1_000_000_000
        last_bin = int(np.floor(self.bandwidth_hz*count/self.refresh_hz+1e-12))
        rng = np.random.Generator(np.random.PCG64(self.seed))
        spectrum = np.zeros(count//2+1, dtype=np.complex128)
        spectrum[1:last_bin+1] = 1j*(2*rng.integers(0, 2, last_bin)-1)
        signal = np.fft.irfft(spectrum, n=count)
        # Odd periodic waveform: both extrema have the same magnitude and
        # DC=0, so normalization changes no waveform shape and clips nothing.
        left = np.arange(1, (count+1)//2)
        signal[count-left] = -signal[left]
        signal[0] = 0.
        if count % 2 == 0:
            signal[count//2] = 0.
        signal *= (self.peak_to_peak/2)/np.max(np.abs(signal))
        return signal.astype(np.float64)

    def _validate(self):
        self._validate_parameters()
        samples = self.fluctuation_samples
        count = self.fluctuation_ns*self.refresh_hz//1_000_000_000
        if (not isinstance(samples, np.ndarray) or samples.dtype != np.float64 or
                samples.shape != (count,) or not np.isfinite(samples).all() or
                self.sequence_sha256 != self._sequence_hash(samples) or
                self.sequence_provenance_sha256 != self._sequence_provenance_hash()):
            raise ValueError('Invalid or changed persisted fluctuation samples')
        if (abs(float(samples.mean())) > 1e-14 or
                not np.isclose(np.ptp(samples), self.peak_to_peak, atol=1e-14, rtol=0) or
                np.max(np.abs(samples)) > self.peak_to_peak/2+1e-14 or
                np.any(self.background+samples < 0) or np.any(self.background+samples > 1)):
            raise ValueError('Fluctuation must preserve its mean, range and unclipped luminance')
        freq = np.fft.rfftfreq(count, d=1/self.refresh_hz)
        power_amplitude = np.abs(np.fft.rfft(samples)/count)
        if np.any(power_amplitude[freq > self.bandwidth_hz+1e-12] > 1e-13):
            raise ValueError('Persisted frame sequence exceeds its nominal spectral support')
        if (not isinstance(self.frame_elapsed_ns, np.ndarray) or self.frame_elapsed_ns.dtype != np.int64
                or not np.array_equal(self.frame_elapsed_ns, self._frame_times())):
            raise ValueError('Persisted optical refresh times disagree with the clock')

    def segment(self, time_ns):
        index = self.refresh_index(time_ns)
        a = self.adaptation_ns*self.refresh_hz//1_000_000_000
        b = a+self.fluctuation_ns*self.refresh_hz//1_000_000_000
        c = b+self.motion_ns*self.refresh_hz//1_000_000_000
        return 'adaptation' if index < a else 'fluctuation' if index < b else 'motion' if index < c else 'post_motion'

    def modulation(self, time_ns):
        index = self.refresh_index(time_ns)-self.adaptation_ns*self.refresh_hz//1_000_000_000
        if self.condition == 'static' or not 0 <= index < len(self.fluctuation_samples):
            return 0.
        return float(self.fluctuation_samples[index])

    def phase(self, time_ns):
        """Grating spatial displacement in screen-coordinate radians."""
        index = self.refresh_index(time_ns)
        onset = (self.adaptation_ns+self.fluctuation_ns)*self.refresh_hz//1_000_000_000
        duration = self.motion_ns*self.refresh_hz//1_000_000_000
        moving = max(0, min(index-onset, duration))
        direction = -1 if self.condition == 'reverse_motion' else 1
        if self.condition == 'static':
            direction = 0
        return self.initial_phase_rad+direction*self.angular_speed_rad_s*moving/self.refresh_hz

    def luminance(self, origins, rays, time_ns):
        stage = self.segment(time_ns)
        azimuth, elevation = self.coordinates(origins, rays)
        aperture = ((np.abs(azimuth) <= self.screen_azimuth_rad/2)
                    & (np.abs(elevation) <= self.screen_elevation_rad/2))
        if self.condition != 'static' and stage in ('motion', 'post_motion'):
            coordinate = azimuth if self.orientation == 'azimuth' else elevation
            phase = self.phase(time_ns) % self.spatial_period_rad
            inside = self.background+self.motion_amplitude*np.sin(2*np.pi*(coordinate-phase)/self.spatial_period_rad)
        else:
            inside = np.full(np.shape(azimuth), self.background+self.modulation(time_ns))
        return np.where(aperture, inside, self.outside_light)

    def edge_azimuth_rad(self, time_ns):
        self.refresh_index(time_ns)
        raise ValueError('Transfer assay has a grating, not a translating single edge')

    def state_dict(self):
        return dict(schema=self.SCHEMA, **copy.deepcopy(self.__dict__))

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or state.get('schema') != cls.SCHEMA:
            raise ValueError('Unknown visual transfer schema')
        # Restoration consumes saved frames; it never draws or regenerates RNG.
        expected = {'start_ns', 'center_mm', 'frame', 'condition', 'orientation', 'background', 'seed',
            'adaptation_ns', 'fluctuation_ns', 'motion_ns', 'radius_mm', 'screen_azimuth_rad',
            'screen_elevation_rad', 'refresh_hz', 'bandwidth_hz', 'peak_to_peak', 'motion_amplitude',
            'spatial_period_rad', 'angular_speed_rad_s', 'initial_phase_rad', 'dark_light',
            'bright_light', 'outside_light', 'sequence_kind', 'fluctuation_samples',
            'sequence_sha256', 'sequence_provenance_sha256', 'frame_elapsed_ns'}
        if set(state) != expected | {'schema'}:
            raise ValueError('Incomplete or unexpected visual transfer state')
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy({k:v for k,v in state.items() if k != 'schema'}))
        for key in ('center_mm', 'frame', 'fluctuation_samples'):
            setattr(obj, key, np.asarray(getattr(obj, key), dtype=np.float64).copy())
        times = np.asarray(obj.frame_elapsed_ns)
        if times.dtype.kind not in 'iu' or np.any(times > np.iinfo(np.int64).max):
            raise ValueError('Persisted refresh times must be exact integer nanoseconds')
        obj.frame_elapsed_ns = times.astype(np.int64).copy()
        obj._validate()
        return obj
