"""Physical retinal challenges; no feature, direction label or target enters CNS.

The sphere texture rotates around a world axis. The body's current eye pose
still determines sampled photons. Uniform flashes test contrast polarity
separately from motion. All conditions and changes are persisted as world state.
"""
import copy
import numpy as np
from retinal_world import LuminousWorld


class VisualChallengeWorld:
    SCHEMA = 'matrix_visual_challenge_world_v1'
    CONDITIONS = ('static', 'yaw_positive', 'yaw_negative', 'pitch_positive',
                  'pitch_negative', 'bright', 'dim')

    def __init__(self, reference, start_ns, condition='static'):
        if condition not in self.CONDITIONS or type(start_ns) is not int or start_ns < 0:
            raise ValueError('Unknown physical visual challenge')
        self.reference = LuminousWorld.from_state(reference.state_dict())
        self.start_ns = start_ns
        self.condition = condition
        self.onset_ns = 100_000_000
        self.offset_ns = 600_000_000
        self.repeat_ns = 1_000_000_000
        self.angular_speed_rad_s = float(np.pi)
        self.flash_delta = .4

    def _phase(self, time_ns):
        if type(time_ns) is not int or time_ns < self.start_ns:
            raise ValueError('Invalid light-world clock')
        cycles, elapsed = divmod(time_ns-self.start_ns, self.repeat_ns)
        moving_ns = cycles*(self.offset_ns-self.onset_ns) + max(0, min(elapsed, self.offset_ns)-self.onset_ns)
        return elapsed, moving_ns*1e-9

    def rotation(self, time_ns):
        _, moving_s = self._phase(time_ns)
        angle = self.angular_speed_rad_s*moving_s
        if self.condition.endswith('negative'):
            angle = -angle
        c, s = np.cos(angle), np.sin(angle)
        if self.condition.startswith('yaw'):
            return np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
        if self.condition.startswith('pitch'):
            return np.array([[c,0.,s], [0.,1.,0.], [-s,0.,c]])
        return np.eye(3)

    def luminance(self, origins, rays, time_ns):
        elapsed, _ = self._phase(time_ns)
        rotation = self.rotation(time_ns)
        center = self.reference.center_mm
        # Explicit physical texture replacement starts with the reference
        # phase held at epoch, independent of neural output or observed head.
        transformed_origins = center + (np.asarray(origins)-center)@rotation
        transformed_rays = np.asarray(rays)@rotation
        image = self.reference.luminance(transformed_origins, transformed_rays,
                                        self.start_ns)
        if self.condition in ('bright', 'dim'):
            value = self.reference.mean_light
            if self.onset_ns <= elapsed < self.offset_ns:
                value += self.flash_delta*(1 if self.condition == 'bright' else -1)
            image = np.full_like(image, value)
        return image

    def state_dict(self):
        return dict(schema=self.SCHEMA, reference=self.reference.state_dict(),
            start_ns=self.start_ns, condition=self.condition, onset_ns=self.onset_ns,
            offset_ns=self.offset_ns, repeat_ns=self.repeat_ns,
            angular_speed_rad_s=self.angular_speed_rad_s, flash_delta=self.flash_delta)

    @classmethod
    def from_state(cls, state):
        if state.get('schema') != cls.SCHEMA:
            raise ValueError('Unknown visual-world schema')
        obj = cls(LuminousWorld.from_state(state['reference']), state['start_ns'], state['condition'])
        if set(state) != set(obj.state_dict()):
            raise ValueError('Incomplete visual challenge')
        for key in ('onset_ns', 'offset_ns', 'repeat_ns', 'angular_speed_rad_s', 'flash_delta'):
            setattr(obj, key, copy.deepcopy(state[key]))
        if any(type(getattr(obj, k)) is not int for k in ('onset_ns','offset_ns','repeat_ns')):
            raise ValueError('Integer stimulus timing required')
        if not 0 <= obj.onset_ns < obj.offset_ns < obj.repeat_ns:
            raise ValueError('Invalid stimulus intervals')
        if not np.isfinite([obj.angular_speed_rad_s,obj.flash_delta]).all() or obj.angular_speed_rad_s < 0 or not 0 <= obj.flash_delta <= min(obj.reference.mean_light,1-obj.reference.mean_light):
            raise ValueError('Invalid physical contrast or speed')
        return obj
