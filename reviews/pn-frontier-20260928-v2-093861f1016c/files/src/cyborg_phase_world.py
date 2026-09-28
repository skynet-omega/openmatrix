"""Restart a stimulus interval while preserving the predecessor's optical phase."""
import math

import numpy as np

from cyborg_visual_world import CyborgVisualWorld


class CyborgPhaseWorld(CyborgVisualWorld):
    SCHEMA = "matrix_cyborg_phase_world_v1"

    def __init__(self, start_ns, center_mm, frame, phase_offset_rad, condition="down", speed_deg_s=60.):
        self.phase_offset_rad = float(phase_offset_rad)
        super().__init__(start_ns, center_mm, frame, condition, speed_deg_s)

    @classmethod
    def from_predecessor(cls, previous, time_ns, condition="down", speed_deg_s=60.):
        if type(previous) is not CyborgVisualWorld:
            raise ValueError("Phase migration requires the original cyborg optical apparatus")
        previous._validate()
        return cls(time_ns, previous.center_mm, previous.frame,
                   previous.phase_rad(time_ns), condition, speed_deg_s)

    def _validate(self):
        super()._validate()
        if not math.isfinite(self.phase_offset_rad):
            raise ValueError("Finite inherited optical phase required")

    def phase_rad(self, time_ns):
        return self.phase_offset_rad + super().phase_rad(time_ns)

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or state.get("schema") != cls.SCHEMA:
            raise ValueError("Unknown phase-preserving optical world")
        obj = cls(state["start_ns"], state["center_mm"], state["frame"], state["phase_offset_rad"],
                  state["condition"], state["speed_deg_s"])
        expected = obj.state_dict()
        if set(state) != set(expected) or any(not np.array_equal(state[k], v) for k, v in expected.items()):
            raise ValueError("Incomplete or modified phase-preserving optical protocol")
        return obj
