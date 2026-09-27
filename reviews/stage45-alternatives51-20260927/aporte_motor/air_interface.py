"""CPU engineering hypothesis for JO-C/E air input; no CNS or body imports.

The projection axes, half-response speed and amplitude are declared engineering
parameters, not fitted antennal mechanics or measured physiological responses.
No odor position, target bearing, controller or artificial odor-wind gate exists.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Parameters:
    half_speed_mm_s: float = 100.0
    max_drive_model_units: float = 80.0


class AirInterface:
    """Stateless extra drive for 335 annotated JO-C/E cells.

    Body axes are forward, left, up. Positive projected deflection is assigned
    to C (pull) and negative to E (push). That projection is an explicit proxy;
    only the qualitative C/E deflection distinction has a physiological source.
    Caller owns sampling, simulation clock, checkpoint and additive integration.
    """

    def __init__(self, anatomy_path=None, parameters=Parameters()):
        require(np.isfinite(parameters.half_speed_mm_s) and parameters.half_speed_mm_s > 0,
                'half_speed_mm_s must be positive and finite')
        require(np.isfinite(parameters.max_drive_model_units) and parameters.max_drive_model_units >= 0,
                'max_drive_model_units must be nonnegative and finite')
        path = anatomy_path or Path(__file__).with_name('JO_anatomy_arrays.npz')
        with np.load(path, allow_pickle=False) as a:
            self.ids = a['source_ids'].copy()
            self.rows = a['source_rows'].copy()
            self.sides = a['source_side'].copy()
            self.types = a['source_type'].copy()
        require(len(self.ids) == 335 and len(np.unique(self.ids)) == 335,
                'Expected 335 unique anatomical sources')
        require(self.rows.shape == self.ids.shape == self.sides.shape == self.types.shape,
                'Source arrays must align')
        require(len(np.unique(self.rows)) == 335 and np.all(self.rows >= 0), 'Invalid source rows')
        require(np.isin(self.sides, ['L', 'R']).all(), 'Use rootSide, not unknown somaSide')
        self.is_c = np.char.startswith(self.types, 'JO-C')
        require(np.all(self.is_c | np.char.startswith(self.types, 'JO-E')), 'Unexpected JO family')
        self.side_index = (self.sides == 'R').astype(np.int64)
        self.parameters = parameters
        self.projection = np.array([[1., 1., 0.], [1., -1., 0.]]) / np.sqrt(2.)
        for a in [self.ids, self.rows, self.sides, self.types, self.is_c,
                  self.side_index, self.projection]:
            a.setflags(write=False)

    def encode(self, *, air_velocity_world_mm_s, body_velocity_world_mm_s, body_to_world):
        air = np.asarray(air_velocity_world_mm_s, dtype=np.float64)
        body = np.asarray(body_velocity_world_mm_s, dtype=np.float64)
        rotation = np.asarray(body_to_world, dtype=np.float64)
        require(air.shape == body.shape == (3,) and np.isfinite(air).all()
                and np.isfinite(body).all(), 'Finite world velocity vectors required')
        require(rotation.shape == (3, 3) and np.isfinite(rotation).all(), 'Finite rotation required')
        require(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
                and abs(np.linalg.det(rotation) - 1) <= 1e-10, 'Proper rotation required')
        relative = rotation.T @ (air - body)
        require(np.isfinite(relative).all(), 'Relative velocity overflow')
        projection = self.projection @ relative
        selected = projection[self.side_index] * np.where(self.is_c, 1., -1.)
        speed = np.maximum(selected, 0.)
        normalized = speed / (speed + self.parameters.half_speed_mm_s)
        return self.parameters.max_drive_model_units * normalized

    def add_to_external_drive(self, base, **kinematics):
        """Return a copy. Preserve recurrent input and all other external ports.

        Caller must verify that base uses the canonical male_v10 row ordering;
        the anatomical archive and input hashes accompany this prototype.
        """
        base = np.asarray(base, dtype=np.float64)
        require(base.shape == (166700,) and np.isfinite(base).all(), 'Canonical finite drive required')
        result = base.copy()
        result[self.rows] += self.encode(**kinematics)
        require(np.isfinite(result).all(), 'Drive overflow')
        return result
