"""Explicit functional neural-to-actuator port; no biological calibration claim.

Calibration consumes only an independent neutral neural record. Inference has
no access to stimulus labels, source position, pose or desired direction.
"""
from dataclasses import dataclass
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pair(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == 2 and len(x) > 0, 'Expected time by two channels')
    require(np.isfinite(x).all(), 'Non-finite neural state')
    require(((x >= 0) & (x <= 1)).all(), 'Release q outside declared domain')
    return x


@dataclass(frozen=True)
class CalibratedPort:
    neuron_ids: tuple
    offset_q: float
    scale_q: float
    ceiling_deg_s: float
    calibration_sha256: str
    input_units: str = 'dimensionless_release_q'
    output_units: str = 'deg_per_s'

    @classmethod
    def from_neutral_record(cls, values, neuron_ids, source_sha256, ceiling_deg_s=5.):
        x = pair(values)
        require(len(x) >= 2, 'Insufficient calibration samples')
        require(len(neuron_ids) == 2 and len(set(neuron_ids)) == 2, 'Distinct ordered neuron identities required')
        require(len(source_sha256) == 64 and all(c in '0123456789abcdef' for c in source_sha256), 'Calibration provenance required')
        require(np.isfinite(ceiling_deg_s) and ceiling_deg_s > 0, 'Invalid actuator envelope')
        d = x[:, 0] - x[:, 1]
        offset = float(d.mean())
        scale = float(np.sqrt(np.mean((d-offset)**2)))
        # Reject an unidentifiable scale instead of manufacturing a denominator.
        require(scale > 64*np.finfo(np.float64).eps, 'Neutral variation is numerically unresolved')
        return cls(tuple(neuron_ids), offset, scale, float(ceiling_deg_s), source_sha256)

    def command(self, values, neuron_ids):
        require(tuple(neuron_ids) == self.neuron_ids, 'Neuron identity/order changed')
        x = pair(values)
        return self.ceiling_deg_s*np.tanh(((x[:,0]-x[:,1])-self.offset_q)/self.scale_q)

