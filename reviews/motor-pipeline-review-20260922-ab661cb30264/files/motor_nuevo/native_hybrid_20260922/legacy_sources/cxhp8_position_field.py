"""Explicit calcium-informed position index; no firing-rate or current calibration."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from session_io import sha256
from cxhp8_geometry import observed_flybody_angles

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/cxhp8_position_20260914/prior.json'
PRIOR_SHA256='614fcb33c9ac9258b6be86417fe448e12b4904dd1c97003d842a7b61f622e1a5'
IDS=[804975,813083,816234,860788,904779,908177,1110706007]
POLICY='median_animal_calcium_field_instant_current80_v1'
AMPLITUDE=80.


class CxHP8PositionField:
    def __init__(self,model):
        import mujoco
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed CxHP8 empirical index prior')
        self.prior=json.loads(PRIOR.read_text());p=self.prior
        if p['sensor_ids']!=IDS or p['policy']!=POLICY or p['drive_amplitude_model_units']!=AMPLITUDE or p['biological_electrical_calibration'] is not False:
            raise ValueError('Invalid CxHP8 index contract')
        field=np.array(p['index'],dtype=float);valid=np.isfinite(field)
        if np.any((field[valid]<0)|(field[valid]>1)):raise ValueError('Index outside declared range')
        self.interpolator=RegularGridInterpolator((p['rot_centers_deg'],p['adduct_centers_deg']),field,bounds_error=True)
        self.model=model;self.view=mujoco.MjData(model)

    def encode(self,angles):
        angles=np.asarray(angles,dtype=float)
        if angles.shape!=(3,) or not np.isfinite(angles).all():raise ValueError('Finite rotation/flexion/adduction required')
        value=float(self.interpolator(angles[[0,2]][None,:])[0])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('CxHP8 position outside empirical interpolation support')
        return value

    def sample(self,body):
        import mujoco
        if body.model is not self.model:raise ValueError('Different physical model')
        # Observation must not mutate integration caches/warmstarts in the live body.
        self.view.qpos[:]=body.data.qpos;mujoco.mj_forward(self.model,self.view)
        angles=observed_flybody_angles(self.model,self.view)
        return dict(angles_deg=angles.tolist(),index=self.encode(angles))
