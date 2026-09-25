"""Head-attached olfactory geometry for the next arena experiment.

Centers of existing antennal collision geoms are explicit sampling proxies,
not measured sensillum locations. Concentration is dimensionless. This module
does not generate motor commands or an artificial food-contact reward.
"""
import numpy as np
import mujoco as mj

class AntennalBoundary:
    def __init__(self,model):
        self.model=model;self.scratch=mj.MjData(model)
        self.geoms=np.array([model.geom('antenna_'+s+'_collision').id for s in ('left','right')])

    def sample(self,data,source_mm,sigma_mm):
        source=np.asarray(source_mm,float)
        if source.shape!=(2,) or not np.isfinite(source).all() or not np.isfinite(sigma_mm) or sigma_mm<=0:
            raise ValueError('Expected finite planar source and positive field width')
        self.scratch.qpos[:]=data.qpos
        mj.mj_kinematics(self.model,self.scratch)
        points=self.scratch.geom_xpos[self.geoms].copy()*10.
        concentration=np.exp(-np.sum((points[:,:2]-source)**2,axis=1)/(2.*sigma_mm**2))
        return dict(antennae_mm=points,concentration=concentration,canonical_ORN_drive=80.*concentration,
                    contact_reinforcement=0.,biological_transduction_calibrated=False)
