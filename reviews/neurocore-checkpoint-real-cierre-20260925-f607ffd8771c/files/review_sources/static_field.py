"""Fixed spatial stimulus; the field never reads neural outputs or motor commands."""
import numpy as np
class StaticLateralField:
    def __init__(self,base,world,arm,antennae_mm,*,width_mm=30.,speed_mm_s=15.,onset_ms=10.):
        if arm not in ('odor_left','odor_right','uniform','sham'):raise ValueError(arm)
        self.base,self.world,self.arm=base,world,arm
        xy=np.asarray(antennae_mm,dtype=float)[:,:2]
        self.center=xy.mean(0);sep=float(np.linalg.norm(xy[0]-xy[1]));assert sep>0
        self.axis=(xy[0]-xy[1])/sep
        if arm=='odor_right':self.axis=-self.axis
        self.origin_ns=int(world.time_ns);self.onset_ms=float(onset_ms)
    def evaluate(self,antennae_mm,time_ns):
        elapsed=(int(time_ns)-self.origin_ns)*1e-6
        if elapsed<0:raise ValueError('Clock precedes field')
        if self.arm=='sham' or elapsed+1e-9<self.onset_ms:return np.zeros(2)
        if self.arm=='uniform':return np.ones(2)
        coordinates=(np.asarray(antennae_mm)[:,:2]-self.center)@self.axis
        return (coordinates>0).astype(float)
    def sample(self,data,source_mm,sigma_mm):
        out=self.base.sample(data,source_mm,sigma_mm)
        c=self.evaluate(out['antennae_mm'],self.world.time_ns)
        return dict(out,concentration=c,canonical_ORN_drive=80.*c,
                    fictive_static_lateral=True,biological_transduction_calibrated=False)
    def metadata(self):
        return dict(arm=self.arm,installed_ns=self.origin_ns,center_mm=self.center.tolist(),
                    odor_axis=self.axis.tolist(),first_ON_after_install_ms=self.onset_ms,
                    live_geometry=True,stationary_world_field=True,calibrated_optogenetic_drive=False)
