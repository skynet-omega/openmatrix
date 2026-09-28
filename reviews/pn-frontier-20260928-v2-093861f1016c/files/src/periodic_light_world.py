"""Repeatable physical light stimulus; no action or error signal enters the brain."""
import copy
import numpy as np
from retinal_world import LuminousWorld


class PeriodicLightWorld:
    SCHEMA='matrix_periodic_light_world_v1'

    def __init__(self,reference,start_ns):
        self.reference=LuminousWorld.from_state(reference.state_dict())
        self.start_ns=int(start_ns)
        self.period_ns=1_000_000_000
        self.yaw_amplitude_rad=float(np.deg2rad(15.))
        self.pitch_amplitude_rad=float(np.deg2rad(10.))

    def rotation(self,time_ns):
        if time_ns<self.start_ns:
            raise ValueError('Stimulus before its declared beginning')
        phase=2*np.pi*((time_ns-self.start_ns)%self.period_ns)/self.period_ns
        yaw=self.yaw_amplitude_rad*np.sin(phase)
        pitch=self.pitch_amplitude_rad*np.sin(2*phase)
        c,s=np.cos(yaw),np.sin(yaw)
        cy,sy=np.cos(pitch),np.sin(pitch)
        return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@np.array([[cy,0.,sy],[0.,1.,0.],[-sy,0.,cy]])

    def luminance(self,origins,rays,time_ns):
        rotation=self.rotation(time_ns)
        center=self.reference.center_mm
        # A rotating sphere texture expressed in the sphere's coordinates.
        # Keep the inherited static texture phase fixed, without resetting it.
        origins=center+(np.asarray(origins)-center)@rotation
        rays=np.asarray(rays)@rotation
        return self.reference.luminance(origins,rays,self.start_ns)

    def state_dict(self):
        return dict(schema=self.SCHEMA,reference=self.reference.state_dict(),start_ns=self.start_ns,
            period_ns=self.period_ns,yaw_amplitude_rad=self.yaw_amplitude_rad,pitch_amplitude_rad=self.pitch_amplitude_rad)

    @classmethod
    def from_state(cls,state):
        if state.get('schema')!=cls.SCHEMA:
            raise ValueError('Unknown periodic world')
        obj=cls(LuminousWorld.from_state(state['reference']),state['start_ns'])
        if set(state)!=set(obj.state_dict()):
            raise ValueError('Incomplete periodic world')
        for key in ('period_ns','yaw_amplitude_rad','pitch_amplitude_rad'):
            setattr(obj,key,copy.deepcopy(state[key]))
        if type(obj.period_ns) is not int or obj.period_ns<=0 or not np.isfinite([obj.yaw_amplitude_rad,obj.pitch_amplitude_rad]).all():
            raise ValueError('Invalid periodic stimulus')
        return obj
