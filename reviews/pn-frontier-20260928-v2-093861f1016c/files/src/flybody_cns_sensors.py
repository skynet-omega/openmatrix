"""Reuse canonical sensory IDs; explicitly register them to FlyBody geometry.

The shared female ray map and FeCO transfer functions remain provisional.
Eye origins are area-weighted centers of the two red source eye surfaces,
not measured ommatidial origins. No detached rotor contributes to optical pose.
"""
import copy
import mujoco
import numpy as np
from anatomical_proprioception import AnatomicalProprioception, POLARITY_KEYS
from cyborg_rotor import CyborgEye
from retinal_world import aperture_rays
from native_motor_body import LEGS
from session_io import sha256


class FlyBodyProprioception(AnatomicalProprioception):
    def _initialize(self, brain, body, polarity, manifest, origin):
        # Frozen predecessor validation retained; only named geometry changes.
        if (not isinstance(polarity,dict) or set(polarity)!=POLARITY_KEYS
                or any(v not in ('flexion','extension') for v in polarity.values())):
            raise ValueError('Explicit FeCO polarity hypotheses required')
        self.brain, self.body = brain, body
        self.polarity = dict(polarity)
        self.manifest, self.manifest_origin = copy.deepcopy(manifest), copy.deepcopy(origin)
        if manifest.get('schema')!='matrix_anatomical_proprioception_ports_v1' or manifest.get('canonical_release')!='MaleCNS v1.0':
            raise ValueError('Expected canonical MaleCNS proprioception ports')
        ports=manifest['ports']; nerves={'ProLN':'F','MesoLN':'M','MetaLN':'H'}
        for p in ports:
            if (type(p['body_id']) is not int or p['body_id']<=0
                    or p['type'] not in ('SNpp39','SNpp41','SNpp50','SNpp51')
                    or p['manc_type']!=p['type'] or p['root_side'] not in ('L','R')
                    or p['entry_nerve'] not in nerves
                    or p['leg']!=p['root_side']+nerves[p['entry_nerve']]
                    or p['family']!=('hook' if p['type'] in ('SNpp39','SNpp41') else 'claw')):
                raise ValueError('Anatomically inconsistent FeCO port')
        self.node_ids=np.array([p['body_id'] for p in ports],dtype=np.int64)
        if not ports or np.any(np.diff(self.node_ids)<=0) or np.any(np.diff(brain.node_ids)<=0):
            raise ValueError('Canonical and sensory IDs must be uniquely sorted')
        self.indices=np.searchsorted(brain.node_ids,self.node_ids)
        if np.any(self.indices>=len(brain.node_ids)) or not np.array_equal(brain.node_ids[self.indices],self.node_ids):
            raise ValueError('Missing canonical FeCO neuron')
        self.leg_indices=np.array([LEGS.index(p['leg']) for p in ports])
        self.cell_types=tuple(p['type'] for p in ports)
        self.joint_ids=body.joints.tolist(); self.landmarks=body.landmarks.tolist()
        if np.any(body.model.jnt_type[self.joint_ids]!=mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Non-hinge proprioception port')
        for array in (self.indices,self.node_ids,self.leg_indices):array.flags.writeable=False
        self.identity=self._identity()

    def _identity(self):
        out=super()._identity()
        out.update(flybody_adapter_sha256=sha256(__file__),geometry_registration='FlyBody native tibial axes and femur/tibia/tarsus landmarks')
        return out


class FlyBodyEye(CyborgEye):
    def _initialize(self, brain, body, state):
        if set(state)!={'ids','sides','rays','manifest'}:raise ValueError('Incomplete inherited ray mapping')
        self.brain,self.body=brain,body;self.saved=copy.deepcopy(state)
        self.ids,self.sides,self.rays=state['ids'].copy(),state['sides'].copy(),state['rays'].copy()
        if self.ids.dtype!=np.int64 or np.any(np.diff(self.ids)<=0) or self.sides.shape!=self.ids.shape or not np.isin(self.sides,['L','R']).all():
            raise ValueError('Invalid optical IDs/sides')
        self.indices=np.searchsorted(brain.node_ids,self.ids)
        if np.any(self.indices>=brain.n_neurons) or not np.array_equal(brain.node_ids[self.indices],self.ids):raise ValueError('Missing canonical photoreceptor')
        self.apertures=aperture_rays(self.rays)
        model=body.model
        self.head_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'head')
        thorax=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'thorax')
        geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'head_red')
        if min(self.head_id,thorax,geom)<0:raise ValueError('Missing native head/eye geometry')
        neutral=mujoco.MjData(model);mujoco.mj_forward(model,neutral)
        head_R=neutral.xmat[self.head_id].reshape(3,3)
        thorax_R=neutral.xmat[thorax].reshape(3,3)
        self.optical_registration=head_R.T@thorax_R
        mesh=model.geom_dataid[geom];a,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        verts=model.mesh_vert[a:a+n].astype(float)
        fa,fn=model.mesh_faceadr[mesh],model.mesh_facenum[mesh];faces=model.mesh_face[fa:fa+fn]
        triangles=verts[faces]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
        centers=triangles.mean(axis=1)
        geom_R=neutral.geom_xmat[geom].reshape(3,3)
        world_verts=verts@geom_R.T+neutral.geom_xpos[geom]
        lateral=(world_verts-neutral.xpos[self.head_id])@thorax_R[:,1]
        if np.any((lateral[faces].min(axis=1)<0)&(lateral[faces].max(axis=1)>0)):
            raise ValueError('Source eye mesh crosses the assumed midplane')
        self.eye_geometry={}
        for side,sign in [('L',1),('R',-1)]:
            mask=lateral[faces].mean(axis=1)*sign>0
            if areas[mask].sum()<=0:raise ValueError('Missing eye hemisphere')
            local=np.average(centers[mask],axis=0,weights=areas[mask])
            self.eye_geometry[side]=(geom,local)

    def pose(self):
        # The retained rotor reference is historical compatibility only.
        body=self.body; body.observe()
        rotation=body.scratch.xmat[self.head_id].reshape(3,3)@self.optical_registration
        centers={side:(body.scratch.geom_xpos[geom]+body.scratch.geom_xmat[geom].reshape(3,3)@local)*10.
                 for side,(geom,local) in self.eye_geometry.items()}
        return rotation,centers
