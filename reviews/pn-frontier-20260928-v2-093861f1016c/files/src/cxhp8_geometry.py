"""Observed proximal frame, not FlyBody Euler qpos or an electrical model.

Dataset axes: posterior/right/dorsal; Anipose chain basis flips y,z.
FlyBody thorax axes: anterior/left/dorsal, yielding posterior/left/ventral.
Landmarks A/B/C are coxa origin, CTr hinge, FeTi hinge. Their correspondence
is a morphological transfer; exact saved-angle reproduction does not establish
that these points duplicate optical tracking landmarks in another animal.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def proximal_angles(a,b,c,*,frame):
    a,b,c=[np.asarray(v,dtype=float) for v in (a,b,c)]
    if a.shape!=b.shape or a.shape!=c.shape or a.shape[-1]!=3 or not all(np.isfinite(v).all() for v in (a,b,c)):
        raise ValueError('Finite matching xyz landmarks required')
    z=b-a;n=np.linalg.norm(z,axis=-1,keepdims=True)
    if np.any(n<1e-12):raise ValueError('Coincident proximal landmarks')
    z=z/n;x=c-b;x=x-z*np.sum(x*z,axis=-1,keepdims=True);n=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(n<1e-12):raise ValueError('Collinear landmarks do not identify coxal rotation')
    x=x/n;y=np.cross(z,x);matrix=np.stack([x,y,z],axis=-1)
    frame=np.asarray(frame,dtype=float)
    if frame.shape!=(3,3) or not np.allclose(frame.T@frame,np.eye(3),atol=1e-12) or not np.isclose(np.linalg.det(frame),1):
        raise ValueError('Proper orthogonal frame required')
    angles=Rotation.from_matrix(frame@matrix).as_euler('zyx',degrees=True)
    # Sensor cares about circular rotation; dataset contains wrapped/unwrapped values.
    angles[...,0]%=360
    return angles # rotation, flexion, adduction (author column is named abduct)


def observed_flybody_angles(model,data):
    root=model.body('thorax').id;rot=data.xmat[root].reshape(3,3)
    points=[(data.xpos[model.body(n+'_T1_left').id]-data.xpos[root])@rot for n in ['coxa','femur','tibia']]
    return proximal_angles(*points,frame=np.diag([-1,1,-1]))
