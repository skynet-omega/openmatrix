"""Map a world torque at the current pose without touching integration state.

Only kinematic stages run on a private MjData. In particular, this must not run
the live forward dynamics, replace warm starts or consume retained contacts.
"""
import mujoco as mj
import numpy as np


class WorldTorque:
    def __init__(self, model, body_id):
        self.model = model
        self.body_id = int(body_id)
        if not 0 < self.body_id < model.nbody:
            raise ValueError('Torque needs a non-world body')
        self.scratch = mj.MjData(model)

    def map(self, data, torque):
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (3,) or not np.isfinite(torque).all():
            raise ValueError('Invalid world torque')
        self.scratch.qpos[:] = data.qpos
        # Mocap can determine bodies' world poses even with unchanged qpos.
        self.scratch.mocap_pos[:] = data.mocap_pos
        self.scratch.mocap_quat[:] = data.mocap_quat
        mj.mj_kinematics(self.model, self.scratch)
        mj.mj_comPos(self.model, self.scratch)
        extra = np.zeros(self.model.nv)
        mj.mj_applyFT(self.model, self.scratch, np.zeros(3), torque,
                      self.scratch.xpos[self.body_id], self.body_id, extra)
        if not np.isfinite(extra).all():
            raise FloatingPointError('Nonfinite generalized world torque')
        return extra
