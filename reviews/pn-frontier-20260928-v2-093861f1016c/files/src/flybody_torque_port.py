"""Separate FlyBody physical candidate with explicit SI torque ports.

Loads only pinned XML/mesh assets through MuJoCo, not FlyBody task policies.
All source actuators (including zero-command position biases and adhesion)
are replaced by six direct tibial motors. Source passive joints remain.
This is a new physical preparation, not a migration of a FlyGym checkpoint.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from contractile_tibia import ContractileTibia
from native_motor_body import LEGS, MAX_TORQUE_NATIVE
from passive_body import reject_external_callbacks
from session_io import sha256

ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / 'data/model_assets_20260909'
ASSETS = ASSETS_ROOT / 'flybody/source/flybody/fruitfly/assets'
FLYGYM_TORQUE_TO_NM = 1e-9  # gram * millimetre**2 / second**2
FLYBODY_TORQUE_TO_NM = 1e-7  # gram * centimetre**2 / second**2
PORTS = tuple(f'tibia_T{segment}_{side}' for segment in (1, 2, 3) for side in ('left', 'right'))


def _pinned_sources():
    manifest = json.loads((ASSETS_ROOT / 'manifest.json').read_text())
    package = next(p for p in manifest['packages']
                   if p.get('source_url') == 'https://github.com/TuragaLab/flybody'
                   and 'files' in p)
    files = {f['path']: f['sha256'] for f in package['files']
             if '/fruitfly/assets/' in f['path']}
    if not files:
        raise ValueError('Missing pinned FlyBody assets')
    for name, digest in files.items():
        if sha256(ASSETS_ROOT / name) != digest:
            raise ValueError('Changed FlyBody asset: ' + name)
    return dict(revision=package['revision'], files=files)


class FlyBodyTorquePort:
    def __init__(self):
        reject_external_callbacks()
        source = _pinned_sources()
        root = ET.parse(ASSETS / 'fruitfly.xml').getroot()
        root.find('compiler').set('meshdir', str(ASSETS))
        floor = ET.parse(ASSETS / 'floor.xml').getroot()
        for child in floor.find('asset'):
            root.find('asset').append(copy.deepcopy(child))
        for child in floor.find('worldbody'):
            root.find('worldbody').append(copy.deepcopy(child))
        actuators = root.find('actuator')
        removed = [dict(tag=a.tag, name=a.get('name')) for a in actuators]
        for actuator in list(actuators):
            actuators.remove(actuator)
        limit = MAX_TORQUE_NATIVE * FLYGYM_TORQUE_TO_NM / FLYBODY_TORQUE_TO_NM
        for name in PORTS:
            ET.SubElement(actuators, 'motor', name='matrix_' + name, joint=name,
                          gear='1', ctrllimited='true', ctrlrange=f'{-limit} {limit}',
                          forcelimited='true', forcerange=f'{-limit} {limit}')
        # Same explicit 25us comparison step as the current MATRIX body.
        root.find('option').set('timestep', '0.000025')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.dt = float(self.model.opt.timestep)
        self.steps = 0
        self.joints = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in PORTS])
        self.landmarks = np.array([[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
            name.replace('tibia', segment)) for segment in ('femur', 'tibia', 'tarsus')] for name in PORTS])
        if min(self.joints) < 0 or self.landmarks.min() < 0:
            raise ValueError('Missing FlyBody tibial geometry')
        if (self.model.nu != 6 or self.model.na or np.any(self.model.actuator_biastype)
                or np.any(self.model.actuator_biasprm) or np.any(self.model.actuator_dyntype)
                or not np.array_equal(self.model.actuator_gainprm[:, 0], np.ones(6))):
            raise ValueError('Ports must be memoryless unit-gain unbiased motors')
        self.sign_calibration_qpos = self.data.qpos.copy()
        self.flexion_sign = -self.observe()['interior_derivative_sign']
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=buffer)
        self.identity = dict(source=source, source_sha256=sha256(__file__),
            model_sha256=hashlib.sha256(buffer).hexdigest(), mujoco_version=mujoco.__version__,
            removed_actuators=removed, leg_order=list(LEGS), joint_ports=list(PORTS),
            torque_unit_Nm=FLYBODY_TORQUE_TO_NM, length_unit_m=.01, mass_unit_kg=.001,
            initialization='Source qpos0 and floor; new body, no inherited pose or warmstart migration',
            passive_stiffness_and_damping_retained=True, learned_controller_loaded=False,
            actuator_adhesion=False, neural_closed_loop=False, biological_validation=False)

    def integration_state(self):
        state = np.empty(mujoco.mj_stateSize(self.model, self.spec))
        mujoco.mj_getState(self.model, self.data, state, self.spec)
        return state

    def observe(self):
        reject_external_callbacks()
        mujoco.mj_setState(self.model, self.scratch, self.integration_state(), self.spec)
        mujoco.mj_forward(self.model, self.scratch)
        if np.any(self.scratch.warning.number):
            raise FloatingPointError('FlyBody observation warning')
        angles, signs = [], []
        for joint, (femur, tibia, tarsus) in zip(self.joints, self.landmarks):
            axis = self.scratch.xaxis[joint]
            u = self.scratch.xpos[femur] - self.scratch.xpos[tibia]
            v = self.scratch.xpos[tarsus] - self.scratch.xpos[tibia]
            u, v = u - u.dot(axis)*axis, v - v.dot(axis)*axis
            norm = np.linalg.norm(u)*np.linalg.norm(v)
            sine = axis.dot(np.cross(u, v))/norm
            if not np.isfinite(sine) or norm <= 0 or abs(sine) < 1e-10:
                raise ValueError('Undefined FlyBody interior-angle derivative')
            angles.append(np.arctan2(abs(sine), np.clip(u.dot(v)/norm, -1., 1.)))
            signs.append(np.sign(sine))
        signs = np.asarray(signs)
        return dict(angles_rad=np.asarray(angles), interior_derivative_sign=signs,
            angular_velocity_rad_s=signs*self.data.qvel[self.model.jnt_dofadr[self.joints]],
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
            contact_count=int(self.scratch.ncon))

    def advance_joint_torque_si(self, torque_nm, nsteps=1):
        torque = np.asarray(torque_nm, dtype=float)
        limit = MAX_TORQUE_NATIVE * FLYGYM_TORQUE_TO_NM
        if (torque.shape != (6,) or not np.isfinite(torque).all() or
                np.any(abs(torque) > limit*(1+1e-12)) or type(nsteps) is not int or nsteps < 0):
            raise ValueError('Invalid FlyBody torque port input')
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError('Undeclared external force')
        for _ in range(nsteps):
            reject_external_callbacks()
            self.data.ctrl[:] = torque/FLYBODY_TORQUE_TO_NM
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            if (np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all()
                    or not np.isfinite(self.data.qvel).all()
                    or abs(self.data.time-self.steps*self.dt) > 1e-10):
                raise FloatingPointError('FlyBody integration failed without rescue')

    def state_dict(self):
        return dict(schema='flybody_torque_port_v1', identity=copy.deepcopy(self.identity),
                    steps=self.steps, integration=self.integration_state())

    @classmethod
    def from_state(cls, state):
        obj = cls()
        if (set(state) != {'schema', 'identity', 'steps', 'integration'} or
                state['schema'] != 'flybody_torque_port_v1' or state['identity'] != obj.identity
                or type(state['steps']) is not int or state['steps'] < 0
                or state['integration'].shape != obj.integration_state().shape
                or not np.isfinite(state['integration']).all()):
            raise ValueError('Invalid/different FlyBody physical state')
        mujoco.mj_setState(obj.model, obj.data, state['integration'], obj.spec)
        obj.steps = state['steps']
        if abs(obj.data.time-obj.steps*obj.dt) > 1e-10:
            raise ValueError('FlyBody physical clock mismatch')
        return obj


class FlyBodyEffectiveTibia(ContractileTibia):
    """Reuse the declared 12-element approximation in mm/g/s muscle units.

    Only joint names/reference geometry differ. Its output MUST be multiplied
    by FLYGYM_TORQUE_TO_NM before entering the SI physical port. New activation
    and reference lengths are a new preparation, not native muscle calibration.
    """
    def _geometry(self):
        self.qadr = self.body.model.jnt_qposadr[self.body.joints]
        self.vadr = self.body.model.jnt_dofadr[self.body.joints]
        self.reference_q = self.body.sign_calibration_qpos[self.qadr].copy()
