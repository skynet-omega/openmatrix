"""CPU: hot versus restored body, 40 corrected physical substeps, no CNS."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import mujoco as mj
import numpy as np
from wind_mapping import WorldTorque

HERE = Path(__file__).resolve().parent
DONOR = HERE.parents[1] / 'campanas/etapa45_postwind_diagnosis_20260925_41'
sys.path.insert(0, str(DONOR))
from state_io import read_state


def need(ok, why):
    if not ok:
        raise ValueError(why)


def main():
    started = time.monotonic()
    need(__debug__, 'Legacy assertions require normal Python')
    state = read_state(DONOR / 'inputs/restart_physical')
    model = mj.MjModel.from_binary_path(str(DONOR / 'inputs/body.mjb'))
    path = DONOR / 'vendor/work/stage2_contact_prosthesis_20260915/controller.py'
    spec = importlib.util.spec_from_file_location('wind_contact_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hot = mj.MjData(model)
    mj.mj_setState(model, hot, state['integration'], state['spec'])
    control = module.Controller.from_state(model, hot, state['controller'])
    def integration(data):
        result = np.empty(mj.mj_stateSize(model, state['spec']))
        mj.mj_getState(model, data, result, state['spec'])
        return result
    def step(data, ctrl, mapper, wind):
        ctrl.set_command(forward_mm_s=.2, yaw_rate_rad_s=np.deg2rad(5.))
        force = ctrl.torque(data, model.opt.timestep)
        extra = mapper.map(data, np.array([0., 0., wind]))
        data.ctrl[:] = 0.
        data.qfrc_applied[:] = force + extra
        try:
            mj.mj_step(model, data)
        finally:
            data.qfrc_applied[:] = 0.
            data.ctrl[:] = 0.
        need(not np.any(data.warning.number), 'MuJoCo warning')
        return force, extra
    mapper = WorldTorque(model, control.thorax)
    # Produce an actual hot witness by integrating, not by restoring twice.
    for _ in range(40):
        step(hot, control, mapper, 0.)
    snapshot = integration(hot)
    ctrl_state = control.state_dict()
    cold = mj.MjData(model)
    mj.mj_setState(model, cold, snapshot, state['spec'])
    restored = module.Controller.from_state(model, cold, ctrl_state)
    cold_mapper = WorldTorque(model, restored.thorax)
    torque = np.array([0., 0., -.004672697857153467])
    f_hot = mapper.map(hot, torque)
    f_cold = cold_mapper.map(cold, torque)
    need(np.array_equal(snapshot, integration(hot)), 'Hot integration mutated by mapping')
    need(np.array_equal(snapshot, integration(cold)), 'Cold integration mutated by mapping')
    need(np.array_equal(f_hot, f_cold) and np.linalg.norm(f_hot) > 0,
         'Corrected first torque differs or vanished')
    independent = mj.MjData(model)
    mj.mj_setState(model, independent, snapshot, state['spec'])
    mj.mj_fwdPosition(model, independent)
    reference_force = np.zeros(model.nv)
    mj.mj_applyFT(model, independent, np.zeros(3), torque,
                 independent.xpos[control.thorax], control.thorax, reference_force)
    need(np.array_equal(f_hot, reference_force), 'Minimal mapping differs from full position pipeline')
    legacy_hot = np.zeros(model.nv)
    legacy_cold = np.zeros(model.nv)
    for data, dest in ((hot, legacy_hot), (cold, legacy_cold)):
        mj.mj_applyFT(model, data, np.zeros(3), torque, data.xpos[control.thorax], control.thorax, dest)
    norms = []
    for _ in range(40):
        a, b = step(hot, control, mapper, torque[2])
        c, d = step(cold, restored, cold_mapper, torque[2])
        for label, x, y in (('traction', a, c), ('wind', b, d),
                             ('integration', integration(hot), integration(cold)),
                             ('contacts', control.last_contact_active, restored.last_contact_active)):
            need(np.array_equal(x, y), 'Hot/restored differs: ' + label)
        norms.append(float(np.linalg.norm(b)))
    result = dict(status='PASS', hot_witness_steps=40, compared_physical_steps=40,
                  neural_steps=0, corrected_torque_norm=float(np.linalg.norm(f_hot)),
                  legacy_hot_difference=float(np.max(np.abs(legacy_hot-f_hot))),
                  legacy_cold_norm=float(np.linalg.norm(legacy_cold)),
                  all_40_wind_forces_nonzero=all(x > 0 for x in norms),
                  full_integration_contacts_forces_exact=True,
                  integration_unchanged_by_mapping=True, mujoco=mj.__version__,
                  minimal_equals_independent_full_position=True,
                  wall_s=time.monotonic()-started,
                  scope='Body fixture only; does not certify an organism restart')
    path = HERE / 'reviews/WIND_CPU_V2.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
