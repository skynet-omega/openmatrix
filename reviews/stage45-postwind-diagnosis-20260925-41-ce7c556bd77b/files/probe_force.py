"""No integration: compare a cold and position-consistent force mapping."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import mujoco as mj

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from state_io import read_state


def main():
    if not __debug__:raise RuntimeError('Original controller requires normal Python')
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    state=read_state(HERE/'inputs/restart_physical')
    model=mj.MjModel.from_binary_path(str(HERE/'inputs/body.mjb'))
    data=mj.MjData(model);mj.mj_setState(model,data,state['integration'],state['spec'])
    spec=importlib.util.spec_from_file_location('force_probe_contact',HERE/'vendor/work/stage2_contact_prosthesis_20260915/controller.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ctrl=mod.Controller.from_state(model,data,state['controller'])
    torque=np.array([0.,0.,-.004672697857153467])
    cold=np.zeros(model.nv)
    mj.mj_applyFT(model,data,np.zeros(3),torque,data.xpos[ctrl.thorax],ctrl.thorax,cold)
    scratch=mj.MjData(model);mj.mj_setState(model,scratch,state['integration'],state['spec'])
    mj.mj_fwdPosition(model,scratch)
    consistent=np.zeros(model.nv)
    mj.mj_applyFT(model,scratch,np.zeros(3),torque,scratch.xpos[ctrl.thorax],ctrl.thorax,consistent)
    with np.load(HERE/'FIRST_WIND_FORCE.npz',allow_pickle=False) as z:
        if not np.array_equal(cold,z['cold_force']) or not np.array_equal(consistent,z['consistent_force']):
            raise ValueError('Portable force mapping differs from original body loader probe')
    r=dict(cold_force_norm=float(np.linalg.norm(cold)),consistent_force_norm=float(np.linalg.norm(consistent)),
           exactly_matches_original_loader=True,physical_steps=0,neural_steps=0,
           note='Original application counted800 calls. The first had zero mapped force after cold restoration; do not label this a complete800-substep nonzero physical dose.')
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))


if __name__=='__main__':main()
