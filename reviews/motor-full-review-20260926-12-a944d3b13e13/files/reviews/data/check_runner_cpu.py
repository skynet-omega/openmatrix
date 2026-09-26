"""Test actual MotorWind class methods with a pure CPU fake body.

This checks scheduling and motor memory only. No MuJoCo integration, CNS,
CuPy, GPU allocation, or organism loading occurs. Geometry was checked by the
parent's existing WIND_CPU_V2 fixture and is not claimed by this check.
"""
import ast
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace as NS
import numpy as np

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30s CPU budget')))
signal.alarm(30)
started=time.monotonic()
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT.parents[1]/'campanas/etapa45_navigation_wind_20260925_40/motor_wind.py'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=HERE/'RUNNER_CPU.json')
args=parser.parse_args()

def need(ok, why):
    if not ok: raise ValueError(why)

def cls(path, namespace):
    tree=ast.parse(path.read_text(),filename=str(path))
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MotorWind']
    need(len(classes)==1,'Unique MotorWind class')
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['MotorWind']

base=cls(OLD,dict(np=np,math=math,mj=None))

class FakeMapper:
    def __init__(self,*_): pass
    def map(self,data,torque):
        out=np.zeros(6);out[3:]=torque
        return out

fixed=cls(ROOT/'motor_wind_fixed.py',dict(np=np,WorldTorque=FakeMapper,_module=NS(MotorWind=base)))
body=NS(dt=25e-6,model=NS(),data=NS(time=0.,qpos=np.zeros(7)))
controller=NS(active=True,thorax=1,torque=lambda data,dt:np.zeros(6))
obj=NS(command_mode='neural',controller=controller,body=body,
       last_dn=np.zeros(4),dn_baseline=np.zeros(4),requested=np.zeros(2))
applied=[]
calls=0
def advance(torque,nsteps):
    global calls
    need(obj.command_mode=='device','Body receives device command')
    need(nsteps==1,'One physical call')
    force=controller.torque(body.data,body.dt)
    need(np.isfinite(force).all(),'Finite fake generalized force')
    calls+=1;body.data.time=calls*body.dt;body.data.qpos[0]=calls
    if calls%40==1:applied.append(obj.requested.copy())
    return force
obj.body_advance=advance
motor=fixed(obj)
filter_reference=0.;checkpoints={}
for k in range(1,2001):
    # Nonzero continuing command crosses the artificial wind window unchanged.
    obj.last_dn[:]=[.01,.02,.001,-.001]
    raw=math.tanh(250*(obj.last_dn[2]-obj.last_dn[3]))*motor.OUTPUT_RAD_S
    filter_reference+=motor.alpha*(raw-filter_reference)
    for _ in range(40):body.advance(np.zeros(6))
    expected=max(0,min(k,1020)-1001+1)*40
    need(motor.trial_step==k and motor.body_calls==40*k,'Common millisecond grid')
    need(motor.wind_substeps==expected,'Exact scheduled pulse dose')
    need(motor.filtered==filter_reference,'Filter continuous across window')
    need(not motor.wind_active and obj.command_mode=='neural','Outer mode restored')
    if k in (20,100,1000,1001,1020,1021,2000):
        checkpoints[str(k)]=dict(body_calls=motor.body_calls,wind_substeps=motor.wind_substeps,
                                filtered=motor.filtered,applied=motor.applied)
need(len(motor.wind_log)==800,'Exactly 800 logged physical applications')
need(motor.wind_log[0]['body_call']==40001 and motor.wind_log[-1]['body_call']==40800,'Exact wind call interval')
need(motor.wind_log[0]['body_time_s']==1. and abs(motor.wind_log[-1]['body_time_s']-1.019975)<1e-14,'Left endpoint timing')
need(all(x['qpos'][0]==x['body_call']-1 for x in motor.wind_log),'Pose sampled before each physical step')
audit=motor.audit()
need(audit['nonzero_substeps']==800,'Nonzero telemetry')
need(json.loads(json.dumps(audit,allow_nan=False))['nonzero_substeps']==800,'Wind audit is JSON serializable')
sources=[ROOT/name for name in ['run_trial.py','PLAN.json','full_snapshot.py','source_inventory.py','freeze_sources.py','wind_mapping.py','motor_wind_fixed.py','restore_prepared.py','state_compare.py']]+[ROOT/'pn/install_pn.py',OLD]
for p in sources:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
result=dict(status='PASS',scope=__doc__,wall_s=time.monotonic()-started,wall_budget_s=30,
    mock_body_calls=calls,wind_calls=len(motor.wind_log),first_wind_call=40001,last_wind_call=40800,
    first_wind_left_s=motor.wind_log[0]['body_time_s'],last_wind_left_s=motor.wind_log[-1]['body_time_s'],
    filters_continuous=True,wind_audit_json_serializable=True,checkpoints=checkpoints,
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    organism_loaded=False,gpu_loaded=False,physical_dynamics_simulated=False)
with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:result[k] for k in ['status','wall_s','mock_body_calls','wind_calls']}))
