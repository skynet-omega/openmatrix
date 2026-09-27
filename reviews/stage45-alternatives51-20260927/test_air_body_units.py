"""Regression at the body-to-air boundary; no CNS or GPU import/execution."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import numpy as np
from pilot_owners_units_fixed import AirOwner
from aporte_motor.air_interface import AirInterface
H=Path(__file__).resolve().parent

def need(ok,msg):
 if not ok:raise ValueError(msg)

def check(root):
 api=AirInterface(root/'aporte_motor/JO_anatomy_arrays.npz');owner=object.__new__(AirOwner)
 checked=[]
 for v in [np.array([2.,-3.,.5]),np.array([-7.,1.,0.]),np.zeros(3)]:
  full=np.zeros(108);full[:3]=v;owner.run=SimpleNamespace(obj=SimpleNamespace(body=SimpleNamespace(data=SimpleNamespace(qvel=full))))
  vel=owner.velocity_mm_s();need(np.array_equal(vel,10*v),'Body cm/s to air mm/s')
  original=api.encode(air_velocity_world_mm_s=10*v,body_velocity_world_mm_s=v,body_to_world=np.eye(3))
  corrected=api.encode(air_velocity_world_mm_s=10*v,body_velocity_world_mm_s=vel,body_to_world=np.eye(3))
  need(np.array_equal(corrected,np.zeros(335)),'Comoving air/body must be zero')
  if np.any(v[:2]):need(original.max()>0,'Fixture must expose old integration bug')
  checked.append(dict(native_velocity=v.tolist(),corrected_drive_max=float(corrected.max()),old_drive_max=float(original.max())))
 # Verify recorded position conversion independently of the candidate helper.
 arms=json.loads((root/'PILOT_PLAN.json').read_text())['arms']
 for arm in arms:
  with np.load(root/arm/'traces.npz') as z:need(np.array_equal(z['position_mm'],10*z['qpos'][:,:3]),'Recorded geometric scale '+arm)
 freeze=json.loads((root/'PILOT_SOURCES.json').read_text());original_hash=hashlib.sha256((root/'pilot_owners.py').read_bytes()).hexdigest()
 need(original_hash==freeze['/home/daroch/AXIOMA_ASTRA/campanas/etapa45_alternativas_20260927_51/pilot_owners.py'],'Frozen source mutated')
 return dict(status='PASS_CPU_REGRESSION',cases=checked,original_preserved=True,geometric_scale_exact_all10=True,corrected_CNS_ms=0,scope='Tests the corrected body-velocity method used by the proposed consumer; no qualification or reproduction of corrected CNS dynamics')

if __name__=='__main__':
 r=check(H);(H/'AIR_BODY_UNITS_TEST.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
