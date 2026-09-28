"""Describe actual body translation separately from a command with speed units."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
def compute(root):
 plan=json.loads((root/'PILOT_PLAN.json').read_text());data={};out={}
 for arm in plan['arms']:
  with np.load(root/arm/'traces.npz') as z:
   data[arm]=z['position_mm'].copy()
   out[arm]=dict(recorded_step1_to90_displacement_mm=(z['position_mm'][-1]-z['position_mm'][0]).tolist(),endpoint_linear_speed_mm_s=float(np.linalg.norm(10*z['qvel'][-1,:3])),mean_unclipped_forward_command_mm_s=float(z['forward_unclipped_mm_s'][50:90].mean()),command_applied_yaw_max=float(np.abs(z['command_yaw_rate_rad_s']).max()))
 for arm in out:
  parent='air0_odor'+arm[-1]
  out[arm]['final_position_difference_from_same_odor_parent_mm']=(data[arm][-1]-data[parent][-1]).tolist()
 return dict(arms=out,scope='Descriptive body positions and speed from recordings; commanded forward speed is not measured body displacement or autonomous locomotion. No stage criterion added after observation.')
if __name__=='__main__':
 r=compute(H);(H/'BODY_SUPPLEMENT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
