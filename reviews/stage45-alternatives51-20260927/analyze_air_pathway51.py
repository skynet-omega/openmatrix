"""Observed pathway response and total-drive confound, without post-hoc fitting."""
from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent

def compute(root):
 out={}
 for odor in [0,1]:
  with np.load(root/f'air0_odor{odor}/neural_and_inputs.npz') as z:base=z['q'].copy()
  out[str(odor)]={}
  for direction in ['L','R']:
   with np.load(root/f'air{direction}_odor{odor}/neural_and_inputs.npz') as z:
    q=z['q'];change=np.abs(q[:,12:]-base[:,12:]).max(axis=0)
    out[str(odor)][direction]={'relay_ids':z['ids'][12:].tolist(),'relay_delta_max_abs':change.tolist(),'relay_response_present':bool(np.any(change>0)),'mean_total_JO_drive':float(z['JO_drive'][50:].sum(axis=1).mean()),'nonzero_JO_at51':int((z['JO_drive'][50]>0).sum())}
 time_course={}
 for odor in [0,1]:
  with np.load(root/f'airL_odor{odor}/traces.npz') as l,np.load(root/f'airR_odor{odor}/traces.npz') as r:
   h=np.rad2deg(l['neural_yaw_unapplied_rad_s']-r['neural_yaw_unapplied_rad_s'])[50:90]/2
   time_course[str(odor)]=dict(min_half_yaw_deg_s=float(h.min()),max_half_yaw_deg_s=float(h.max()),positive_samples=int((h>0).sum()),negative_samples=int((h<0).sum()),sign_changes=int(np.count_nonzero(h[1:]*h[:-1]<0)))
 return dict(observations=out,within_prespecified_window=time_course,scope='Four anatomically selected AMMC/WED candidate relays fixed before run; no claim of experimentally identified APN2/APN3 function.',limitation='Opposite world fields change both which neurons receive drive and total JO drive, because anatomical C/E left/right counts differ. Directional dependence is demonstrated for this interface; selective direction encoding at matched total input is not established. Raw paired yaw changes sign within the analysis window, so its mean is not stable directional steering.')

if __name__=='__main__':
 r=compute(H);(H/'AIR_PATHWAY_SUPPLEMENT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
