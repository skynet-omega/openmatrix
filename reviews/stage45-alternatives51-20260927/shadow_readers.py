from pathlib import Path
import json,argparse
import numpy as np
H=Path(__file__).resolve().parent

def calculate(root):
 plan=json.loads((root/'READOUT_SHADOW_PLAN.json').read_text());pilot=json.loads((root/'PILOT_PLAN.json').read_text());ix=np.array(plan['pair_positions_in_neural_array']);sign=np.array(plan['signs'])
 with np.load(root/'anatomical_projection.npz') as z:A=z['A'].copy();ids=z['ids'].copy()
 values={}
 for arm in pilot['arms']:
  with np.load(root/arm/'neural_and_inputs.npz') as z:
   if not np.array_equal(z['ids'][ix].ravel(),ids):raise ValueError('Anatomical reader identity')
   q=z['q'][:,ix];dq=q-q[:10].mean(axis=0,keepdims=True);opp=(dq[:,:,0]-dq[:,:,1])*sign
  pop=dq.reshape(90,10)
  anatomical=(pop*A[:,1]).sum(axis=1)-(pop*A[:,0]).sum(axis=1)
  values[arm]={'five_pair_median':np.median(opp,axis=1),'five_pair_equal_mean':opp.mean(axis=1),'DNb05_DNb06_opponent':opp[:,2:4].mean(axis=1),'DNa02_DNg13_actions':opp[:,[1,4]].mean(axis=1),'fixed_anatomical_soma_projection':anatomical}
 report={}
 for reader in plan['readers']:
  means={a:float(v[reader][50:90].mean()) for a,v in values.items()};air={str(o):(means[f'airL_odor{o}']-means[f'airR_odor{o}'])/2 for o in [0,1]};gi=(means['G_odor1']-means['G_odor0'])-(means['I_odor1']-means['I_odor0'])
  report[reader]={'means':means,'air_L_minus_R_half':air,'G_minus_I_odor_interaction':gi,'applied_to_body':False}
 return {'readers':report,'scope':'Shadow readout of new native q; no trained weights, motor application, behavioral or physiological claim.'},values

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=H);p.add_argument('--out',type=Path);a=p.parse_args();r,_=calculate(a.root);(a.out or a.root/'SHADOW_READERS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
