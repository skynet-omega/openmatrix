"""Reconstruct the historical mechanical-support criterion from observed impulses."""
from pathlib import Path
import json,sys
import numpy as np

def check(folder,criteria):
    with np.load(folder/'SUPPORT.npz',allow_pickle=False) as z:
        t=z['duration_s'];imp=z['vertical_impulse_Ns'];mg=float(z['mg_N'])
        if len(t)!=441 or imp.shape!=(441,3) or not np.isfinite(imp).all() or not (mg>0):raise ValueError('Incomplete support')
        if not np.allclose(t,np.arange(441)*.001,rtol=0,atol=1e-9):raise ValueError('Mechanical clock mismatch')
        support=(imp[10:]-imp[:-10])/(.010*mg)
    with np.load(folder/'traces.npz',allow_pickle=False) as z:upright=float(np.min(z['upright']))
    result={'min_leg_10ms_weight_fraction':float(np.min(support[:,1])),
            'max_abdominal_10ms_weight_fraction':float(np.max(support[:,0])),
            'upright_min':upright,'mg_N':mg,'scope':'Integrated completed MuJoCo contact forces; effective synthetic contact-roller body, not biological gait'}
    result['passed']=(result['min_leg_10ms_weight_fraction']>=criteria['min_leg_10ms_weight_fraction'] and
                      result['max_abdominal_10ms_weight_fraction']<=criteria['max_abdominal_10ms_weight_fraction'] and
                      upright>=criteria['upright_min'])
    return result
if __name__=='__main__':
    folder=Path(sys.argv[1]);criteria=json.loads((Path(__file__).parent/'PLAN.json').read_text())['criteria']
    result=check(folder,criteria);(folder/'SUPPORT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));raise SystemExit(0 if result['passed'] else 2)
