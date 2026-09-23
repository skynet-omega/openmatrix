"""Reconstruct all density statistics from the portable real-state pair."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    original=json.loads((HERE/'RESULT.json').read_text());receipt=json.loads((HERE/'EXPORT.json').read_text())
    raw=(HERE/'state_pair.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt['files']['state_pair.npz']:raise ValueError('State pair changed')
    counts={}
    with np.load(HERE/'state_pair.npz',allow_pickle=False) as z:
        if list(z['time_ns'])!=original['source_clocks_ns']:raise ValueError('Different interval')
        for label,prefix in [('release_q','q'),('transmission_s','s')]:
            a,b=z[prefix+'0'],z[prefix+'1'];delta=np.abs(b-a)
            if a.shape!=b.shape or not np.isfinite(delta).all():raise ValueError('Invalid pair')
            counts[label]={'coordinates':len(a),'nonzero_initial':int(np.count_nonzero(a)),'nonzero_final':int(np.count_nonzero(b)),
              'changed_exactly':int(np.count_nonzero(delta)),'absolute_change_gt':{str(e):int(np.count_nonzero(delta>e)) for e in (1e-10,1e-8,1e-6,1e-4,1e-2)},
              'max_change':float(delta.max()),'percentiles_abs_change':np.percentile(delta,[0,25,50,75,90,99,100]).tolist()}
    if counts!=original['counts']:raise ValueError('Density report not reconstructed')
    print(json.dumps({'portable_real_pair_counts_exact':True,'GPU_executed':False}))
if __name__=='__main__':main()
