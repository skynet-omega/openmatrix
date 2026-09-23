"""Export only the previously selected q/s coordinates for portable readback."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import scan
HERE=Path(__file__).resolve().parent
def main():
    start=time.perf_counter();r=json.loads((HERE/'RESULT.json').read_text());destination=HERE/'state_pair.npz'
    if destination.exists():raise FileExistsError(destination)
    for name,digest in r['sources'].items():
        if scan.sha(Path(name))!=digest:raise ValueError('Original audit source changed')
    x,m,t0=scan.read(scan.A);y,n,t1=scan.read(scan.B);N=166700;start_s=N+2*m
    np.savez_compressed(destination,q0=x[:N],q1=y[:N],s0=x[start_s:start_s+N],s1=y[start_s:start_s+N],time_ns=np.asarray([t0,t1],dtype=np.int64))
    elapsed=time.perf_counter()-start
    if elapsed+r['wall_s']>30:raise RuntimeError('Aggregate offline wall budget')
    receipt={'files':{'state_pair.npz':scan.sha(destination)},'source_audit_sha256':scan.sha(HERE/'RESULT.json'),
             'wall_s':elapsed,'fields':'Exact q/s slices only; not a complete organism checkpoint','script_sha256':scan.sha(Path(__file__))}
    (HERE/'EXPORT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
