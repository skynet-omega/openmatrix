"""Replay the negative-input certificate from its exact ten49 raw captures.

Requires the full evidence capsule. The compact50 capsule does not include
these prior49 operands; this check fails clearly when they are absent.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
def verify():
    start=time.process_time();c=json.loads((H/'SUBTHRESHOLD49_CERTIFICATE.json').read_text());bound=[]
    for name,digest in c['sources'].items():
        source=Path(name)
        p=ROOT/(source.relative_to('/home/daroch/AXIOMA_ASTRA') if source.is_absolute() else source)
        if not p.is_file():raise ValueError('This verification requires the full capsule49 operand files')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Changed49 capture')
        with np.load(p,allow_pickle=False) as z:
            f=z['records'][:,:384].reshape(-1,4,6,16)[:,:,:2,:]
            net=f[...,1].astype(np.float32)+f[...,4].astype(np.float32)
            if not np.isfinite(net).all() or not np.all(net<0):raise ValueError('Nonnegative frozen input')
            bound.append(net.max(axis=(0,1)))
    if np.max(bound,axis=0).tolist()!=c['max_net_plus_drive']:raise ValueError('Certificate differs')
    return dict(frozen49_input_certificate_verified=True,CPU_s=time.process_time()-start,new_neural_ms=0)
if __name__=='__main__':print(json.dumps(verify()))
