"""Supplemental law/clock checks requested by the nonauthor review.

No change to acquisition, J, windows, or gates. Positive tanhf uses the same
documented 2-ULP library comparison as49; sums/margins/parameters stay exact.
"""
import math
from pathlib import Path
import numpy as np
from analyze50 import HERE,ARMS,need,read_npz,sha

REFERENCE_SHA256='9414489fb9f8a238d7ac2ec9a6ee318597be6612e467227a415bd3d0625158d6'

def check_law(d,reference=None):
    r=d['records'];n=len(r);f=r[:,:128].reshape(n,4,2,16)
    need(np.isfinite(r).all(),'Nonfinite RHS')
    need(np.all(f[...,9]>0),'Nonpositive tau')
    need(np.all(f[...,6]>0),'Nonpositive gain')
    rate=(np.float32(1)/f[...,9].astype(np.float32)).astype(np.float64)
    need(np.array_equal(f[...,8],rate) and np.array_equal(f[...,12],rate),'Rate/tau mismatch')
    need(np.array_equal(f[...,15],np.broadcast_to(np.array([0.,.5,.75,0.])[None,:,None],(n,4,2))),'RHS stage fraction')
    expected=r[:,128,None,None]+f[...,15]*r[:,129,None,None]
    expected[:,3,:]=np.nextafter(r[:,130],r[:,128])[:,None]
    need(np.array_equal(f[...,14],expected),'RHS clock mismatch')
    margin=(f[...,1].astype(np.float32)+f[...,4].astype(np.float32))-f[...,5].astype(np.float32)
    need(np.array_equal(f[...,10],margin),'Exact FP32 margin')
    argument=f[...,6].astype(np.float32)*margin
    target=f[...,7].astype(np.float32)
    need(np.array_equal(target.astype(np.float64),f[...,7]),'Target FP32 representation')
    need(np.array_equal(f[...,7],f[...,11]),'Final target overwritten')
    need(np.all(f[...,7][argument<=0]==0),'Negative argument did not rectify')
    need(np.all((target>=0)&(target<=1)),'Target out of range')
    desired=np.array([max(0.,math.tanh(float(v))) for v in argument.flat],np.float32).reshape(argument.shape)
    ulp=np.abs(target.view(np.uint32).astype(np.int64)-desired.view(np.uint32).astype(np.int64))
    need(np.all(ulp<=2),'tanhf exceeds documented comparison2ULP')
    need(np.array_equal(f[...,13],f[...,12]*(f[...,11]-f[...,0])),'RHS derivative')
    if reference is not None:
        need(np.array_equal(d['ids'],reference['target_ids'][:2]),'Frozen destination IDs')
        for name,index in [('sham_cuda_theta',5),('sham_cuda_gain',6),('sham_cuda_tau',9)]:
            need(np.array_equal(f[...,index],np.broadcast_to(reference[name][:2],(n,4,2))),'Frozen parameter '+name)
        need(np.all(f[...,4]==0),'Undeclared DN drive')
    return dict(RHS_evaluations=n*4*2,maximum_target_ULP=int(ulp.max()),
                law_clock_and_rate_checked=True,frozen_parameters_checked=reference is not None)

def verify_all(folder=HERE):
    folder=Path(folder);p=folder/'reference/frozen_selection49.npz'
    need(sha(p)==REFERENCE_SHA256,'Frozen49 parameter reference')
    reference=read_npz(p)
    return {a:check_law(read_npz(folder/a/'dng100_observed.npz'),reference) for a in ARMS}

if __name__=='__main__':
    import json
    print(json.dumps(verify_all()))
