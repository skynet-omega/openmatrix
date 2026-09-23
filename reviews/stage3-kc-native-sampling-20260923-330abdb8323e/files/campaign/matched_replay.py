"""Offline CUDA arithmetic discriminator with real epoch inputs and equal h.

No organism calls. Identical frozen inputs feed original and device-style WARP.
This distinguishes compilation/arithmetic from adaptive schedule selection.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import cupy as cp

HERE=Path(__file__).resolve().parent
PLAN={'inputs':[0,1], 'trial_steps_ns':[25000,12500,6250,3125],
      'implementation_absolute_limit':1e-12,'historical_state_limit':1e-4,
      'cases':8,'scope':'same input and same full/two-half step; no cadence or model change'}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def main():
    dest=HERE/'matched_replay_01';dest.mkdir(exist_ok=False)
    (dest/'PLAN.json').write_text(json.dumps(PLAN,indent=2)+'\n')
    raw=(HERE/'warp_original.cu').read_text()
    original=cp.RawKernel(raw,'warp_midpoint',options=('--fmad=false',))
    dev=raw.replace('extern "C" __global__ void warp_midpoint','__device__ __noinline__ void warp_midpoint')
    for needle in ('for(int j=0;j<17;j++)','for(int j=16;j>=0;j--)','for(int k=j+1;k<17;k++)'):
        dev=dev.replace(needle,'\n#pragma unroll\n'+needle)
    dev+='''
extern "C" __global__ void replay_step(int count,double dt,double rest,const double*v,const double*g,
const double*ge,const double*gi,const double*cur,const double*C,const double*G,
const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
double*vo,double*go,double*errors){
 warp_midpoint(count,dt,rest,v,g,ge,gi,cur,C,G,chanG,chanb,shuntG,shuntb,ena,vo,go,errors);
}
'''
    device=cp.RawKernel(dev,'replay_step',options=('--fmad=false',))
    rows=[];start=time.perf_counter()
    for call in PLAN['inputs']:
        source=HERE/'native_on_01/kc_native_0'/f'input_{call:02d}.npz'
        with np.load(source,allow_pickle=False) as z:
            arrays={k:cp.asarray(z[k]) for k in z.files if z[k].dtype.kind=='f'}
            rest=float(z['rest'])
        v=arrays['state_delta'];g=arrays['state_gates'];n=len(v)
        def step(kernel,vi,gi,h):
            vo=cp.empty_like(vi);go=cp.empty_like(gi);err=cp.zeros((n,2))
            kernel((n,),(32,),(np.int32(n),np.float64(h*1e-9),np.float64(rest),vi,gi,
                *(arrays[k] for k in ('ge','gi','current','C','G','chanG','chanb','shuntG','shuntb','ena')),
                vo,go,err))
            require(bool(cp.isfinite(err).all()) and bool(cp.isfinite(vo).all()) and bool(cp.isfinite(go).all()),'invalid trial')
            return vo,go
        for h in PLAN['trial_steps_ns']:
            outputs=[]
            for kernel in (original,device):
                vf,gf=step(kernel,v,g,h);va,ga=step(kernel,v,g,h//2);vb,gb=step(kernel,va,ga,h-h//2)
                cubes=(h//2/h)**3+((h-h//2)/h)**3;factor=cubes/(1-cubes)
                norm=cp.maximum(cp.max(cp.abs(vb-vf),axis=1)*factor/2e-5,
                                cp.max(cp.abs(gb-gf),axis=(1,2))*factor/2e-7)
                outputs.append([cp.asnumpy(x) for x in (vf,gf,va,ga,vb,gb,norm)])
            diffs={k:float(np.max(np.abs(a-b))) for k,a,b in zip(('vf','gf','va','ga','vb','gb','estimator'),*outputs)}
            rows.append({'call':call,'h_ns':h,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'maximum_absolute_differences':diffs,
                'original_estimator_by_cell':outputs[0][-1].tolist(),
                'same_acceptance_decisions':bool(np.array_equal(outputs[0][-1]<=1,outputs[1][-1]<=1)),
                'implementation_pass':max(diffs[k] for k in ('vf','gf','va','ga','vb','gb'))<=PLAN['implementation_absolute_limit']})
    result={'rows':rows,'all_implementation_pass':all(r['implementation_pass'] and r['same_acceptance_decisions'] for r in rows),
        'wall_s':time.perf_counter()-start,'warp_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'scope':'Selected cells, two first-epoch inputs; supports or refutes arithmetic mismatch only. No full trajectory equivalence or stage3 admission.'}
    (dest/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
