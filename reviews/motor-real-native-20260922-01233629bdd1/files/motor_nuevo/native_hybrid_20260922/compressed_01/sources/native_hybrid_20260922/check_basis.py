from pathlib import Path
import sys,json,time
from types import SimpleNamespace
import numpy as np,cupy as cp
HERE=Path(__file__).resolve().parent;OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work')
sys.path[:0]=[str(OLD/'motor13_20260922'),str(HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922')]
from verify_transport import read_state
from kc_fused_warp import WARP,KERNEL
from basis_compile import compile_basis,compile_warp
s=read_state(HERE/'baseline_01/brain_final');c=s['kc_spatial_manifest']['cell'];st=s['kc_spatial_state'];n=len(st['delta'])
G=c['channel_G_nS'].reshape(51,17,17);b=c['channel_b_nS'].reshape(51,17)
cG,cb,groups,report=compile_basis(G,b)
# Native channel reversals relative to rest, from the declared inherited model.
ena=np.tile(np.array([60.,60.,-80.])-c["rest_mV"],17)
rng=np.random.default_rng(20260922);operator=[]
for _ in range(8):
 f=rng.uniform(0,1,51);cg=np.array([sum(f[i]*k for i,k in members) for members in groups]);cf=np.array([sum(f[i]*ena[i]*k for i,k in members) for members in groups])
 actualG=np.einsum('k,kij->ij',f,G);actualb=(f*ena)@b;newG=np.einsum('k,kij->ij',cg,cG);newb=cf@cb
 if not np.allclose(actualG,newG,rtol=1e-11,atol=1e-10) or not np.allclose(actualb,newb,rtol=1e-11,atol=1e-10):raise RuntimeError('Operator changed')
 operator.append(max(float(np.max(abs(actualG-newG))),float(np.max(abs(actualb-newb)))))
compiled=cp.RawKernel(compile_warp(WARP,groups),'warp_midpoint',options=('--fmad=false',))
common=[cp.asarray(x) for x in (st['delta'],st['gates'],rng.uniform(0,.5,(n,4)),rng.uniform(0,.5,(n,4)),rng.uniform(-1,1,(n,17)),c['C_nF'],c['G_nS'])]
def run(kernel,compressed,dt):
 vo=cp.empty((n,17));go=cp.empty((n,17,4));err=cp.zeros((n,2))
 args=(np.int32(n),np.float64(dt),np.float64(c['rest_mV']),*common,cp.asarray(cG if compressed else G),cp.asarray(cb if compressed else b),cp.asarray(c['shunt_G']),cp.asarray(c['shunt_b']),cp.asarray(ena),vo,go,err)
 kernel((n,),(32,),args);cp.cuda.get_current_stream().synchronize()
 if not bool(cp.isfinite(err).all()) or not bool(cp.isfinite(vo).all()):raise RuntimeError('Bad dense residual')
 start=cp.cuda.Event();end=cp.cuda.Event();start.record()
 for _ in range(20):kernel((n,),(32,),args)
 end.record();end.synchronize()
 return vo.get(),go.get(),float(cp.cuda.get_elapsed_time(start,end)/20),kernel.attributes
checks=[]
for ns in (200,1562,6250,25000):
 a,ga,t0,ka=run(KERNEL,False,ns*1e-9);b1,gb,t1,kb=run(compiled,True,ns*1e-9)
 if not np.allclose(a,b1,rtol=1e-11,atol=1e-10) or not np.allclose(ga,gb,rtol=1e-11,atol=1e-10):raise RuntimeError('Step changed')
 checks.append(dict(ns=ns,voltage_max_abs=float(np.max(abs(a-b1))),gate_max_abs=float(np.max(abs(ga-gb))),reference_gpu_ms=t0,compiled_gpu_ms=t1,speedup=t0/t1))
report.update(operator_max_abs=max(operator),checks=checks,kernel_reference=ka,kernel_compiled=kb)
(HERE/'BASIS_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
