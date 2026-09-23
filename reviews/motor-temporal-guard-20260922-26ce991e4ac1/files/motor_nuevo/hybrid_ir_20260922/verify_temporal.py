"""Verify the saved counterexample, then exercise the protected wrapper and CUDA."""
from pathlib import Path
import sys,json,hashlib,subprocess,os,numpy as np
from scipy.linalg import expm
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
manifest=json.loads((ROOT/'MANIFEST.json').read_text())
for row in manifest['files']:
 if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed evidence')
p=R/'late_event_cuda_01';r=json.loads((p/'RESULT.json').read_text());a=np.load(p/'states.npz',allow_pickle=False)
tau=125e-6;exact=expm(np.array([[-1.,0.,0.],[1.,-1.,0.],[0.,1.,-1.]])*.1)@np.array([.5,0.,0.])
if np.max(abs(exact-a['exact']))>1e-16 or r['criterion']!=1e-4:raise ValueError('Changed reference/criterion')
for name in ('uncertified','event_cut'):
 error=float(np.max(abs(a[name]-a['exact'])))
 if r['runs'][name]['max_abs_error']!=error:raise ValueError('Unreconstructed error')
if not (r['runs']['uncertified']['max_abs_error']>1e-4 and r['runs']['uncertified']['maximum_estimator']==0. and r['runs']['event_cut']['max_abs_error']<=1e-4 and r['status']=='COUNTEREXAMPLE_REPRODUCED'):raise ValueError('Unjustified verdict')
N=ROOT/'motor_nuevo/native_hybrid_20260922'
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(N/'graph_control_v2.cpp'),'-o',str(N/'libgraph_control_v2.so'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64','-lcudart'],check=True,timeout=30)
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUPY_CACHE_DIR=str(ROOT/'cuda_cache'))
subprocess.run([sys.executable,'-B','-O',str(R/'check_temporal_fallback.py')],cwd=ROOT,env=env,check=True,timeout=20)
subprocess.run([sys.executable,'-B','-O',str(R/'check_late_event_cuda.py'),str(R/'clean_late_replay')],cwd=ROOT,env=env,check=True,timeout=30)
print(json.dumps({'saved_counterexample_reconstructed':True,'CPU_wrapper_checked':True,'native_CUDA_rerun':True,'organism_loads_added':0}))
