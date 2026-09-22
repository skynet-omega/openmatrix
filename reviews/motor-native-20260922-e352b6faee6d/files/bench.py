"""Long-horizon generic workloads, measured from import-complete model construction through scans."""
from pathlib import Path
import argparse,sys,time,json,resource,hashlib
import numpy as np
import cupy as cp
from model import Model,require
from extensions import fixture
from runtime import Engine,PROFILES
from implicit import Implicit
H=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--implementation',choices=['native','python','reference','reference_fine'],required=True);p.add_argument('--profile',default='precise');p.add_argument('--seconds',type=float,default=1);p.add_argument('--reference-dir');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();spec=fixture((90000,74700,2000),seed=0);spec['connections'][0]['offsets']=list(range(1,257));spec['connections'][0]['weight']=.2/256
    m=Model(spec);times=np.linspace(0,a.seconds,21 if a.seconds==1 else 41);probes=np.unique(np.linspace(0,m.n-1,4096,dtype=np.int64))
    reference=a.implementation.startswith('reference');tols=(1e-12,1e-14) if a.implementation=='reference_fine' else (1e-10,1e-12)
    if reference:PROFILES['precise']=tols;e=Engine(spec,profile='precise')
    else:e=Implicit(spec,a.profile,implementation=a.implementation)
    setup=time.perf_counter()-begin;start=time.perf_counter();traces=[e.read()[probes]];failure=None
    try:
        for t in times[1:]:
            e.advance(float(t),wall_limit_s=max(.01,235-(time.perf_counter()-begin)));traces.append(e.read()[probes])
            print(json.dumps({'time':e.t,'wall_s':time.perf_counter()-start}),flush=True)
    except Exception as exc:failure=repr(exc)
    final=e.read();advance=time.perf_counter()-start;trace=np.array(traces).T
    stats={'accepted':e.gpu.accepted,'rejected':e.gpu.rejected} if reference else e.stats()
    if not reference:e.close()
    completed=e.t==a.seconds and trace.shape==(4096,len(times)) and failure is None
    np.savez_compressed(out/'trajectory.npz',times=times[:len(traces)],probes=probes,trace=trace,final=final,scale=m.scale)
    err=None;refhash=None
    if a.reference_dir:
        rp=Path(a.reference_dir)/'trajectory.npz';refhash=hashlib.sha256(rp.read_bytes()).hexdigest()
        with np.load(rp,allow_pickle=False) as f:
            require(np.array_equal(f['probes'],probes) and np.array_equal(f['times'],times),'reference sampling context')
            if completed:err=float(max(np.max(abs(trace-f['trace'])/(m.scale[probes,None]+abs(f['trace']))),np.max(abs(final-f['final'])/(m.scale+abs(f['final'])))))
    result={'implementation':a.implementation,'profile':a.profile,'reference_tolerances':tols if reference else None,'states':m.n,'connections':m.connection.nnz,'model_identity':m.identity,'descriptor_hash':m.descriptor_hash,'simulated_s':e.t,'target_s':a.seconds,'completed':completed,'failure':failure,'setup_s':setup,'advance_scan_s':advance,'measured_window_s':time.perf_counter()-begin,'error':err,'reference_sha256':refhash,'eligible':completed and (err is None or err<=({'fast':.01,'precise':1e-5}[a.profile])),'stats':stats,'host_peak_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'accuracy_scope':'4096 probes at each sample and every final state, not a continuous global bound','reference_scope':'1s DOP853 CPU historical refined reference; 10s generated explicit GPU RK4 with two tighter tolerances (independent integration, shared descriptor/IR)'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
