"""Finite complete one-second generic workload, synthetic; not the complete fly."""
from pathlib import Path
import sys,json,time,resource,signal,platform
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
from model import Model,require
from runtime import Engine
from extensions import fixture
from extension_test import freeze_check

def main():
    role=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);freeze_check()
    # Fixed BEFORE any large results, shared by both algorithms and both references.
    counts=(90000,74700,2000);spec=fixture(counts,seed=0)
    spec['connections'][0]['offsets']=list(range(1,257));spec['connections'][0]['weight']=.2/256
    times=np.linspace(0,1,21);started=time.perf_counter()
    signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('individual 300s budget exhausted')))
    signal.alarm(300)
    result={'scope':spec['description'],'role':role,'counts':counts,'simulated_target_s':1.,'trajectory_probes':4096,'sample_count':len(times),'cpu_threads':1,'python':sys.version,'numpy':np.__version__}
    try:
        if role.startswith('reference'):
            m=Model(spec);setup=time.perf_counter()-started;refined=role=='reference_fine'
            probe=np.unique(np.linspace(0,m.n-1,4096,dtype=np.int64));advance=time.perf_counter()
            sol=solve_ivp(m.rhs,(0,1),m.initial,method='DOP853',rtol=1e-12 if refined else 1e-11,atol=(1e-14 if refined else 1e-13)*m.scale,t_eval=times)
            require(sol.success,'reference incomplete');trace=sol.y[probe];final=sol.y[:,-1].copy()
            result.update(steps=None,nfev=sol.nfev,advance_including_scans_s=time.perf_counter()-advance,simulated_actual_s=float(sol.t[-1]))
        else:
            algorithm,mode=role.split('_');e=Engine(spec,profile=mode,algorithm=algorithm);m=e.model
            probe=np.unique(np.linspace(0,m.n-1,4096,dtype=np.int64));setup=time.perf_counter()-started
            require(e.gpu.cp.get_default_memory_pool().total_bytes()<12*1024**3,'device memory budget exceeded')
            ys=[];advance=time.perf_counter();ys.append(e.gpu.read(probe))
            for t in times[1:]:
                e.advance(float(t));ys.append(e.gpu.read(probe))
                print(json.dumps({'role':role,'simulated_s':e.t,'wall_s':time.perf_counter()-advance}),flush=True)
            final=e.read();trace=np.asarray(ys).T
            result.update(steps=e.gpu.accepted,rejected=e.gpu.rejected,advance_including_scans_s=time.perf_counter()-advance,simulated_actual_s=e.t,
                          device_pool_GiB=e.gpu.cp.get_default_memory_pool().total_bytes()/1024**3,
                          explicit_copy_payloads=e.gpu.transfers,
                          hardware=e.gpu.cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
        conservation=sum(final[sl] for sl in m.by_id['chemistry']['states'].values())
        require(np.isfinite(final).all() and np.isfinite(trace).all(),'nonfinite recorded trajectory')
        result.update(status='COMPLETE',model_sha256=m.identity,states=m.n,connections=m.connection.nnz,setup_s=setup,chemical_conservation_absolute=float(np.max(abs(conservation-1))))
        np.savez_compressed(out/'trajectory.npz',times=times,probes=probe,trace=trace,final=final,scale=m.scale,chemical_total=conservation)
    except Exception as ex:
        result.update(status='INCOMPLETE',exception_type=type(ex).__name__,message=str(ex))
        raise
    finally:
        signal.alarm(0);result['total_worker_s']=time.perf_counter()-started;result['peak_RSS_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
