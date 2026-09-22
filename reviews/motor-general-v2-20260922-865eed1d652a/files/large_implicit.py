"""One large continuous second; reuse only verified unchanged reference equations."""
from pathlib import Path
import sys,json,time,hashlib,importlib.util,resource,signal
import numpy as np
from model import Model,require
from extensions import fixture
from implicit import Implicit
H=Path(__file__).resolve().parent

def main():
    out=Path(sys.argv[1]);refs=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    spec=fixture((90000,74700,2000),seed=0);spec['connections'][0]['offsets']=list(range(1,257));spec['connections'][0]['weight']=.2/256
    profile=sys.argv[3] if len(sys.argv)>3 else 'fast';require(profile in ('fast','precise'),'profile');m=Model(spec);loader=importlib.util.spec_from_file_location('legacy_model',refs/'model.py');old=importlib.util.module_from_spec(loader);sys.modules[loader.name]=old;loader.loader.exec_module(old);oldm=old.Model(spec)
    original=json.loads((refs/'reference_fine/RESULT.json').read_text());require(oldm.identity==original['model_sha256'],'legacy reference descriptor mismatch')
    drift=[]
    for t,factor in [(0,1),(.317,.8),(1,1.1)]:
        x=m.initial*factor;a=m.raw_rhs(t,x);b=oldm.rhs(t,x);d=float(np.max(abs(a-b)/(1+abs(b))));drift.append(d);require(d<1e-14,'legacy literal normalization changed this reference RHS')
    def load(p):
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    fine=load(refs/'reference_fine/trajectory.npz');coarse=load(refs/'reference/trajectory.npz');times=fine['times'];probes=fine['probes'];scale=m.scale
    def metric(trace,final):
        a=np.max(abs(trace-fine['trace'])/(scale[probes,None]+abs(fine['trace'])));b=np.max(abs(final-fine['final'])/(scale+abs(fine['final'])));return float(max(a,b))
    refinement=metric(coarse['trace'],coarse['final']);require(refinement<1e-7,'reference unresolved')
    del oldm
    e=Implicit(spec,profile);ys=[];start=time.perf_counter();failure=None;ys.append(e.read()[probes])
    try:
        for t in times[1:]:
            e.advance(float(t),wall_limit_s=max(.01,180-(time.perf_counter()-started)));ys.append(e.read()[probes]);print(json.dumps({'simulated':e.t,'wall_s':time.perf_counter()-start}),flush=True)
    except Exception as exc:failure=repr(exc)
    final=e.read();elapsed=time.perf_counter()-start;trace=np.array(ys).T;stats=e.stats();e.close()
    completed=failure is None and e.t==1 and trace.shape==fine['trace'].shape
    err=metric(trace,final) if completed else None
    result={'profile':profile,'states':m.n,'connections':m.connection.nnz,'model_identity':m.identity,'descriptor_hash':m.descriptor_hash,'reference_legacy_identity':original['model_sha256'],'reference_rhs_drift':drift,'reference_refinement':refinement,'reference_hashes':{p:hashlib.sha256((refs/p).read_bytes()).hexdigest() for p in ['model.py','reference_fine/trajectory.npz','reference/trajectory.npz']},'setup_s':e.setup_s,'advance_scan_s':elapsed,'measured_window_s':time.perf_counter()-started,'actual_simulated_s':e.t,'completed':completed,'error':err,'eligible':completed and err<=({'fast':.01,'precise':1e-5}[profile]),'failure':failure,'stats':stats,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'scope':'Synthetic continuously coupled ODE; 4096 probes x21 times and every final state; not a brain, not biological validation'}
    np.savez_compressed(out/'trajectory.npz',times=times[:len(ys)],probes=probes,trace=trace,final=final,scale=scale)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
