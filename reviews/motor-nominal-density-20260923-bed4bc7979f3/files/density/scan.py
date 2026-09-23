"""Count real graded coordinates from a saved one-ms sham; no GPU imports."""
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
A=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage234_settling_extension_20260915/settled_700ms/core_carrier')
B=ROOT/'campanas/etapa3_pn629_intervention_20260923_15/smoke_on_01/final_state'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(folder):
    d=json.loads((folder/'session.json').read_text());h=d['hybrid']
    with np.load(folder/'session.npz',allow_pickle=False) as z:
        state=z[h['state']['__array__']].copy()
        photo=h['photo_ids'];photo=z[photo['__array__']] if isinstance(photo,dict) else np.asarray(photo)
        return state,len(photo),int(h['time_ns'])
def main():
    start=time.perf_counter();plan=json.loads((HERE/'PLAN.json').read_text())
    x,m,t0=read(A);y,n,t1=read(B);N=166700
    if m!=n or t1-t0!=1_000_000 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Unmatched or nonfinite saved states')
    output={}
    for name,sl in [('release_q',slice(0,N)),('transmission_s',slice(N+2*m,2*N+2*m))]:
        a,b=x[sl],y[sl]
        if len(a)!=N:raise ValueError('Wrong canonical layout')
        delta=np.abs(b-a)
        output[name]={'coordinates':N,'nonzero_initial':int(np.count_nonzero(a)),'nonzero_final':int(np.count_nonzero(b)),
            'changed_exactly':int(np.count_nonzero(delta)),
            'absolute_change_gt':{str(e):int(np.count_nonzero(delta>e)) for e in (1e-10,1e-8,1e-6,1e-4,1e-2)},
            'max_change':float(delta.max()),'percentiles_abs_change':np.percentile(delta,[0,25,50,75,90,99,100]).tolist()}
    elapsed=time.perf_counter()-start;ram=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    if elapsed>plan['budget']['wall_s'] or ram>plan['budget']['RAM_GiB']:raise RuntimeError('Offline audit budget exceeded')
    result={'source_clocks_ns':[t0,t1],'photoreceptor_coordinates':m,'full_state_size':len(x),'counts':output,
      'sources':{str(p):sha(p) for p in (A/'session.json',A/'session.npz',B/'session.json',B/'session.npz',Path(__file__))},
      'wall_s':elapsed,'peak_RAM_GiB':ram,'GPU_executed':False,
      'scope':'Single existing1ms sham after declared PN629/parameter preparation; q/s are graded model coordinates, not measured biological spikes. No recurrent influence bound or whole-life sparsity claim.'}
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
