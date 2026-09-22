from pathlib import Path
import sys,json,hashlib,copy,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
from model import Model,require
from runtime import Engine
from extensions import fixture
H=Path(__file__).resolve().parent

def freeze_check():
    freeze=json.loads((H/'CORE_FREEZE.json').read_text())
    repair=json.loads((H/'CORE_REPAIR.json').read_text()) if (H/'CORE_REPAIR.json').exists() else None
    if repair:
        require(repair['before']==freeze['files'],'repair provenance mismatch')
        for name,digest in freeze['files'].items():require(hashlib.sha256((H/'frozen_initial'/name).read_bytes()).hexdigest()==digest,'initial frozen source corrupted')
    for name,digest in (repair['after'] if repair else freeze['files']).items():require(hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,'unregistered core change')

def main():
    freeze_check();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);results=[];start=time.perf_counter()
    for seed in range(4):
        spec=fixture(seed=seed);m=Model(spec);ts=np.linspace(0,.1,21)
        ref=solve_ivp(m.rhs,(0,.1),m.initial,method='DOP853',rtol=1e-11,atol=1e-13*m.scale,t_eval=ts)
        finer=solve_ivp(m.rhs,(0,.1),m.initial,method='DOP853',rtol=1e-12,atol=1e-14*m.scale,t_eval=ts)
        require(ref.success and finer.success,'reference failure')
        require(np.max(abs(ref.y-finer.y)/(m.scale[:,None]+abs(finer.y)))<=1e-7,'reference refinement fails')
        for algorithm in ('A','C'):
            for mode,limit in [('precise',1e-5),('fast',.01)]:
                e=Engine(spec,profile=mode,algorithm=algorithm);ys=[e.read()]
                for t in ts[1:]:e.advance(float(t));ys.append(e.read())
                y=np.array(ys).T;err=float(np.max(abs(y-finer.y)/(m.scale[:,None]+abs(finer.y))))
                total=sum(y[s] for s in m.by_id['chemistry']['states'].values());conservation=float(np.max(abs(total-1)))
                np.savez_compressed(out/f'seed{seed}_{algorithm}_{mode}.npz',times=ts,candidate=y,reference=finer.y,reference_coarse=ref.y,scale=m.scale,chemical_total=total)
                results.append({'seed':seed,'algorithm':algorithm,'mode':mode,'global_scaled_error':err,'limit':limit,'conservation_absolute':conservation,'eligible':err<=limit and conservation<=1e-7})
        if seed==0:
            # Rename anatomical labels and permute storage order without changing equations.
            renamed=copy.deepcopy(spec);mapping={p['id']:f'neutral_{i}' for i,p in enumerate(renamed['populations'])}
            for p in renamed['populations']:p['id']=mapping[p['id']]
            for c in renamed['connections']:c['source'][0]=mapping[c['source'][0]];c['target'][0]=mapping[c['target'][0]]
            a=Engine(spec);b=Engine(renamed);a.advance(.05);b.advance(.05)
            require(np.array_equal(a.read(),b.read()),'anatomical labels changed trajectory')
            reordered=copy.deepcopy(spec);reordered['populations']=reordered['populations'][::-1]
            c=Engine(reordered);c.advance(.05)
            for pop in a.model.pops:
                for field in pop['states']:
                    av=np.array(a.scan(pop['id'],field)['values']);cv=np.array(c.scan(pop['id'],field)['values'])
                    require(np.max(abs(av-cv))<1e-9,'population storage permutation changes trajectory')
            # Restoring C and modifying its model also exercise the second algorithm's graph ownership.
            c=Engine(spec,algorithm='C');c.advance(.02);c.checkpoint(out/'checkpoint_C');d=Engine.restore(out/'checkpoint_C')
            c.advance(.04);d.advance(.04);require(np.array_equal(c.read(),d.read()),'C replay')
    freeze_check();result={'core_matches_registered_repair':True,'reserve_status':'author-exposed extension after hash freeze; generic dtype repair subsequently applied, all affected arms repeated; not blind','models':results,'labels_bit_exact':True,'population_order_error_below':1e-9,'C_checkpoint_bit_exact':True,'total_wall_s':time.perf_counter()-start}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
