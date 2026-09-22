from pathlib import Path
import sys,json,time,numpy as np
from runtime import Engine
from pilot import case
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);rows=[];spec,end=case('hh')
with np.load(Path(sys.argv[2])/'reference.npz',allow_pickle=False) as z:times=z['times'];reference=z['tight'];scale=z['scale']
for profile in ['fast','precise']:
    started=time.perf_counter();e=Engine(spec,profile=profile);setup=time.perf_counter()-started;start=time.perf_counter();ys=[e.read()]
    for t in times[1:]:e.advance(float(t));ys.append(e.read())
    elapsed=time.perf_counter()-start;y=np.array(ys).T;err=float(np.max(abs(y-reference)/(scale[:,None]+abs(reference))))
    np.savez_compressed(out/(profile+'.npz'),times=times,candidate=y,reference=reference,scale=scale)
    rows.append({'profile':profile,'model_identity':e.model.identity,'simulated_s':end,'setup_s':setup,'advance_scan_s':elapsed,'error':err,'eligible':err<=({'fast':.01,'precise':1e-5}[profile])})
(out/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
