"""Narrow retrospective repair: exact extension cohort and reconstructed eligibility.
Reads preserved evidence; never upgrades the old scientific scope.
"""
from pathlib import Path
import json,copy,sys,numpy as np
from model import require

def verify(root,records=None):
    root=Path(root);records=json.loads((root/'RESULT.json').read_text())['models'] if records is None else records
    expected={(seed,algorithm,mode) for seed in range(4) for algorithm in ['A','C'] for mode in ['fast','precise']}
    keys=[(r['seed'],r['algorithm'],r['mode']) for r in records]
    require(len(keys)==len(set(keys)) and set(keys)==expected,'extension cohort duplicated/missing/unexpected')
    rows=[]
    for r in records:
        with np.load(root/f'seed{r["seed"]}_{r["algorithm"]}_{r["mode"]}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        require(all(np.isfinite(v).all() for v in d.values()),'nonfinite data')
        require(d['candidate'].shape==d['reference'].shape==d['reference_coarse'].shape,'trajectory shape')
        require(np.array_equal(d['times'],np.linspace(0,.1,21)),'time context')
        err=float(np.max(abs(d['candidate']-d['reference'])/(d['scale'][:,None]+abs(d['reference']))))
        refinement=float(np.max(abs(d['reference_coarse']-d['reference'])/(d['scale'][:,None]+abs(d['reference']))))
        total=d['candidate'][-144:].reshape(12,12,-1).sum(axis=0);conservation=float(np.max(abs(total-1)))
        limit={'fast':.01,'precise':1e-5}[r['mode']]
        require(r['limit']==limit and refinement<=1e-7,'criterion/reference')
        require(np.isclose(err,r['global_scaled_error'],rtol=1e-12,atol=0),'error summary')
        require(np.allclose(total,d['chemical_total'],rtol=0,atol=1e-14),'chemical summary')
        eligible=bool(err<=limit and conservation<=1e-7)
        require(r['eligible'] is eligible,'eligible contradicts trajectories/conservation')
        rows.append({'seed':r['seed'],'algorithm':r['algorithm'],'mode':r['mode'],'error':err,'eligible':eligible})
    return rows

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);rows=verify(root)
    records=json.loads((root/'RESULT.json').read_text())['models'];detected=[]
    for kind in ['duplicate','eligible','criterion','context']:
        bad=copy.deepcopy(records)
        if kind=='duplicate':bad[-1]=bad[0]
        if kind=='eligible':bad[0]['eligible']=not bad[0]['eligible']
        if kind=='criterion':bad[0]['limit']*=10
        if kind=='context':bad[0]['seed']=99
        try:verify(root,bad)
        except ValueError:detected.append(kind)
        else:raise RuntimeError('escaped corruption '+kind)
    result={'scope':'previous extension only; exact cohort and eligible flag reconstructed; no numerical rerun','rows':rows,'corruptions_detected':detected}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'conditions':len(rows),'corruptions':detected}))
if __name__=='__main__':main()
