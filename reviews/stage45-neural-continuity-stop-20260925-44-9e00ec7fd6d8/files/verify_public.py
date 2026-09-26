"""Rebuild public observable/budget evidence; full initial states are local-only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for item in manifest['files']:
        f=HERE/item['path'];need(f.is_file() and f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],'Changed public member')
    plan=json.loads((HERE/'context/PLAN.json').read_text());r=json.loads((HERE/'context/RESULT.json').read_text())
    need(r['stage4_admission'] is False and r['stage5_admission'] is False,'Corrupt admission flag')
    with np.load(HERE/'data/sham_traces.npz',allow_pickle=False) as s,np.load(HERE/'data/donor_traces.npz',allow_pickle=False) as d:
        selected=(d['fase']=='ensayo')&(d['paso']>=1001)&(d['paso']<=1120)
        need(np.array_equal(s['paso'],np.arange(1001,1121)),'Wrong window')
        for key in d.files:
            x,y=s[key],d[key][selected]
            need(x.shape==y.shape and np.array_equal(x,y),'Different raw observable '+key)
            if key!='fase':need(x.dtype==y.dtype,'Changed numeric dtype '+key)
            if x.dtype.kind in 'fc':need(np.isfinite(x).all(),'Nonfinite '+key)
        count=len(d.files)
    cpu=r['cpu_s'];limit=plan['budget']['cpu_s_aggregate_max']
    need(type(cpu) in (int,float) and np.isfinite(cpu) and cpu>limit,'Budget stop not reconstructed')
    out={'schema':'stage45_public_sham_budget44_recomputed_v1','classification':'BLOQUEADO',
         'exact_observable_fields':count,'exact_rows':120,'measured_cpu_s':cpu,'cpu_budget_s':limit,
         'cpu_excess_s':cpu-limit,'common_and_virtual_runs':0,'stage4_admission':False,'stage5_admission':False,
         'complete_initial_state_recomputed':False,'scope':'Public subset reconstructs observable equality and CPU stop. Full checkpoints and their clean CPU state comparison are in the separately hashed local full capsule; not in this public subset.'}
    with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
