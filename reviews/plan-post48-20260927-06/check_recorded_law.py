"""CPU-only frozen-input diagnostic; portable verification uses local NPZ only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def contract():
    p=json.loads((HERE/'OFFLINE_PROTOCOL.json').read_text())
    require(p['schema']==1 and p['arms']==['sham','dm1','profile','permuted'], 'Contract arms')
    require(p['blocks']==['00901_01000ms','02901_03000ms'] and p['selected_stages']==[0,1,2], 'Contract selection')
    require(p['ids']==[10045,10056] and p['no_admission_of_stage4_or5'] is True, 'Contract interpretation')
    return p
def extract(root):
    p=contract(); arrays={}; provenance=[]
    require(not (HERE/'recorded_law_inputs.npz').exists(),'Preserve previous extraction')
    frozen=json.loads((root/'FROZEN.json').read_text())
    for name in ['coefficient_observed.cu','observe.cu','observer.py']:
        require(sha(root/name)==frozen['files'][name],'Changed frozen source '+name)
    for arm in p['arms']:
        for block in p['blocks']:
            folder=root/arm/'blocks'/block; path=folder/'dng100_observed.npz'
            digest=sha(path); manifest=json.loads((folder/'MANIFEST.json').read_text())
            require(digest==manifest['hashes'][path.name],'Changed source block')
            with np.load(path,allow_pickle=False) as z:
                require(np.array_equal(z['ids'],p['ids']),'Cell identities')
                require(z['fields'].tolist()[:14]==p['fields'],'Field identities')
                r=z['records']; epochs=np.repeat(np.arange(len(z['trials'])),z['trials'])
                require(np.isfinite(r).all(),'Nonfinite source record')
                require(np.array_equal(r[:,138],((r[:,131]<=1)&(r[:,132]==0)&(r[:,133]==0)).astype(float)), 'Accepted predicate')
                take=z['committed'][epochs] & (r[:,138]==1)
                selected=r[take,:128].reshape(-1,4,2,16)[:,:3,:,:14]
                weights=r[take,129,None]*np.array([2/9,1/3,4/9])
                key=arm+'_'+block
                arrays[key+'_fields']=selected
                arrays[key+'_weights']=weights
                provenance.append(dict(key=key,source=str(path),source_sha256=digest,
                    source_manifest_sha256=sha(folder/'MANIFEST.json'),selected_trials=len(selected)))
    np.savez_compressed(HERE/'recorded_law_inputs.npz',**arrays)
    write(HERE/'OFFLINE_SOURCES.json',dict(contract_sha256=sha(HERE/'OFFLINE_PROTOCOL.json'),
        input_sha256=sha(HERE/'recorded_law_inputs.npz'),sources=provenance,
        frozen_sources={n:sha(root/n) for n in ['coefficient_observed.cu','observe.cu','observer.py']},
        coverage='Committed accepted stages1-3 in two100ms windows per arm; no per-presynaptic-cell trajectories'))
def calculate():
    p=contract(); provenance=json.loads((HERE/'OFFLINE_SOURCES.json').read_text())
    require(sha(HERE/'OFFLINE_PROTOCOL.json')==provenance['contract_sha256'],'Changed contract')
    require(sha(HERE/'recorded_law_inputs.npz')==provenance['input_sha256'],'Changed input')
    out={}
    with np.load(HERE/'recorded_law_inputs.npz',allow_pickle=False) as z:
        for arm in p['arms']:
            for block in p['blocks']:
                key=arm+'_'+block; f=z[key+'_fields']; w=z[key+'_weights']
                require(f.shape[1:]==(3,2,14) and w.shape==f.shape[:2], 'Shape mismatch')
                require(np.isfinite(f).all() and np.isfinite(w).all() and (w>0).all(),'Invalid values')
                require(abs(float(w.sum())-.1)<1e-12,'Window duration')
                v={name:f[...,i] for i,name in enumerate(p['fields'])}
                net=v['net'].astype(np.float32); drive=v['drive'].astype(np.float32)
                theta=v['theta'].astype(np.float32)
                margin=((net+drive)-theta).astype(np.float64)
                require(np.array_equal(margin,v['margin']),'Recorded margin identity')
                require(np.all(v['gain']>0) and np.all(v['tau']>0),'Positive gain/tau required')
                require(np.array_equal(v['base_target'],v['final_target']), 'Final target writer differs')
                require(np.array_equal(v['base_rate'],v['final_rate']), 'Final rate writer differs')
                require(np.array_equal(v['derivative'],v['final_rate']*(v['final_target']-v['state'])), 'RHS identity')
                require(np.all(v['margin']<0) and np.all(v['base_target']==0), 'Zero rectification observation changed')
                no_theta=(net+drive).astype(float)
                signed_error=v['net']-(v['positive_aux']+v['negative_aux'])
                cells=[]
                for i,ident in enumerate(p['ids']):
                    cells.append(dict(id=ident,means={n:float(np.sum(x[:,:,i]*w)/w.sum()) for n,x in v.items()},
                        state_min=float(v['state'][:,:,i].min()),state_max=float(v['state'][:,:,i].max()),
                        derivative_min=float(v['derivative'][:,:,i].min()),derivative_max=float(v['derivative'][:,:,i].max()),
                        max_margin=float(v['margin'][:,:,i].max()),max_margin_theta_zero=float(no_theta[:,:,i].max()),
                        theta_zero_would_still_be_rectified_zero=bool(np.all(no_theta[:,:,i]<0)),
                        auxiliary_signed_sum_max_abs_error=float(np.max(np.abs(signed_error[:,:,i])))))
                out[key]=dict(duration_s=float(w.sum()),evaluations_per_cell=int(f.shape[0]*3),cells=cells)
    return dict(scope=p['scope'],windows=out,identities_reconstructed=True,
        stage4_admission=False,stage5_admission=False,neural_steps=0,body_steps=0,
        limitation='Frozen recorded inputs only. No dynamic counterfactual, biological calibration or presynaptic causal attribution.')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--extract',type=Path);ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    t=time.process_time()
    if args.extract:extract(args.extract)
    result=calculate()
    if args.verify:
        require(result==json.loads((HERE/'OFFLINE_RESULT.json').read_text()),'Result mismatch')
    else:
        require(not (HERE/'OFFLINE_RESULT.json').exists(),'Preserve prior result');write(HERE/'OFFLINE_RESULT.json',result)
    cpu=time.process_time()-t;require(cpu<=contract()['CPU_seconds_max'],'CPU budget exceeded')
    print(json.dumps(dict(CPU_s=cpu,verified=args.verify,windows=len(result['windows']),neural_steps=0,body_steps=0)))
if __name__=='__main__':main()
