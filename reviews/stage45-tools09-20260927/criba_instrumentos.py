"""External, retrospective conditional forecasting. Never imports a CNS runner."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse, hashlib, json, resource, time, warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.linear_model import Ridge

H = Path(__file__).resolve().parent
EXPECTED_IDS = [10045,10056,10118,10065,10442,10760,523769,10360,10888,11067,11074,512006,11960,11702,523640,10371]
def need(x, message):
    if not x: raise ValueError(message)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def prepare():
    need(not (H/'datos.npz').exists(), 'Preserve existing evidence')
    c=json.loads((H/'CONTRATO_CRIBA.json').read_text())
    base=H.parents[1]/'campanas'/c['source_campaign']
    manifest=json.loads((base/'MANIFEST.json').read_text())['files']
    a={}; hashes={}; arms=c['train_arms']+c['test_arms']
    for name in arms:
        p=base/name/'neural_and_inputs.npz'; k=str(p.relative_to(base))
        need(sha(p)==manifest[k]['sha256'], 'Frozen source hash '+name)
        hashes[k]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            need(z['ids'].tolist()==EXPECTED_IDS, 'Neuron order')
            for key in ['q','JO_drive','nominal_Hz']:
                v=z[key];need(np.isfinite(v).all(), 'Finite '+key);a[name+'__'+key]=v
            if 'ids' not in a:
                for key in ['ids','JO_ids','ORN_ids']: a[key]=z[key]
            else:
                for key in ['ids','JO_ids','ORN_ids']:need(np.array_equal(a[key],z[key]),'Identity '+key)
    np.savez_compressed(H/'datos.npz', **a)
    dump(H/'PROCEDENCIA.json',{'source_hashes':hashes,'campaign_manifest_sha256':sha(base/'MANIFEST.json'),
         'data_sha256':sha(H/'datos.npz'),'contract_sha256':sha(H/'CONTRATO_CRIBA.json'),
         'code_sha256':sha(Path(__file__)),'sklearn':sklearn.__version__,'numpy':np.__version__})

def calculate():
    started=time.process_time(); c=json.loads((H/'CONTRATO_CRIBA.json').read_text())
    provenance=json.loads((H/'PROCEDENCIA.json').read_text())
    for key,filename in [('data_sha256','datos.npz'),('contract_sha256','CONTRATO_CRIBA.json'),('code_sha256','criba_instrumentos.py')]:
        need(sha(H/filename)==provenance[key], 'Provenance '+key)
    with np.load(H/'datos.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
    need(a['ids'].tolist()==EXPECTED_IDS,'Neuron identities')
    for v in a.values():need(np.isfinite(v).all(),'Nonfinite input')
    tr=c['train_arms'];te=c['test_arms'];arms=tr+te
    first=c['time']['first_input_row'];last=c['time']['last_input_row_exclusive'];lag=c['time']['forecast_rows'];nt=last-first
    q=np.stack([a[n+'__q'][first:last] for n in arms])
    future=np.stack([a[n+'__q'][first+lag:last+lag] for n in arms])
    u=np.stack([np.concatenate([a[n+'__JO_drive'][first:last],a[n+'__nominal_Hz'][first:last]],axis=1) for n in arms])
    need(q.shape==(6,70,16) and future.shape==q.shape, 'Frozen shape')
    qm=q[:4].reshape(-1,16).mean(axis=0);qs=np.maximum(q[:4].reshape(-1,16).std(axis=0),1e-6)
    x=(q-qm)/qs; um=u[:4].reshape(280,-1).mean(axis=0);us=u[:4].reshape(280,-1).std(axis=0);keep=us>1e-9
    u=(u[:,:,keep]-um[keep])/us[keep]
    delta=(future-q)/qs; xtr=x[:4].reshape(280,16);xte=x[4:].reshape(140,16)
    utr=u[:4].reshape(280,-1);ute=u[4:].reshape(140,-1)
    metrics={};preds={};fits={};warns={}; recon={};bases={}
    def evaluate(name,pred):
        need(pred.shape==(2,70,16) and np.isfinite(pred).all(),'Finite predictions')
        err=pred-future[4:];bil=err[:,:,2]-err[:,:,3]
        metrics[name]={n:{'q_rmse':float(np.sqrt(np.mean(err[i]**2))),
                             'DNb05_difference_rmse':float(np.sqrt(np.mean(bil[i]**2)))} for i,n in enumerate(te)}
        preds[name]=pred
    evaluate('persistence', q[4:].copy())
    variants=[('state16',None),('PCA4',PCA(n_components=4,svd_solver='full'))]
    for seed in c['dictionary']['seeds']:
        cfg={k:v for k,v in c['dictionary'].items() if k not in ['class','seeds']}
        variants.append(('dictionary_'+str(seed),DictionaryLearning(**cfg,random_state=seed,n_jobs=1)))
    for name,model in variants:
        tick=time.process_time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            if model is None: ztr=xtr;zte=xte;xr=xte
            else:
                model.fit(xtr);ztr=model.transform(xtr);zte=model.transform(xte)
                if name=='PCA4':xr=model.inverse_transform(zte)
                else:xr=zte@model.components_
                bases[name]=model.components_
            reg=Ridge(**{k:v for k,v in c['regression'].items() if k!='class'})
            reg.fit(np.concatenate([ztr,utr],axis=1),delta[:4].reshape(280,16))
            dp=reg.predict(np.concatenate([zte,ute],axis=1)).reshape(2,70,16)*qs
            evaluate(name,q[4:]+dp)
            warns[name]=sorted(set(str(v.message) for v in w))
        fits[name]=time.process_time()-tick
        recon[name]=float(np.sqrt(np.mean(((xr-xte)*qs)**2)))
    singular=np.linalg.svd(xtr-xtr.mean(axis=0),compute_uv=False)
    variance=singular**2; frac=np.cumsum(variance)/variance.sum()
    # The frozen gate is evaluated in both withheld conditions, never averaged away.
    gate={}
    for seed in c['dictionary']['seeds']:
        n='dictionary_'+str(seed);gate[n]={}
        for arm in te:
            best=min(['state16','PCA4'],key=lambda k:metrics[k][arm]['DNb05_difference_rmse'])
            ratio=metrics[n][arm]['DNb05_difference_rmse']/metrics[best][arm]['DNb05_difference_rmse']
            qratio=metrics[n][arm]['q_rmse']/metrics[best][arm]['q_rmse']
            gate[n][arm]={'comparator':best,'DN_error_ratio':ratio,'q_error_ratio':qratio,'pass':bool(ratio<=.9 and qratio<=1.05)}
    promote=all(v['pass'] for d in gate.values() for v in d.values())
    for n,by in metrics.items():
        for arm, v in by.items():
            v['DN_error_vs_persistence']=v['DNb05_difference_rmse']/metrics['persistence'][arm]['DNb05_difference_rmse']
    result={'scope':c['scope'],'conditions':{'train':tr,'excluded_from_fit':te,'independent_preparations':1},
       'rows_train':280,'rows_test':140,'state_columns':16,'input_columns_retained':int(keep.sum()),
       'state_dims_for_99pct_training_variance':int(np.searchsorted(frac,.99)+1),
       'state_dims_for_999pct_training_variance':int(np.searchsorted(frac,.999)+1),
       'train_state_variance_cumulative':frac.tolist(),'metrics':metrics,'reconstruction_q_rmse':recon,
       'cpu_fit_s':fits,'warnings':warns,'gate':gate,'dictionary_next_test_supported':promote,
       'classification':'PROMETEDOR_NO_CONFIRMADO' if promote else 'DESCARTADO_EN_ESTA_CRIBA',
       'stage4_pass':False,'stage5_pass':False,'CPU_s':time.process_time()-started,
       'RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       'numpy':np.__version__,'sklearn':sklearn.__version__}
    return result,dict(**preds,truth=future[4:],current=q[4:],**{'basis_'+k:v for k,v in bases.items()})

def stable(r):return {k:v for k,v in r.items() if k not in ['CPU_s','RSS_MiB','cpu_fit_s']}
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(30,31))
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.prepare:prepare();return
    result,arrays=calculate()
    if args.verify:
        saved=json.loads((H/'CRIBA.json').read_text())
        need(stable(result)==stable(saved),'Exact same-environment recomputation differs')
        with np.load(H/'PREDICCIONES.npz',allow_pickle=False) as z:
            need(set(z.files)==set(arrays),'Prediction keys')
            for k,v in arrays.items():need(np.array_equal(v,z[k]),'Exact prediction '+k)
        print(json.dumps({'verified':True,'CPU_s':result['CPU_s'],'RSS_MiB':result['RSS_MiB']}))
    else:
        need(not (H/'CRIBA.json').exists(),'Preserve result')
        dump(H/'CRIBA.json',result);np.savez_compressed(H/'PREDICCIONES.npz',**arrays)
        print(json.dumps({k:result[k] for k in ['classification','state_dims_for_99pct_training_variance','metrics','cpu_fit_s','CPU_s']}))
if __name__=='__main__':main()
