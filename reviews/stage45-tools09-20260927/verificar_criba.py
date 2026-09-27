"""Recompute metrics and decisions from saved predictions; no fitting or CNS."""
from pathlib import Path
import copy, hashlib, json, re
import numpy as np
H=Path(__file__).resolve().parent
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(result, arrays, contract, data):
    need(data['ids'].tolist()==[10045,10056,10118,10065,10442,10760,523769,10360,10888,11067,11074,512006,11960,11702,523640,10371],'Neuron identities')
    need(contract['train_arms']==['air0_odor0','air0_odor1','airL_odor0','airR_odor0'],'Training split')
    need(contract['test_arms']==['airL_odor1','airR_odor1'],'Test split')
    need(contract['time']=={'first_input_row':10,'last_input_row_exclusive':80,'forecast_rows':10,'sampling_ms':1},'Time contract')
    truth=np.stack([data[n+'__q'][20:90] for n in contract['test_arms']])
    current=np.stack([data[n+'__q'][10:80] for n in contract['test_arms']])
    need(np.array_equal(truth,arrays['truth']) and np.array_equal(current,arrays['current']),'Target/current mapping')
    need(np.array_equal(current,arrays['persistence']),'Persistence')
    names=['persistence','state16','PCA4']+['dictionary_'+str(s) for s in [11,29,47,71]]
    metrics={}
    for n in names:
        p=arrays[n];need(p.shape==truth.shape and np.isfinite(p).all(),'Prediction finite/shape')
        e=p-truth;be=e[:,:,2]-e[:,:,3];metrics[n]={}
        for i,arm in enumerate(contract['test_arms']):
            v={'q_rmse':float(np.sqrt(np.mean(e[i]**2))),
               'DNb05_difference_rmse':float(np.sqrt(np.mean(be[i]**2)))}
            v['DN_error_vs_persistence']=v['DNb05_difference_rmse']/(v['DNb05_difference_rmse'] if n=='persistence' else metrics['persistence'][arm]['DNb05_difference_rmse'])
            metrics[n][arm]=v
    need(result['metrics']==metrics,'Recomputed metrics')
    rule=re.search(r'mejorar al menos (\d+)%.*sin empeorar RMSE q >(\d+)%',contract['promotion_to_next_instrument_test'])
    need(rule is not None,'Unrecognized frozen rule')
    maximum_DN_ratio=1-int(rule.group(1))/100
    maximum_q_ratio=1+int(rule.group(2))/100
    gates={}
    for name in names[3:]:
        gates[name]={}
        for arm in contract['test_arms']:
            best=min(['state16','PCA4'],key=lambda k:metrics[k][arm]['DNb05_difference_rmse'])
            r=metrics[name][arm]['DNb05_difference_rmse']/metrics[best][arm]['DNb05_difference_rmse']
            qr=metrics[name][arm]['q_rmse']/metrics[best][arm]['q_rmse']
            gates[name][arm]={'comparator':best,'DN_error_ratio':r,'q_error_ratio':qr,'pass':bool(r<=maximum_DN_ratio and qr<=maximum_q_ratio)}
    promote=all(v['pass'] for d in gates.values() for v in d.values())
    need(result['gate']==gates and result['dictionary_next_test_supported']==promote,'Recomputed decision')
    need(result['classification']==('PROMETEDOR_NO_CONFIRMADO' if promote else 'DESCARTADO_EN_ESTA_CRIBA'),'Classification')
    need(result['stage4_pass'] is False and result['stage5_pass'] is False,'Stage status')
    return promote
def main():
    contract=json.loads((H/'CONTRATO_CRIBA.json').read_text());result=json.loads((H/'CRIBA.json').read_text());prov=json.loads((H/'PROCEDENCIA.json').read_text())
    for k,n in [('contract_sha256','CONTRATO_CRIBA.json'),('data_sha256','datos.npz'),('code_sha256','criba_instrumentos.py')]:need(sha(H/n)==prov[k],'Source '+n)
    with np.load(H/'PREDICCIONES.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    with np.load(H/'datos.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    verify(result,arrays,contract,data)
    tests=[]
    for label in ['stage','criterion','metric','neuron','nonfinite','target']:
        r=copy.deepcopy(result);a={k:v.copy() for k,v in arrays.items()};c=copy.deepcopy(contract);d={k:v.copy() for k,v in data.items()}
        if label=='stage':r['stage4_pass']=True
        elif label=='criterion':c['promotion_to_next_instrument_test']=c['promotion_to_next_instrument_test'].replace('10%','90%')
        elif label=='metric':r['metrics']['state16']['airL_odor1']['q_rmse']=0
        elif label=='neuron':d['ids'][0]=-1
        elif label=='nonfinite':a['dictionary_11'][0,0,0]=np.nan
        elif label=='target':a['truth'][0,0,0]+=1e-3
        try:verify(r,a,c,d)
        except ValueError:tests.append(label)
        else:raise ValueError('Corruption undetected: '+label)
    print(json.dumps({'verified_from_arrays':True,'corruptions_rejected':tests,'dictionary_next_test_supported':result['dictionary_next_test_supported'],'stage4_pass':False,'stage5_pass':False}))
if __name__=='__main__':main()
