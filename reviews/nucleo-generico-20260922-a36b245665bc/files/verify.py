"""Reconstruct numerical decisions from arrays and frozen thresholds; no trusted PASS flags."""
from pathlib import Path
import sys,json,hashlib,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from model import require
from extension_test import freeze_check
H=Path(__file__).resolve().parent

def load(path):return json.loads(path.read_text())
def arrays(path):
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    for k,v in d.items():require(np.isfinite(v).all(),'nonfinite array: '+k)
    return d

def errors(data):
    require(np.array_equal(data['times'],np.sort(np.unique(data['times']))),'non-increasing sample times')
    require(data['candidate'].shape==data['reference'].shape,'shape mismatch')
    require(data['candidate'].shape[1]==len(data['times']) and data['candidate'].shape[0]==len(data['scale']),'layout mismatch')
    require((data['scale']>0).all(),'invalid state scales')
    err=float(np.max(abs(data['candidate']-data['reference'])/(data['scale'][:,None]+abs(data['reference']))))
    ref=float(np.max(abs(data['reference_coarse']-data['reference'])/(data['scale'][:,None]+abs(data['reference']))))
    return err,ref

def validate_record(data,stored,limit):
    err,ref=errors(data);require(stored['limit']==limit,'criterion changed')
    require(ref<=1e-7,'unconverged reference')
    if 'eligible_trajectory' in stored:require(stored['eligible_trajectory']==(err<=limit),'trajectory flag contradicts arrays')
    require(np.isclose(err,stored['global_scaled_error'],rtol=1e-12,atol=0),'reported error contradicts arrays')
    return {'error':err,'reference_refinement':ref,'eligible':err<=limit}

def small():
    plan=load(H/'PLAN.json');rows=[];records=load(H/'validation_05/RESULT.json')['models']
    require(len(records)==8,'missing model/mode/algorithm condition')
    seen=set()
    for r in records:
        key=(r['name'],r['algorithm'],r['mode']);require(key not in seen,'duplicate condition');seen.add(key)
        d=arrays(H/'validation_05'/('_'.join(key)+'.npz'));limit=plan['modes'][r['mode']]['global_scaled_error']
        rows.append({'name':r['name'],'algorithm':r['algorithm'],'mode':r['mode'],**validate_record(d,r,limit)})
    expected={(name,algorithm,mode) for name in ['hh','recurrent'] for algorithm in ['A','C'] for mode in ['fast','precise']}
    require(seen==expected,'model/mode condition mismatch')
    exrows=[];records=load(H/'extension_02/RESULT.json')['models'];require(len(records)==16,'extension conditions missing')
    for r in records:
        d=arrays(H/'extension_02'/f'seed{r["seed"]}_{r["algorithm"]}_{r["mode"]}.npz');limit=plan['modes'][r['mode']]['global_scaled_error']
        metric=validate_record(d,r,limit)
        # Extension descriptor has chemistry last, 12 species * 12 cells.
        total=d['candidate'][-144:].reshape(12,12,-1).sum(axis=0)
        conservation=float(np.max(abs(total-1)));require(conservation<=1e-7,'chemical conservation violation')
        require(np.allclose(total,d['chemical_total'],rtol=0,atol=1e-14),'chemical summary corrupted')
        exrows.append({'seed':r['seed'],'algorithm':r['algorithm'],'mode':r['mode'],**metric,'conservation_absolute':conservation})
    return rows,exrows

def large():
    root=H/'large_02';plan=load(H/'PLAN.json');rows=[]
    coarse=arrays(root/'reference/trajectory.npz');fine=arrays(root/'reference_fine/trajectory.npz')
    meta=load(root/'reference_fine/RESULT.json')
    require(meta['status']=='COMPLETE' and meta['simulated_actual_s']==1 and meta['states']==726900 and meta['connections']==23296700,'large reference scope incomplete')
    def compare(d):
        for key in ('times','probes','scale'):require(np.array_equal(d[key],fine[key]),'large context/layout mismatch: '+key)
        require(d['trace'].shape==(4096,21) and d['final'].shape==(726900,),'large array shape')
        t=float(np.max(abs(d['trace']-fine['trace'])/(fine['scale'][fine['probes'],None]+abs(fine['trace']))))
        f=float(np.max(abs(d['final']-fine['final'])/(fine['scale']+abs(fine['final']))))
        return max(t,f)
    referr=compare(coarse);require(referr<=1e-7,'large reference not refined enough')
    for algorithm in ('A','C'):
        for mode in ('fast','precise'):
            role=algorithm+'_'+mode;m=load(root/role/'RESULT.json');d=arrays(root/role/'trajectory.npz')
            require(m['model_sha256']==meta['model_sha256'] and m['simulated_actual_s']==1 and m['status']=='COMPLETE','candidate model/scope mismatch')
            err=compare(d);limit=plan['modes'][mode]['global_scaled_error']
            total=d['final'][-24000:].reshape(12,2000).sum(axis=0);conservation=float(np.max(abs(total-1)))
            require(conservation<=1e-7,'large conservation fails')
            require(np.isfinite(m['advance_including_scans_s']) and m['advance_including_scans_s']>0,'invalid wall measurement')
            rows.append({'algorithm':algorithm,'mode':mode,'error':err,'limit':limit,'eligible_for_this_workload':err<=limit,
                         'advance_including_scans_s':m['advance_including_scans_s'],'setup_s':m['setup_s'],'whole_worker_s':m['total_worker_s'],
                         'steps':m['steps'],'device_pool_GiB':m['device_pool_GiB'],'peak_RSS_GiB':m['peak_RSS_GiB'],'conservation_absolute':conservation})
    return {'scope':'one synthetic continuously coupled ODE system; not complete fly or biological validation','states':726900,'connections':23296700,
            'reference_refinement':referr,'accuracy_coverage':'4096 probes at 21 sample times plus all 726900 states at final time; no continuous global bound','rows':rows}

def corruptions():
    r=load(H/'validation_05/RESULT.json')['models'][0];d=arrays(H/'validation_05'/f'{r["name"]}_{r["algorithm"]}_{r["mode"]}.npz');detected=[]
    for kind in ('criterion','flag','state','time'):
        altered=copy.deepcopy(d);record=copy.deepcopy(r)
        if kind=='criterion':record['limit']*=10
        if kind=='flag':record['eligible_trajectory']=not record['eligible_trajectory']
        if kind=='state':altered['candidate'][0,0]+=1
        if kind=='time':altered['times'][1]=altered['times'][0]
        try:validate_record(altered,record,1e-5)
        except ValueError:detected.append(kind)
        else:raise RuntimeError('corruption escaped: '+kind)
    return detected

def main():
    freeze_check();s,e=small();l=large();c=corruptions()
    result={'core_matches_registered_repair':True,'small':s,'extensions':e,'large':l,'corruption_tests_detected':c,
            'classification':'PROMETEDOR_NO_CONFIRMADO','C_HH':'DESCARTADO en ambos perfiles según límites prospectivos; no se descarta toda familia exponencial',
            'complete_brain_speed_or_stage3_pass':False}
    path=Path(sys.argv[1]) if len(sys.argv)>1 else H/'VERIFIED.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'large':l,'classification':result['classification'],'corruptions':c},indent=2))
if __name__=='__main__':main()
