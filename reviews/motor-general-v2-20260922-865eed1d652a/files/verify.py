"""Rebuild v2 decisions from frozen contexts and numeric evidence, including negative tests."""
from pathlib import Path
import json,hashlib,sys,copy
import numpy as np
from model import Model,require
from pilot import case,exact
from audit_previous import verify as audit_previous
H=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    require(all(np.isfinite(v).all() for v in d.values()),'nonfinite saved array');return d

def check_record(name,profile,d,r):
    spec,end=case(name);m=Model(spec);times=np.linspace(0,end,21)
    require(r['case']==name and r['profile']==profile and r['model_identity']==m.identity,'context identity')
    require(np.array_equal(d['times'],times) and np.array_equal(d['scale'],m.scale),'time/scale context')
    require(d['candidate'].shape==d['reference'].shape==(m.n,21),'incomplete/wrong trajectory layout')
    require(np.isfinite(d['candidate']).all() and np.isfinite(d['reference']).all(),'nonfinite data')
    truth=exact(name,times)
    if truth is None:
        rp=H/'pilots'/(name+'_reference')/'reference.npz';ref=arrays(rp)
        require(hashlib.sha256(rp.read_bytes()).hexdigest()==r['reference_sha256'],'reference source mismatch')
        delta=float(np.max(abs(ref['tight']-ref['loose'])/(m.scale[:,None]+abs(ref['tight']))));require(delta<=1e-7,'unresolved reference');truth=ref['tight']
    require(np.array_equal(d['reference'],truth),'reference differs from reconstructed source')
    require(np.array_equal(d['candidate'][:,0],m.initial),'initial condition changed')
    error=float(np.max(abs(d['candidate']-truth)/(m.scale[:,None]+abs(truth))))
    limit={'fast':.01,'precise':1e-5}[profile];passed=bool(error<=limit)
    require(r['failure'] is None and r['eligible'] is passed,'status contradicts trajectory')
    require(np.isclose(error,r['max_scaled_error'],rtol=1e-12,atol=0),'error summary contradicts arrays')
    if name=='clamp':require(np.array_equal(d['candidate'][0],np.zeros(21)),'clamped variable changed')
    if name=='mixed':
        total=d['candidate'][-144:].reshape(12,12,21).sum(axis=0);require(np.max(abs(total-1))<=1e-7,'chemical conservation')
    return {'case':name,'profile':profile,'error':error,'eligible':passed,'advance_scan_s':r['advance_scan_s'],'setup_s':r['setup_s']}

def large(folder,profile):
    r=read(folder/'RESULT.json');d=arrays(folder/'trajectory.npz');fine=arrays(H/'prior_evidence/reference_fine/trajectory.npz');coarse=arrays(H/'prior_evidence/reference/trajectory.npz')
    for k in ['times','probes','scale']:require(np.array_equal(d[k],fine[k]),'large layout/context: '+k)
    require(d['trace'].shape==(4096,21) and d['final'].shape==(726900,),'large incomplete shape')
    for p,h in r['reference_hashes'].items():require(hashlib.sha256((H/'prior_evidence'/p).read_bytes()).hexdigest()==h,'large reference provenance')
    def metric(a):return float(max(np.max(abs(a['trace']-fine['trace'])/(fine['scale'][fine['probes'],None]+abs(fine['trace']))),np.max(abs(a['final']-fine['final'])/(fine['scale']+abs(fine['final'])))))
    refinement=metric(coarse);require(refinement<=1e-7,'large reference refinement')
    error=metric(d);limit={'fast':.01,'precise':1e-5}[profile]
    require(r['profile']==profile and r['states']==726900 and r['connections']==23296700 and r['actual_simulated_s']==1,'large scope')
    require(r['completed'] is True and r['failure'] is None and r['eligible'] is bool(error<=limit),'large flags')
    require(np.isclose(error,r['error'],rtol=1e-12,atol=0),'large error summary')
    total=d['final'][-24000:].reshape(12,2000).sum(axis=0);require(np.max(abs(total-1))<1e-7,'large conservation')
    return {'profile':profile,'error':error,'eligible':error<=limit,'advance_scan_s':r['advance_scan_s'],'setup_s':r['setup_s'],'reference_refinement':refinement}

def main():
    freeze=read(H/'PILOT_FREEZE.json')['files']
    for p,h in freeze.items():require(hashlib.sha256((H/p).read_bytes()).hexdigest()==h,'unregistered numeric pilot source change: '+p)
    rows=[]
    for name in ['clamp','stiff','hh','mixed']:
        for profile in ['fast','precise']:
            folder=H/'pilots'/(name+'_'+profile);rows.append(check_record(name,profile,arrays(folder/'trajectory.npz'),read(folder/'RESULT.json')))
    detected=[];p=H/'pilots/clamp_precise';d=arrays(p/'trajectory.npz');r=read(p/'RESULT.json')
    for kind in ['flag','time','state','identity','reference']:
        dd=copy.deepcopy(d);rr=copy.deepcopy(r)
        if kind=='flag':rr['eligible']=not rr['eligible']
        if kind=='time':dd['times'][1]=dd['times'][0]
        if kind=='state':dd['candidate'][1,3]+=1
        if kind=='identity':rr['model_identity']='0'*64
        if kind=='reference':dd['reference'][1,1]+=.1
        try:check_record('clamp','precise',dd,rr)
        except ValueError:detected.append(kind)
        else:raise RuntimeError('escaped corruption '+kind)
    lr=[large(H/'large_fast_02','fast'),large(H/'large_precise_01','precise')]
    previous=audit_previous(H/'prior_evidence/extension_02')
    explicit=[]
    for r in read(H/'explicit_hh_01/RESULT.json'):
        d=arrays(H/'explicit_hh_01'/(r['profile']+'.npz'));ref=arrays(H/'pilots/hh_reference/reference.npz')
        require(np.array_equal(d['reference'],ref['tight']) and np.array_equal(d['times'],ref['times']),'explicit comparison reference')
        err=float(np.max(abs(d['candidate']-ref['tight'])/(ref['scale'][:,None]+abs(ref['tight']))));require(np.isclose(err,r['error'],rtol=1e-12,atol=0),'explicit error summary');explicit.append(r)
    result={'classification':'PROMETEDOR_NO_CONFIRMADO','scope':'author-exposed engineering only; no stage3 or complete biological brain','small':rows,'large':lr,'explicit_same_HH_comparison':explicit,'prior_cohort_count':len(previous),'corruptions_detected':detected}
    out=Path(sys.argv[1]) if len(sys.argv)>1 else H/'VERIFIED.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
