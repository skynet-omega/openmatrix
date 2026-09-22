"""Reconstruct numerical decisions from contexts, arrays and frozen tolerances."""
from pathlib import Path
import json,hashlib,copy,sys
import numpy as np
from model import Model,require
from pilot import case,exact
from extensions import fixture
H=Path(__file__).resolve().parent
LIMIT={'fast':.01,'precise':1e-5}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    require(all(np.isfinite(v).all() for v in a.values()),'nonfinite raw evidence');return a
def small(name,profile,d,r):
    spec,end=case(name);m=Model(spec);t=np.linspace(0,end,21)
    require(r['case']==name and r['profile']==profile and r['model_identity']==m.identity,'small identity')
    require(np.array_equal(d['times'],t) and np.array_equal(d['scale'],m.scale),'small layout')
    require(d['candidate'].shape==(m.n,21),'incomplete small trajectory')
    truth=exact(name,t)
    if truth is None:
        rp=H/'references'/(name+'_reference')/'reference.npz';ref=arr(rp)
        require(sha(rp)==r['reference_sha256'] and np.array_equal(ref['times'],t),'small reference context')
        require(np.max(abs(ref['tight']-ref['loose'])/(m.scale[:,None]+abs(ref['tight'])))<=1e-7,'small reference unresolved');truth=ref['tight']
    require(np.array_equal(d['reference'],truth),'small reference content')
    require(np.array_equal(d['candidate'][:,0],m.initial),'small initial state')
    error=float(np.max(abs(d['candidate']-truth)/(m.scale[:,None]+abs(truth))))
    passed=bool(error<=LIMIT[profile]);require(r['failure'] is None and r['eligible'] is passed,'small flag contradiction')
    require(np.isclose(error,r['max_scaled_error'],rtol=1e-12,atol=0),'small error contradiction')
    if name=='clamp':require(np.array_equal(d['candidate'][0],np.zeros(21)),'clamp changed')
    return {'case':name,'profile':profile,'error':error,'eligible':passed,'advance_scan_s':r['advance_scan_s'],'setup_s':r['setup_s']}
def context(a,b,complete=True):
    for k in ['probes','scale']:require(np.array_equal(a[k],b[k]),'large layout '+k)
    count=len(a['times']);require(np.array_equal(a['times'],b['times'][:count]),'large times')
    require(a['trace'].shape==(4096,count) and a['final'].shape==(726900,),'large shape')
    if complete:require(count==len(b['times']),'incomplete reference grid')
def metric(a,b,full=True):
    context(a,b,full);n=len(a['times'])
    v=float(np.max(abs(a['trace']-b['trace'][:,:n])/(b['scale'][b['probes'],None]+abs(b['trace'][:,:n]))))
    return max(v,float(np.max(abs(a['final']-b['final'])/(b['scale']+abs(b['final']))))) if full else v
def main():
    for fn,key in [('FREEZE.json','files'),('BLOCKS_PLAN.json','source_hashes'),('REFERENCE_PLAN.json','sources')]:
        f=read(H/fn)
        for p,s in f.get(key,{}).items():require(sha(H/p)==s,'changed frozen source '+p)
    plan=read(H/'PLAN.json');require(plan['precision']['external_global_limits']==LIMIT and plan['precision']['reference_refinement']==1e-7,'criteria changed')
    smallrows=[]
    for name in ['clamp','stiff','hh','mixed']:
        for profile in ['fast','precise']:
            folder=H/'runs'/(name+'_'+profile);smallrows.append({'run':folder.name}|small(name,profile,arr(folder/'trajectory.npz'),read(folder/'RESULT.json')))
    for profile in ['fast','precise']:
        folder=H/'runs'/('hh_python_'+profile);smallrows.append({'run':folder.name}|small('hh',profile,arr(folder/'trajectory.npz'),read(folder/'RESULT.json')))
    fine=arr(H/'references/large_fine/trajectory.npz');coarse=arr(H/'references/large_coarse/trajectory.npz')
    oldref=metric(coarse,fine);require(oldref<=1e-7,'one-second reference unresolved')
    rk8=arr(H/'runs/ten_reference_erk/trajectory.npz');rk4=arr(H/'runs/ten_reference/trajectory.npz');partial=arr(H/'runs/ten_reference_fine/trajectory.npz')
    refinement=metric(rk4,rk8);require(refinement<=1e-7,'ten-second reference unresolved')
    partialerr=metric(partial,rk8,False);require(partialerr<=1e-7,'partial tight RK4 mismatch')
    ii=[int(np.argmin(abs(fine['times']-t))) for t in rk8['times'][:5]];require(np.array_equal(fine['times'][ii],rk8['times'][:5]),'CPU overlap times')
    cpu_overlap=float(np.max(abs(rk8['trace'][:,:5]-fine['trace'][:,ii])/(fine['scale'][fine['probes'],None]+abs(fine['trace'][:,ii]))));require(cpu_overlap<=1e-7,'CPU reference overlap mismatch')
    spec=fixture((90000,74700,2000),seed=0);spec['connections'][0]['offsets']=list(range(1,257));spec['connections'][0]['weight']=.2/256;m=Model(spec)
    require(m.n==726900 and m.connection.nnz==23296700,'large dimensions')
    for d in [fine,rk8]:require(np.array_equal(d['scale'],m.scale) and np.array_equal(d['trace'][:,0],m.initial[d['probes']]),'large model initial/scale')
    large=[]
    for folder in sorted((H/'runs').iterdir()):
        if not (folder.name.startswith('one_') or folder.name.startswith('ten_native')):continue
        d=arr(folder/'trajectory.npz');ref=fine if folder.name.startswith('one_') else rk8
        profile='fast' if folder.name.endswith('fast') else 'precise';complete=np.array_equal(d['times'],ref['times'])
        e=metric(d,ref,complete);rp=folder/'RESULT.json';r=read(rp) if rp.exists() else None
        if r:
            require(r['model_identity']==m.identity and r['descriptor_hash']==m.descriptor_hash and r['profile']==profile,'large identity')
            require(r['completed'] is bool(complete and r['failure'] is None),'large completion flag')
            if complete:require(np.isclose(e,r['error'],rtol=1e-12,atol=0) and r['eligible'] is bool(e<=LIMIT[profile]),'large error/flag contradiction')
        else:require(folder.name=='ten_native_fast','unexpected missing result')
        conservation=float(np.max(abs(d['final'][-24000:].reshape(12,2000).sum(axis=0)-1)))
        require(conservation<=1e-7,'conserved chemistry changed')
        large.append({'run':folder.name,'profile':profile,'completed_grid':bool(complete),'last_saved_sample_s':float(d['times'][-1]),'error':e,'error_scope':'probes plus all final states' if complete else 'saved probes only; partial final time differs from grid','eligible':bool(complete and e<=LIMIT[profile] and (r is None or r['failure'] is None)),'advance_scan_s':None if r is None else r['advance_scan_s'],'setup_s':None if r is None else r['setup_s'],'failure':None if r is None else r['failure'],'conservation_error':conservation,'provenance':'raw trajectory recovered after original reference-grid failure; original exit1 preserved, missing solver statistics not reconstructed' if r is None else 'raw arrays plus original execution record'})
    detected=[];f=H/'runs/clamp_precise';data=arr(f/'trajectory.npz');record=read(f/'RESULT.json')
    for kind in ['flag','time','state','identity','reference']:
        d=copy.deepcopy(data);r=copy.deepcopy(record)
        if kind=='flag':r['eligible']=not r['eligible']
        if kind=='time':d['times'][1]=d['times'][0]
        if kind=='state':d['candidate'][1,3]+=1
        if kind=='identity':r['model_identity']='0'*64
        if kind=='reference':d['reference'][1,1]+=.1
        try:small('clamp','precise',d,r)
        except ValueError:detected.append(kind)
        else:raise RuntimeError('corruption escaped '+kind)
    ledger=read(H/'CAMPAIGN.json')+read(H/'FINISH_LEDGER.json')
    scientific=sum(bool(r.get('scientific',True)) for r in ledger)+1 # additional ERK8 reference
    require(scientific<=plan['limits']['science_runs'],'science budget exceeded')
    wall=sum(r['subprocess_wall_s'] for r in ledger)+read(H/'runs/ten_reference_erk/RESULT.json')['measured_window_s']
    require(wall<=plan['limits']['compute_s'],'compute budget exceeded')
    result={'classification':'PROMETEDOR_NO_CONFIRMADO','scope':'author-exposed synthetic engineering; stage3 open, no complete biological brain/body','small':smallrows,'large':large,'references':{'one_second_refinement':oldref,'ten_second_RK4_vs_RK8':refinement,'partial_tight_RK4_vs_RK8':partialerr,'first_second_CPU_overlap_probes':cpu_overlap,'ERK8':read(H/'runs/ten_reference_erk/RESULT.json')},'corruptions_detected':detected,'science_runs':scientific,'measured_process_windows_s':wall,'time_scope':'outer process time except ERK8 uses its recorded measured window; compilation and short checks separate'}
    out=Path(sys.argv[1]);require(not out.exists(),'do not overwrite verification');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
