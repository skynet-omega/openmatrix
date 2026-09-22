"""Reconstruct numerical decisions from all stored trajectories and frozen tolerances."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(run,condition,spec,contract):
 p=run/(condition+'_'+spec['id']);metadata=json.loads((p/'timing.json').read_text())
 require(metadata['condition']==condition and metadata['spec']==spec,'Run context mismatch')
 require(metadata['duration_s']==contract['duration_s'] and metadata['scope']=='base recurrent q/s only','Scope/time mismatch')
 with np.load(p/'states.npz',allow_pickle=False) as z:t=z['time_s'];a=z['state']
 expected=np.arange(round(contract['duration_s']/contract['sample_s'])+1)*contract['sample_s']
 require(np.array_equal(t,expected),'Sample clocks differ')
 require(a.dtype==np.float64 and a.shape==(len(t),2*contract['neurons']),'Trajectory structure differs')
 require(np.isfinite(a).all() and metadata['finite'] is True,'Finiteness mismatch')
 require(float(a.min())==metadata['state_min'] and float(a.max())==metadata['state_max'],'Summary differs from arrays')
 require(metadata['max_implicit_residual']>=0 and np.isfinite(metadata['max_implicit_residual']),'Invalid residual')
 return a,metadata

def verify(run):
 run=Path(run);c=json.loads((HERE/'contract.json').read_text());f=json.loads((run/'freeze.json').read_text())
 require(sha(HERE/'contract.json')==f['contract_sha256'],'Frozen contract hash differs')
 for p,h in f['files'].items():require(sha(HERE.parent/p)==h,'Frozen source/input differs: '+p)
 b=c['bounds'];rows=[];reference_rows=[]
 for condition in c['conditions']:
  refs=[load(run,condition,{'id':'R'+str(i),'method':'rk4','h_s':h},c)[0] for i,h in enumerate(c['reference_steps_s'])]
  refinement=float(np.max(abs(refs[0]-refs[1])));require(refinement<=b['reference_max_qs'],'Reference not converged')
  reference_rows.append({'condition':condition,'max_qs_refinement':refinement})
  for spec in c['candidates']:
   a,m=load(run,condition,spec,c);require(np.array_equal(a[0],refs[1][0]),'Initial states differ')
   e=float(np.max(abs(a-refs[1])));domain=a.min()>=-b['domain_slack'] and a.max()<=1+b['domain_slack']
   ok=e<=b['candidate_max_qs'] and domain and m['max_implicit_residual']<=b['implicit_fixed_point_residual']
   rows.append({'condition':condition,'id':spec['id'],'max_qs_error':e,'pass':bool(ok),'domain_pass':bool(domain),'wall_s':m['wall_with_sampling_s'],'implicit_residual':m['max_implicit_residual'],'updated_edges':m['updated_edges']})
 eligible=[s['id'] for s in c['candidates'] if all(r['pass'] for r in rows if r['id']==s['id'])]
 winner=min(eligible,key=lambda name:sum(r['wall_s'] for r in rows if r['id']==name)) if eligible else None
 return {'classification':'PROMETEDOR_NO_CONFIRMADO' if winner else 'DESCARTADO','scope':c['scope'],'reference_refinement':reference_rows,'rows':rows,'eligible':eligible,'selected_base_only':winner,'bounds':b,'contract_sha256':f['contract_sha256'],'no_whole_organism_promotion':True}
if __name__=='__main__':
 result=verify(sys.argv[1]);s=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if len(sys.argv)>2:Path(sys.argv[2]).write_text(s)
 print(s)
