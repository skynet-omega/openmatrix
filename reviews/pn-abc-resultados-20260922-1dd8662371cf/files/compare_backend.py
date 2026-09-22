from pathlib import Path
import sys,time,json,argparse,hashlib,resource
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from portable import PortablePN,read,errors
from compact_step import advance_compact

def write_state(stem,state):
 arrays={}
 def encode(x):
  if isinstance(x,np.ndarray):
   key='a'+str(len(arrays));arrays[key]=x;return {'__array__':key}
  if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
  if isinstance(x,list):return [encode(v) for v in x]
  return x
 stem.with_suffix('.json').write_text(json.dumps(encode(state),indent=2)+'\n');np.savez_compressed(stem.with_suffix('.npz'),**arrays)

def main():
 arg=argparse.ArgumentParser();arg.add_argument('--out',required=True);arg.add_argument('--route',choices=['A','B','C'],default='A');args=arg.parse_args()
 out=H/args.out;out.mkdir(exist_ok=False);contract=json.loads((H/'CONTRACT.json').read_text());p=PortablePN(H/'capture_01')
 if args.route=='A':candidate=lambda dt,cur,**opt:advance_compact(p,dt,cur,_local_channel=p.calcium_port,**opt)
 elif args.route=='B':
  from backend_integration import install
  fast_solver,original_solver,integration_stats=install(p)
  def candidate(dt,cur,**opt):
   p.backend.solve=fast_solver
   try:return p.advance(dt,cur,**opt)
   finally:p.backend.solve=original_solver
 else:
  from compiled_step import advance_compiled
  candidate=lambda dt,cur,**opt:advance_compiled(p,dt,cur,_local_channel=p.calcium_port,**opt)
 def check(a,b):
  e=errors(a,b);passed=True
  for key,v in e.items():
   limit=contract['voltage_abs_mV'] if key=='/voltage' else contract['charge_abs_pC'] if 'charge' in key or key=='/charge' else contract['probability_abs'] if 'gate' in key else contract['other_numeric_abs']
   passed=passed and v<=limit
  return {'max_state_errors':e,'within_contract':passed}
 result={'route':args.route,'scope':'Prescribed real PN inputs; not body feedback','contract_sha256':hashlib.sha256((H/'CONTRACT.json').read_bytes()).hexdigest(),'cases':[]}
 for i in range(2):
  case=read(H/f'capture_01/case_{i:02d}');f=case['inputs'];ticks={}
  for name,fn in [('parent',p.advance),('candidate',candidate)]:
   measurements=[]
   for trial in range(4):
    p.restore(case['before']);start=time.perf_counter();r=fn(f['dt_ns'],f['current'],**f['options']);elapsed=time.perf_counter()-start
    if not r['accepted']:
     (out/f'failure_{i}_{name}_{trial}.json').write_text(json.dumps(r,indent=2)+'\n');raise ArithmeticError('Rejected '+str(out))
    if trial:measurements.append(elapsed)
   ticks[name]=float(np.median(measurements));state=p.state();write_state(out/f'case_{i}_{name}',state)
   (out/f'case_{i}_{name}_report.json').write_text(json.dumps(r,indent=2)+'\n')
   if name=='parent':parent=state
  row={'case':i,'median_wall_s':ticks,**check(parent,state)};result['cases'].append(row);print(json.dumps(row),flush=True)
 # Same64stage inputs, evolved independently from the exact same initial PN state.
 for name,fn in [('parent',p.advance),('candidate',candidate)]:
  p.restore(read(H/'capture_01/initial'));inputs=[read(H/f'capture_01/input_{i:03d}') for i in range(64)]
  start=time.perf_counter();reports=[]
  for f in inputs:
   r=fn(f['dt_ns'],f['current'],**f['options']);reports.append(r)
   if not r['accepted']:
    (out/f'failure_{name}_replay.json').write_text(json.dumps(reports,indent=2)+'\n');raise ArithmeticError('Replay rejected '+name)
  elapsed=time.perf_counter()-start;state=p.state();write_state(out/f'replay_{name}',state)
  (out/f'replay_{name}_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
  if name=='parent':parent=state;parentwall=elapsed
  else:result['replay']={'steps':64,'simulated_ns':state['time_ns']-read(H/'capture_01/initial')['time_ns'],'wall_s':{'parent':parentwall,'candidate':elapsed},'speedup':parentwall/elapsed,**check(parent,state)}
 if args.route=='B':result['integration_statistics']=integration_stats
 result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['material_speedup']=result['replay']['speedup']>=contract['minimum_material_speedup']
 result['all_state_checks']=all(x['within_contract'] for x in result['cases']) and result['replay']['within_contract']
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['replay'],indent=2))
if __name__=='__main__':main()
