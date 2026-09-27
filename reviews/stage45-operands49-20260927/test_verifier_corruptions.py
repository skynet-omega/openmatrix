"""Independent review's real counterexamples plus consistent law corruption."""
from pathlib import Path
import json,hashlib,time,copy,argparse
from observation_codec import decode
import numpy as np
from verify_operands import verify,need
H=Path(__file__).resolve().parent
p=H/'observer_parity_02/operands_001ms.npz';start=time.process_time()
if p.exists():
 with np.load(p,allow_pickle=False) as z:base={k:z[k] for k in z.files}
 source_digest=hashlib.sha256(p.read_bytes()).hexdigest()
else:
 base,metadata=decode(p.with_name(p.stem+'.packed.npz'));source_digest=metadata['original_npz_sha256']
result=verify(base);need(len(result['margin_ranges'])==6,'All six cells')
cases={}
for case in ['pre_id','committed','gain','clock','transmission']+['law_'+str(c) for c in range(6)]:
 c=dict(base)
 key={'pre_id':'pre_ids','committed':'committed','clock':'start_ns','transmission':'operands'}.get(case,'records');c[key]=base[key].copy()
 if case=='pre_id':c[key][0]+=1
 elif case=='committed':c[key][0]=~c[key][0]
 elif case=='clock':c[key][0]+=1
 elif case=='gain':c[key][0,2*16+6]*=2
 elif case=='transmission':
  a=c[key][0,0];k=int(np.argmax(np.abs(a[:,0]*a[:,1]*a[:,2])));a[k,1]*=.5
 else:
  cell=int(case[4:]);r=c[key][:,:384].reshape(-1,4,6,16);x=r[0,0,cell];x[7]=x[11]=float(np.float32(.25));x[13]=x[12]*(x[11]-x[0])
 need(not np.array_equal(c[key],base[key]),'Mutation actually changes '+case)
 try:verify(c)
 except ValueError as exc:cases[case]=dict(rejected=True,error=str(exc))
 else:raise ValueError('Undetected corruption '+case)
report=dict(schema='operand49_corruption_v1',original_verified=result,cases=cases,CPU_s=time.process_time()-start,source_sha256=source_digest,verifier_sha256=hashlib.sha256((H/'verify_operands.py').read_bytes()).hexdigest(),new_neural_ms=0)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=H/'VERIFIER_CORRUPTIONS.json');args=parser.parse_args();f=args.output;need(not f.exists(),'Preserve evidence');f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
