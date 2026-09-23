from pathlib import Path
import sys,json,time
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'pn_abc_20260922';sys.path[:0]=[str(H),str(P)]
from resident import ResidentPN
from portable import PortablePN,read,errors
from compare_pn import write_state
out=H/sys.argv[1];out.mkdir(exist_ok=False);parent=PortablePN(P/'capture_01');candidate=ResidentPN(P/'capture_01');case=read(P/'capture_01/case_00');f=case['inputs'];result={}
for name,p in [('parent',parent),('resident',candidate)]:
 p.restore(case['before']);start=time.perf_counter();r=p.advance(f['dt_ns'],f['current'],**f['options']);cold=time.perf_counter()-start
 (out/(name+'_first_report.json')).write_text(json.dumps(r,indent=2)+'\n')
 if not r['accepted']:raise ArithmeticError('Initial step rejected '+name)
 p.restore(read(P/'capture_01/initial'));inputs=[read(P/f'capture_01/input_{i:03d}') for i in range(64)];start=time.perf_counter();reports=[]
 for f in inputs:
  r=p.advance(f['dt_ns'],f['current'],**f['options']);reports.append(r)
  if not r['accepted']:
   (out/(name+'_failure.json')).write_text(json.dumps(reports,indent=2)+'\n');raise ArithmeticError('Replay rejected '+name)
 elapsed=time.perf_counter()-start;state=p.state();write_state(out/name,state);(out/(name+'_reports.json')).write_text(json.dumps(reports,indent=2)+'\n');result[name]={'cold_step_s':cold,'wall_s':elapsed}
 print(json.dumps({name:result[name]}),flush=True)
 if name=='parent':reference=state
result['errors']=errors(reference,state);result['speedup']=result['parent']['wall_s']/result['resident']['wall_s'];result['fallbacks']=candidate.backend.fallbacks;result['scope']='PN electrical resident, release chemistryCPU; prescribed1ms, not organism';(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
