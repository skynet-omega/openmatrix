from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;PN=HERE.parent/'resident_pn_20260922';P=HERE.parent/'pn_abc_20260922';sys.path[:0]=[str(PN),str(P)]
from resident import ResidentPN
from portable import read
import graph_step
candidate=ResidentPN(P/'capture_01');case=read(P/'capture_01/case_00');candidate.restore(case['before']);f=case['inputs'];options=f['options'].copy();options['max_newton']=1
called=[]
def forbidden(*args,**kw):called.append(True);raise RuntimeError('Two-correction graph exceeded one-correction budget')
graph_step.graph_stage=forbidden
result=graph_step.advance_graph_resident(candidate,f['dt_ns'],f['current'],_local_channel=candidate.calcium_port,**options)
if called:raise RuntimeError('Forbidden graph used')
for stage in result['stages']:
 if len(stage.get('history',[]))>2:raise RuntimeError('Too many correction iterations')
report={'accepted':result['accepted'],'two_correction_graph_called':False,'history_lengths':[len(x.get('history',[])) for x in result['stages']],'max_newton':1,'scope':'Actual captured PN state and input; low budget honored even when step rejects.'}
(HERE/'NEWTON_BUDGET_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
