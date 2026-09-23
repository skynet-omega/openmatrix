from pathlib import Path
import sys,json,hashlib
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'causal_runtime_20260922'))
from run_storage import verify_snapshot
from operator_state import OperatorState,LEGACY_BINDINGS
sys.path.insert(0,str(H.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from verify_transport import verify,read_state,flatten
reports={}
for name in ('smoke_causal_01','smoke_reference_01'):
 run=H/name;r=json.loads((run/'RESULT.json').read_text());m=verify_snapshot(run/'final_state')
 if r['status']!='COMPLETE' or r['completed_trial_ms']!=1 or not r['smoke_only']:raise RuntimeError('Incomplete/mislabelled smoke')
 expected='causal_cuda' if 'causal' in name else 'reference_cuda'
 if r['runtime']['profile']!=expected or not r['runtime']['event_boundaries']:raise RuntimeError('Wrong installed engine')
 if expected=='causal_cuda' and r['runtime']['cell']['scheduler']!='device_independent_block':raise RuntimeError('Device controller not exercised')
 registry=OperatorState.from_state(read_state(run/'final_state/effective_operator'),expected_bindings=LEGACY_BINDINGS)
 reports[name]={'status':r['status'],'engine':expected,'snapshot_hashes_checked':len(m['files']),'operator_identity':registry.state_dict()['identity_sha256'],'full_body_resume_tested':False,'wall_s':r['wall_total_s']}
comparison=verify(H/'smoke_reference_01',H/'smoke_causal_01')
result={'smokes':reports,'historical_partial_comparison':comparison,'biological_admission':False}
(H/'SMOKE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'smokes':reports,'partial_comparison':comparison['screen_pass']}))
