"""Recompute the prospective event-scope comparison from full raw snapshots."""
from pathlib import Path
import sys,json,hashlib,argparse
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'transient_localization_20260922'))
from compare_candidates import brain_comparison
from qualify_transport import qualify

def compare(ms):
 plan=R/'PLAN.json'
 if hashlib.sha256(plan.read_bytes()).hexdigest()!= (R/'PLAN.sha256').read_text().split()[0]:raise ValueError('Changed prospective plan')
 a,b=[R/f'{name}{ms}_01' for name in ('baseline','scoped')];perf={}
 for folder in (a,b):
  r=json.loads((folder/'RESULT.json').read_text());times=[json.loads(x) for x in (folder/'TIMES.jsonl').read_text().splitlines()]
  if r['status']!='COMPLETE' or r['ms']!=ms or times!=r['times'] or [x['step'] for x in times]!=list(range(1,ms+1)):raise ValueError('Run incomplete/inconsistent')
  for path,digest in json.loads((folder/'FROZEN.json').read_text()).items():
   if hashlib.sha256((folder/'executed_sources'/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source corrupted')
  advance=sum(x['advance_s'] for x in times);cell=r['runtime']['cell']['native_wall_s'];pn=r['runtime']['PN']['wall_s']
  perf[folder.name]={'advance_s':advance,'wall_s':r['wall_s'],'observation_s':sum(x['observation_s'] for x in times),'membrane_s':cell,'PN_s':pn,'remainder_s':advance-cell-pn,'CNS':r['runtime']['CNS'],'event_policy':r['effective_event_policy'],'simulated_ms':ms,'ratio_to_operational_time_budget':advance/(ms*.06)}
 ac,bc=[json.loads((f/'EVENT_CONTRACT.json').read_text()) for f in (a,b)]
 if ac!=bc or ac['actual_full_RHS_probe_errors']!=[0.,0.,0.]:raise ValueError('Event scope changed/contradicted')
 snap={str(i):brain_comparison(a/f'brain_{i:02d}ms',b/f'brain_{i:02d}ms') for i in (1,5,20,50) if i<=ms}
 endpoint=qualify(a,b);limit=json.loads(plan.read_text())['criteria']['normalized_state_limit']
 short=endpoint['historical_partial_screen']['screen_pass'] and all(v['normalized_state_max_abs']<=limit and not any(v[x] for x in ('changed_discrete_fields','layout_differences','nonfinite')) for v in snap.values())
 speedup=perf[a.name]['advance_s']/perf[b.name]['advance_s']
 result={'ms':ms,'snapshots':snap,'endpoint':endpoint,'performance':perf,'complete_advance_speedup':speedup,'prospective_short_screen':bool(short),'material_speedup':speedup>=json.loads(plan.read_text())['criteria']['material_speedup_complete_advance'],'full_state_qualified':endpoint['full_state_qualified'],'stage3_admission':False,'scope':'Same corrected event transport, complete organism sham. Continuity does not guarantee temporal accuracy; exposed development snapshots, not biological or long-time admission.'}
 (R/f'COMPARISON_{ms}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'ms':ms,'short_screen':short,'speedup':speedup,'snapshots':{k:{f:v[f] for f in ('normalized_state_max_abs','gate_max_abs','voltage_max_abs_mV','changed_discrete_fields')} for k,v in snap.items()},'endpoint_screen':endpoint['historical_partial_screen'],'uncovered_fields':len(endpoint['uncovered_changed_numeric_fields']),'performance':perf},indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('ms',type=int,choices=[20,50]);compare(p.parse_args().ms)
