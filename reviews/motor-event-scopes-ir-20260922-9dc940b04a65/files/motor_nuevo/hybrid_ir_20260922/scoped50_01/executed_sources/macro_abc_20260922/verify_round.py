"""Reconstruct short evidence from raw snapshots, frozen contract and time logs."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parent;T=R.parent/'transient_localization_20260922';sys.path.insert(0,str(T))
from compare_candidates import brain_comparison
from qualify_transport import qualify

def main():
 for name,sha in json.loads((R/'PROPOSALS_FROZEN.json').read_text()).items():
  if hashlib.sha256((R/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen input changed: '+name)
 perf={}
 for name in ('fine_reset_01','guard_reset_01'):
  p=R/name;r=json.loads((p/'RESULT.json').read_text());times=[json.loads(l) for l in (p/'TIMES.jsonl').read_text().splitlines()]
  if r['status']!='COMPLETE' or r['ms']!=20 or r['times']!=times or [x['step'] for x in times]!=list(range(1,21)):raise ValueError('Incomplete or inconsistent run')
  advance=sum(x['advance_s'] for x in times);cell=r['runtime']['cell']['native_wall_s'];pn=r['runtime']['PN']['wall_s']
  perf[name]={'simulated_ms':20,'advance_s':advance,'wall_s':r['wall_s'],'observation_s':sum(x['observation_s'] for x in times),'mean_s_per_simulated_ms':advance/20,'ratio_to_60_s_per_simulated_s_budget':advance/1.2,'membrane_s':cell,'PN_s':pn,'remainder_s':advance-cell-pn,'CNS':r['runtime']['CNS'],'zero_CNS_lower_bound_ratio':(cell+pn)/1.2,'scope':'Measured20ms only; lower bound assumes recorded sequential component timers. Not1second measurement.'}
 fine,guard=R/'fine_reset_01',R/'guard_reset_01'
 snapshots={str(i):brain_comparison(fine/f'brain_{i:02d}ms',guard/f'brain_{i:02d}ms') for i in (1,5,20)}
 repair={str(i):brain_comparison(T/'fine1562_20'/f'brain_{i:02d}ms',fine/f'brain_{i:02d}ms') for i in (1,5)}
 raw=np.load(R/'domain_01/failure_arrays.npz',allow_pickle=False);bad=np.flatnonzero(~np.isfinite(raw['fine'])|(raw['fine']<0)|(raw['fine']>1))
 if bad.tolist()!=raw['indices'].tolist():raise ValueError('Failure payload inconsistency')
 domain={'offending_coordinates':bad.tolist(),'values':raw['fine'][bad].tolist(),'all_captured_targets_valid':all(np.all((raw[f'stage_{i}_target']>=0)&(raw[f'stage_{i}_target']<=1)) for i in range(6)),'first_output_pre':raw['out_0_pre'].tolist(),'first_output_post':raw['out_0_post'].tolist()}
 inventory=json.loads((fine/'OPERATOR_INVENTORY.json').read_text());inventory['possible_overwritten_edge_fraction']=inventory['union_base_edges']/inventory['base_edges']
 result={'snapshots':snapshots,'transport_repair_vs_same_old_method':repair,'twenty_ms':qualify(fine,guard),'performance':perf,'original_failure':domain,'operator_inventory':inventory,'stage3_admission':False,'scope':'Engineering milestone only.20ms sham, not long-time reliability or biological performance. Historical partial criteria do not qualify every changed physical field.'}
 (R/'VERIFIED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'snapshots':{k:{f:v[f] for f in ('normalized_state_max_abs','gate_max_abs','voltage_max_abs_mV','changed_discrete_fields')} for k,v in snapshots.items()},'partial_screen':result['twenty_ms']['historical_partial_screen']['screen_pass'],'uncovered_fields':len(result['twenty_ms']['uncovered_changed_numeric_fields']),'repair_errors':{k:v['normalized_state_max_abs'] for k,v in repair.items()},'performance':perf},indent=2))
if __name__=='__main__':main()
