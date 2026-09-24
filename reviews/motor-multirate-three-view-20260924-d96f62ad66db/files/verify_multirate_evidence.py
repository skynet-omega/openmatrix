"""Rebuild factual trace/cache summaries from the raw local NPZ and frozen receipts.

No organism is rerun; this does not validate interval arithmetic, effective
nonlinear coefficients, recurrence, behavioral equivalence, or GPU speed.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE/'trace_real_rhs_1ms_03/base_csr_rhs_first64.npz'
BASE=HERE/'fusion_base_1ms_01/final_state/session.npz'

def need(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(path.read_text())

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def close(a,b):return abs(float(a)-float(b))<=max(1e-12,1e-12*max(abs(float(a)),abs(float(b))))

def verify_trace(plan,receipt,raw_hash,base_state_hash,times,rhs,weights,indices,degree):
    need(receipt['status']=='COMPLETE_SHADOW_ONLY' and receipt['simulated_trial_ms']==1,'trace status')
    need(receipt['plan_sha256']==sha_file(HERE/'TRACE_PLAN_01.json'),'trace plan identity')
    need(receipt['source_sha256']==sha_file(HERE/'trace_real_rhs_1ms.py'),'trace source identity')
    need(receipt['capsule_sha256']==raw_hash and receipt['capsule_bytes']==RAW.stat().st_size,'trace capsule identity')
    need(receipt['final_state_sha256']==base_state_hash==plan['identity']['final_state_sha256'],'parent final state')
    need(receipt['captured_rhs']==len(times)==rhs.shape[0]==plan['budget']['captured_rhs_max'],'trace frame count')
    need(receipt['trace_count']==12+6*(receipt['runtime_counts']['CNS']['accepted']+
                                      receipt['runtime_counts']['CNS']['rejected']),'trace evaluated stages')
    for group,expected in (('CNS',plan['baseline_cns']),):
        for field in ('accepted','rejected','epochs'):
            need(receipt['runtime_counts'][group][field]==expected[field],'baseline CNS '+field)
    need(receipt['runtime_counts']['events']['events']==plan['baseline_cns']['events'],'baseline events')
    backward=int(np.count_nonzero(np.diff(times)<0))
    need(backward==receipt['timestamp_backward_pairs'],'backward time pairs')
    need(close(times[0],receipt['timestamp_first_s']) and close(times[-1],receipt['timestamp_last_s']),'trace endpoint times')
    need(int(np.unique(times).size)==receipt['timestamp_unique'],'trace unique times')
    need(sha_array(weights)==plan['identity']['graph_weights_sha256'],'trace weights')
    need(sha_array(indices)==plan['identity']['graph_indices_sha256'],'trace indices')
    need(int(degree.sum())==len(indices),'edge mass')
    return {'frames':len(times),'backward_time_pairs':backward,
            'unique_times':int(np.unique(times).size),'max_time_s':float(times.max())}

def verify_deltas(report,raw_hash,times,rhs,degree,edges):
    need(report['input_sha256']==raw_hash and report['frames']==len(times),'delta provenance')
    need(report['backward_time_pairs']==int(np.count_nonzero(np.diff(times)<0)),'delta backwards')
    need(len(report['consecutive'])==4,'delta thresholds')
    for row,threshold in zip(report['consecutive'],(0,1e-12,1e-9,1e-6)):
        need(row['abs_cutoff']==threshold,'delta threshold')
        changed=np.abs(rhs[1:]-rhs[:-1])>threshold
        source_counts=changed.sum(axis=1)
        edge_counts=changed@degree
        for key,value in (('changed_sources_min',int(source_counts.min())),
                          ('changed_sources_median',float(np.median(source_counts))),
                          ('changed_sources_max',int(source_counts.max())),
                          ('edge_mass_min',float(edge_counts.min()/edges)),
                          ('edge_mass_median',float(np.median(edge_counts)/edges)),
                          ('edge_mass_max',float(edge_counts.max()/edges))):
            need(close(row[key],value),'delta '+key)
    return {'median_exact_edge_mass':report['consecutive'][0]['edge_mass_median'],
            'median_1e_minus_6_edge_mass':report['consecutive'][3]['edge_mass_median']}

def verify_cache(result,raw_hash,plan):
    need(result['status']=='COMPLETE_SHADOW_ONLY' and result['input_sha256']==raw_hash,'cache input/status')
    need(result['code_sha256']==sha_file(HERE/'chatgpt_temporal_cache_probe.py'),'cache source identity')
    r=result['result'];rows=r['rows']
    need(len(rows)==r['queries']==64,'cache queries')
    need(sum(not row['reuse'] for row in rows)==r['candidate_full_products'],'cache full count')
    need(r['one_time_norm_passes']==1,'cache setup count')
    reduction=len(rows)/(r['candidate_full_products']+r['one_time_norm_passes'])
    need(close(reduction,r['full_pass_reduction_including_setup']),'cache reduction')
    gate=reduction>=plan['followup_cache_work_gate_minimum']
    need(r['order10_work_gate']==gate,'cache gate')
    need(r['candidate_wall_s']>r['reference_RN_wall_s']>0,'cache measured wall negative')
    need(all(row['radius_max']>=row['error_vs_full_FP64']>=0 for row in rows),'cache observed error')
    return {'full_products':r['candidate_full_products'],'pass_reduction':reduction,
            'work_gate':gate,'candidate_wall_s':r['candidate_wall_s'],
            'reference_wall_s':r['reference_RN_wall_s']}

def main():
    plan=read(HERE/'TRACE_PLAN_01.json')
    # Convert the plan's prose gate to an explicit threshold for recomputation.
    need(plan['followup_cache_work_gate'].startswith('at least 10x'),'Unexpected plan gate')
    plan['followup_cache_work_gate_minimum']=10
    receipt=read(HERE/'trace_real_rhs_1ms_03/RESULT.json')
    delta=read(HERE/'TRACE_DELTA_01.json')
    cache=read(HERE/'chatgpt_temporal_cache_real_01/RESULT.json')
    raw_hash=sha_file(RAW)
    with np.load(BASE,allow_pickle=False) as saved:
        base_state_hash=sha_array(saved['array_261'])
    with np.load(RAW,allow_pickle=False) as z:
        times=z['query_s'];rhs=z['rhs'];weights=z['weights'];indices=z['indices']
        n=rhs.shape[1]
        need(n==166700 and z['indptr'].shape==(n+1,) and len(indices)==len(weights)==25582938,'CSR shape')
        need(np.isfinite(rhs).all() and np.isfinite(times).all(),'finite trace')
        degree=np.bincount(indices,minlength=n)
        trace_summary=verify_trace(plan,receipt,raw_hash,base_state_hash,times,rhs,weights,indices,degree)
        delta_summary=verify_deltas(delta,raw_hash,times,rhs,degree,len(indices))
    cache_summary=verify_cache(cache,raw_hash,plan)
    corruptions=0
    bad=copy.deepcopy(cache);bad['result']['candidate_full_products']=2
    try:verify_cache(bad,raw_hash,plan)
    except ValueError:corruptions+=1
    bad=copy.deepcopy(receipt);bad['timestamp_backward_pairs']=0
    with np.load(RAW,allow_pickle=False) as z:
        try:verify_trace(plan,bad,raw_hash,base_state_hash,z['query_s'],z['rhs'],z['weights'],z['indices'],
                         np.bincount(z['indices'],minlength=166700))
        except ValueError:corruptions+=1
    need(corruptions==2,'Corruption controls')
    out={'schema':'motor_multirate_raw_trace_verify_v1','status':'PASS',
         'optimized_python':bool(sys.flags.optimize),
         'trace':trace_summary,'delta':delta_summary,'cache':cache_summary,
         'corruptions_rejected':corruptions,
         'scope':'Raw capsule/receipt arithmetic and identities; no GPU output, interval proof, or organism rerun'}
    path=HERE/('VERIFY_MULTIRATE_OPTIMIZED.json' if sys.flags.optimize else 'VERIFY_MULTIRATE.json')
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
