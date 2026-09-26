"""Recompute functional and numerical evidence from saved runs, CPU only."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
from comparison_math import array_error,load_arrays,saved_tree_errors
HERE=Path(__file__).resolve().parent

def need(ok,message):
    if not ok:raise ValueError(message)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event_difference(left,right,ms):
    a=json.loads(left.read_text())[:16*ms];b=json.loads(right.read_text())[:16*ms]
    need(len(a)==len(b)==16*ms,'Incomplete event blocks')
    out={k:{'control_count':0,'candidate_count':0,'blocks_with_identity_difference':0,
            'max_time_difference_s_where_order_matches':0.} for k in ('committed','predictor')}
    out['first_identity_difference']=None
    origin=a[0]['start_elapsed_ns']
    total={k:[collections.Counter(),collections.Counter()] for k in ('committed','predictor')}
    for i,(x,y) in enumerate(zip(a,b)):
        kind='predictor' if i%2==0 else 'committed';duration=62500 if i%2==0 else 125000
        start=origin+(i//2)*125000
        need(x['start_elapsed_ns']==y['start_elapsed_ns']==start and
             x['duration_ns']==y['duration_ns']==duration,'Event-block clocks differ')
        sig=[]
        for z in (x,y):
            for e in z['events']:
                need(np.isfinite(e['time_s']) and 0<=e['time_s']<=duration*1e-9,
                     'Invalid event time')
                need(np.isfinite(e['jump']) and (e['post_q'] is None or np.isfinite(e['post_q'])),
                     'Invalid event payload')
            sig.append([(e['row'],e['neuron_id'],e['producer']) for e in z['events']])
        sx,sy=map(collections.Counter,sig)
        total[kind][0].update(sx);total[kind][1].update(sy)
        out[kind]['control_count']+=len(sig[0]);out[kind]['candidate_count']+=len(sig[1])
        if sx!=sy:
            out[kind]['blocks_with_identity_difference']+=1
            if out['first_identity_difference'] is None:
                out['first_identity_difference']={'kind':kind,'block':i,'ms':(start-origin)/1e6,
                    'control_only':[list(e) for e in (sx-sy).elements()],
                    'candidate_only':[list(e) for e in (sy-sx).elements()]}
        if sig[0]==sig[1]:
            d=max((abs(e['time_s']-f['time_s']) for e,f in zip(x['events'],y['events'])),default=0.)
            out[kind]['max_time_difference_s_where_order_matches']=max(d,out[kind]['max_time_difference_s_where_order_matches'])
    for kind,(x,y) in total.items():
        out[kind]['same_total_identity_counts']=x==y
        out[kind]['unmatched_control_total']=sum((x-y).values())
        out[kind]['unmatched_candidate_total']=sum((y-x).values())
    out['scope']='Committed events separated from discarded predictors; time metric only where ordered identity agrees.'
    return out

def compare(control,candidate,ms,*,paired_timing=False):
    plan=json.loads((HERE/'PLAN.json').read_text());limits=plan['comparison_limits_engineering_only']
    results=[]
    for folder in (control,candidate):
        r=json.loads((folder/'RESULT.json').read_text())
        need(r['status']=='COMPLETE' and r['completed_ms']>=ms and not r.get('cleanup_errors'),
             'Run incomplete or cleanup failure')
        initial=json.loads((folder/'INITIAL.json').read_text());need(initial['exact'],'Initial state not qualified')
        results.append(r)
    need(results[1]['plan_sha256']==digest(HERE/'PLAN.json'),'Candidate plan changed')
    need(results[1]['sources_sha256']==digest(HERE/'SOURCES.json'),'Candidate source inventory changed')
    if paired_timing:
        need(ms==100 and all(r['completed_ms']==ms for r in results),'Unequal paired duration')
        need(results[0]['plan_sha256']==results[1]['plan_sha256'] and
             results[0]['sources_sha256']==results[1]['sources_sha256'],'Pair provenance differs')
    need(results[0]['parameters']==results[1]['parameters'],'Numerical or biological parameters differ')
    s={k:v[:ms] for k,v in load_arrays(control/'traces.npz').items()}
    c={k:v[:ms] for k,v in load_arrays(candidate/'traces.npz').items()}
    need(set(s)==set(c),'Trace schema differs')
    need(np.array_equal(c['paso'],np.arange(1,ms+1)) and np.array_equal(s['paso'],c['paso']),
         'Missing or duplicated trace rows')
    fields={k:array_error(s[k],c[k]) for k in s}
    for key in ('CNS_time_ns','PN_time_ns','body_time_ns','DN_baseline','wind_torque_native'):
        need(fields[key]['exact'],'Clock/baseline/wind changed '+key)
    need(np.array_equal(s['sensores_usados'][0],c['sensores_usados'][0]),'Initial input differs')
    for name,z in (('control',s),('candidate',c)):
        need(np.array_equal(z['sensores_usados'][1:],z['sensores_pendientes'][:-1]),name+' sensor lag changed')
        need(np.array_equal(z['DN_q_usada'][1:],z['DN_q_actual'][:-1]),name+' motor lag changed')
    releases=('ORN_q_L','ORN_q_R','DN_q_actual','PN_general_transmission')
    gates={'finite_coherent_clocks':True,
           'release_bound':all(fields[k]['max_abs']<=limits['release_max_abs'] for k in releases),
           'position_bound':fields['position_mm']['max_abs']<=limits['position_max_abs_mm'],
           'yaw_bound':fields['yaw_delta_deg']['max_abs']<=limits['yaw_max_abs_deg'],
           'same_applied_commands':fields['motor_filter_applied_rad_s']['exact'],
           'same_contacts':fields['contact_active']['exact']}
    ns=load_arrays(control/'neural_states.npz');nc=load_arrays(candidate/'neural_states.npz')
    need(set(ns)==set(nc),'Neural field sets differ')
    neural={};pn={};published={}
    origin=int(s['CNS_time_ns'][0])-1000000
    for i,clock in enumerate(nc['time_ns']):
        step=(int(clock)-origin)//1000000
        if step>ms:continue
        ix=np.flatnonzero(ns['time_ns']==clock)
        if not len(ix):continue
        need(len(ix)==1,'Repeated neural timestamp');j=int(ix[0])
        neural[str(step)]={k:array_error(ns[k][j],nc[k][i]) for k in ns}
        for name,out in [('pn_state',pn),('published',published)]:
            left=control/f'state_{step}ms/{name}';right=candidate/f'state_{step}ms/{name}'
            if left.with_suffix('.json').exists() and right.with_suffix('.json').exists():
                out[str(step)]=saved_tree_errors(left,right)
    need(str(ms) in neural,'Missing final neural observation')
    if paired_timing:
        need('0' in neural and all(v['exact'] for v in neural['0'].values()),'Different recorded initial neurons/body')
        for tree in (pn,published):need('0' in tree and all(v['exact'] for v in tree['0'].values()),'Different initial PN/publication')
    timing={'candidate_advance_s':results[1]['advance_total_s'],'candidate_process_s':results[1]['wall_total_s'],
            'advance_minutes_per_simulated_second':results[1]['advance_total_s']/60*1000/ms,
            'process_minutes_per_simulated_second':results[1]['wall_total_s']/60*1000/ms,
            'scope':'100ms rates are extrapolations including repeated setup; only1000ms measures a full second.'}
    if paired_timing:
        timing.update(control_advance_s=results[0]['advance_total_s'],control_process_s=results[0]['wall_total_s'],
                      advance_fraction_saved=1-results[1]['advance_total_s']/results[0]['advance_total_s'],
                      process_fraction_saved=1-results[1]['wall_total_s']/results[0]['wall_total_s'])
        gates['useful_advance']=timing['advance_fraction_saved']>=plan['short_pair']['minimum_advance_fraction_saved']
        gates['process_no_regression']=timing['process_fraction_saved']>=0
    return {'status':'PASS' if all(gates.values()) else 'FAIL','ms':ms,'gates':gates,'limits':limits,
            'fields':fields,'neural':neural,'PN':pn,'published':published,
            'events':event_difference(control/'EVENTS.json',candidate/'EVENTS.json',ms),'timing':timing,
            'counts':{'control':results[0]['runtime']['CNS'],'candidate':results[1]['runtime']['CNS']},
            'sources':{str(folder/name):digest(folder/name) for folder in (control,candidate)
                       for name in ('RESULT.json','traces.npz','neural_states.npz','EVENTS.json')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--ms',type=int,required=True)
    p.add_argument('--paired-timing',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=compare(a.control.resolve(),a.candidate.resolve(),a.ms,paired_timing=a.paired_timing)
    need(not a.out.exists(),'Comparison output must be new')
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','ms','gates','timing','events','counts')},indent=2))

if __name__=='__main__':main()
