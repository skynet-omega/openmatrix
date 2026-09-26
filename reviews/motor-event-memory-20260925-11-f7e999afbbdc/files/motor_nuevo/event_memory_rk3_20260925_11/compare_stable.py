"""Functional trajectory and recorded neural-state comparison, no GPU required."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
from comparison_math import array_error,load_arrays,saved_tree_errors

HERE=Path(__file__).resolve().parent

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def events_compare(ref,new,initial_elapsed_ns=1522000000,ms=1000):
    a=json.loads(ref.read_text())[:16*ms];b=json.loads(new.read_text())
    need(len(a)==len(b)==16*ms,'Incomplete midpoint event log')
    result={'committed':{'stable':0,'optimized':0,'blocks_with_different_identity':0},
            'predictor':{'stable':0,'optimized':0,'blocks_with_different_identity':0},
            'first_identity_difference':None,'scope':'Identities/counts; no assertion of identical event times'}
    for i,(x,y) in enumerate(zip(a,b)):
        kind='predictor' if i%2==0 else 'committed';duration=62500 if i%2==0 else 125000
        start=initial_elapsed_ns+(i//2)*125000
        need(x['start_elapsed_ns']==y['start_elapsed_ns']==start and
             x['duration_ns']==y['duration_ns']==duration,'Different event-block schedule')
        signature=lambda z:collections.Counter((v['row'],v['neuron_id'],v['producer']) for v in z['events'])
        sx,sy=signature(x),signature(y)
        result[kind]['stable']+=sum(sx.values());result[kind]['optimized']+=sum(sy.values())
        if sx!=sy:
            result[kind]['blocks_with_different_identity']+=1
            if result['first_identity_difference'] is None:
                result['first_identity_difference']={'kind':kind,'block_index':i,
                    'trial_start_ms':(start-initial_elapsed_ns)/1e6,
                    'missing':[list(v) for v in (sx-sy).elements()],
                    'additional':[list(v) for v in (sy-sx).elements()]}
    return result

def compare(run,*,prefix=False):
    ref=HERE.parent/'equivalence_1s_20260925_09/reference';plan=json.loads((HERE/'PLAN.json').read_text())
    for name,h in json.loads((ref/'MANIFEST.json').read_text()).items():
        need(sha(ref/name)==h,'Reference changed '+name)
    initial=json.loads((run/'INITIAL.json').read_text())
    need(initial['exact'],'Run did not restore exact initial state')
    result=None
    if not prefix:
        result=json.loads((run/'RESULT.json').read_text())
        need(result['status']=='COMPLETE' and result['completed_ms'] in (100,1000),'Incomplete 1s execution')
        need(result['plan_sha256']==sha(HERE/'PLAN.json'),'Plan changed after launch')
        need(not result.get('cleanup_errors'),'Cleanup failed')
    c=load_arrays(run/('traces_prefix.npz' if prefix else 'traces.npz'))
    s=load_arrays(ref/'traces.npz');n=len(c['paso'])
    need(1<=n<=1000 and (prefix or n==result['completed_ms']),'Unexpected sample count')
    s={k:v[:n] for k,v in s.items()}
    need(set(s)==set(c),'Trace fields differ')
    need(np.array_equal(c['paso'],np.arange(1,n+1)),'Missing/duplicated simulated interval')
    fields={k:array_error(s[k],c[k]) for k in s}
    for key in ('CNS_time_ns','PN_time_ns','body_time_ns','DN_baseline','wind_torque_native'):
        need(fields[key]['exact'],'Clock/baseline/wind changed '+key)
    need(np.array_equal(s['sensores_usados'][0],c['sensores_usados'][0]),'First neural input differs')
    need(np.array_equal(c['sensores_usados'][1:],c['sensores_pendientes'][:-1]),'Candidate sensor lag differs')
    need(np.array_equal(c['DN_q_usada'][1:],c['DN_q_actual'][:-1]),'Candidate DN lag differs')
    limits=plan['comparison_limits_engineering_only']
    releases=('ORN_q_L','ORN_q_R','DN_q_actual','PN_general_transmission')
    cmd=s['motor_filter_applied_rad_s']!=c['motor_filter_applied_rad_s']
    first=np.flatnonzero(cmd)
    command={'different_ticks':int(cmd.sum()),'first_difference':None,
             'integrated_abs_difference_deg':float(np.rad2deg(np.abs(s['motor_filter_applied_rad_s']-c['motor_filter_applied_rad_s']).sum()*.001))}
    onset={}
    for key in ('DN_q_actual','neural_command_raw_rad_s','motor_filter_state_rad_s',
                'motor_filter_applied_rad_s','position_mm','yaw_delta_deg'):
        exact_diff=np.any((s[key]!=c[key]).reshape(n,-1),axis=1)
        ids=np.flatnonzero(exact_diff)
        onset[key]={'first_exact_difference_ms':int(ids[0]+1) if len(ids) else None}
        bound={'DN_q_actual':limits['release_max_abs'],
               'position_mm':limits['position_max_abs_mm'],
               'yaw_delta_deg':limits['yaw_max_abs_deg']}.get(key)
        if bound is not None:
            ids=np.flatnonzero(np.max(np.abs(s[key]-c[key]).reshape(n,-1),axis=1)>bound)
            onset[key]['first_prospective_bound_exceeded_ms']=int(ids[0]+1) if len(ids) else None
    for name,z in (('stable',s),('optimized',c)):
        v=z['motor_filter_applied_rad_s']
        command[name+'_ticks']={'negative':int((v<0).sum()),'zero':int((v==0).sum()),'positive':int((v>0).sum())}
    if len(first):
        i=int(first[0]);command['first_difference']={'step_ms':i+1}
        for name,z in (('stable',s),('optimized',c)):
            command['first_difference'][name]={k:float(z[k][i]) for k in
                ('neural_command_raw_rad_s','motor_filter_state_rad_s','motor_filter_applied_rad_s')}
            command['first_difference'][name]['threshold_margin_rad_s']=float(.0005-abs(z['motor_filter_state_rad_s'][i]))
            command['first_difference'][name]['used_DN']=z['DN_q_usada'][i].tolist()
    gates={'all_finite_and_clocks_coherent':True,
           'releases_within_bound':all(fields[k]['max_abs']<=limits['release_max_abs'] for k in releases),
           'position_within_bound':fields['position_mm']['max_abs']<=limits['position_max_abs_mm'],
           'yaw_within_bound':fields['yaw_delta_deg']['max_abs']<=limits['yaw_max_abs_deg'],
           'same_applied_commands':not bool(cmd.any()),
           'same_contacts':fields['contact_active']['exact'],
           'same_forward_commands':fields['command_forward_mm_s']['exact']}
    neural={};pn={};published={}
    cs=load_arrays(run/('neural_prefix.npz' if prefix else 'neural_states.npz'))
    ss=load_arrays(ref/'neural_states.npz')
    need(set(cs)==set(ss),'Recorded neural field sets differ')
    for j,clock in enumerate(ss['time_ns']):
        ix=np.flatnonzero(cs['time_ns']==clock)
        if not len(ix):continue
        need(len(ix)==1,'Repeated neural timestamp');i=int(ix[0]);step=int((clock-44486000000)//1000000)
        neural[str(step)]={k:array_error(ss[k][j],cs[k][i]) for k in ss}
        pn[str(step)]=saved_tree_errors(ref/f'state_{step}ms/pn_state',run/f'state_{step}ms/pn_state')
        published[str(step)]=saved_tree_errors(ref/f'state_{step}ms/published',run/f'state_{step}ms/published')
    if not prefix:need(set(neural)==({str(k) for k in (100,1000) if k<=n}),'Missing neural checkpoints')
    out={'status':('PREFIX_WITHIN_SCREEN' if prefix else 'FUNCTIONAL_SCREEN_PASS') if all(gates.values()) else 'FUNCTIONAL_DISCREPANCY',
         'completed_ms':n,'initial_direct_comparison_passed':initial['exact'],
         'initial_comparison_method':initial['method'],
         'engineering_screen':gates,'limits':limits,
         'fields':fields,'motor_commands':command,'first_differences':onset,'neural_snapshots':neural,'PN_snapshots':pn,
         'published_snapshots':published,
         'stability':{name:{'min_contacts':int(z['contact_active'].sum(axis=1).min()),
                            'min_upright':float(z['upright'].min()),
                            'final_position_mm':z['position_mm'][-1].tolist(),
                            'final_yaw_delta_deg':float(z['yaw_delta_deg'][-1])} for name,z in (('stable',s),('optimized',c))},
         'scope':'One matched initial state and first second before wind; engineering compatibility, not biological equivalence or navigation'}
    if not prefix:
        out['events']=events_compare(ref/'EVENTS.json',run/'EVENTS.json',ms=n)
        history=json.loads((ref/'PROGRESS.json').read_text())
        out['timing']={'candidate_advance_s':result['advance_total_s'],
                       'candidate_process_s':result['wall_total_s'],
                       'candidate_advance_min_per_simulated_second':result['advance_total_s']/60*1000/n,
                       'candidate_process_min_per_simulated_second':result['wall_total_s']/60*1000/n,
                       'historical_observed_step_wall_s':sum(r['step_wall_s'] for r in history[:n]),
                       'historical_scope':'Different recording overhead and potentially resource contention; not controlled speedup'}
        out['raw_hashes']={str(p.relative_to(HERE)):sha(p) for p in
            [run/'traces.npz',run/'neural_states.npz',run/'RESULT.json',run/'EVENTS.json',HERE/'PLAN.json']}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=HERE/'candidate_1000ms_01')
    p.add_argument('--prefix',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=compare(a.run.resolve(),prefix=a.prefix)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','completed_ms','engineering_screen','motor_commands','stability')},indent=2))

if __name__=='__main__':main()
