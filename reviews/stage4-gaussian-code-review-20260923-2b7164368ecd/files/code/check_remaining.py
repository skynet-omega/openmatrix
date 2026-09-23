"""Reconstruct the coupled withdrawal and continuation gates from raw observations."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from checkpoint_compare import compare,sha
from check_support import check as support
HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'etapa3_pn629_intervention_20260923_15'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def js(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
    for k,v in out.items():
        if v.dtype.kind in 'fc':require(np.isfinite(v).all(),'Nonfinite '+k)
    return out
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def clock(t,start,stop):
    expected=np.arange(start,stop+1)
    require(np.array_equal(t['paso'],expected),'Wrong trial samples')
    require(np.all(t['fase']=='ensayo'),'Wrong phase')
    c=t['CNS_time_ns'];require(c.dtype.kind in 'iu' and np.all(np.diff(c)==1_000_000),'Wrong CNS clock')
    require(np.array_equal(c,t['PN_time_ns']) and np.array_equal(c,t['body_time_ns']),'Physical clocks differ')
    require(np.array_equal(t['sensores_usados'][1:],t['sensores_pendientes'][:-1]),'Broken sensory lag')
    require(np.array_equal(t['DN_q_usada'][1:],t['DN_q_actual'][:-1]),'Broken neural reader lag')
def pose_yaw(t,origin):
    q=t['qpos'][:,3:7];require(np.max(np.abs(np.sum(q*q,axis=1)-1))<1e-8,'Invalid quaternions')
    def angles(v):
        w,x,y,z=v.T;return np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    yaw=(angles(q)-float(angles(np.asarray(origin).reshape(1,4))[0])+180)%360-180
    require(np.max(np.abs(yaw-t['yaw_delta_deg']))<1e-10,'Yaw does not match recorded physical pose')
    return yaw
def exposure(t,boundary):
    xy=t['antenas_mm'][:,:,:2];center=np.asarray(boundary['center_mm']);axis=np.asarray(boundary['odor_axis'])
    expected=((xy-center)@axis>0).astype(float)
    require(np.array_equal(expected,t['concentracion_campo']),'Static field geometry differs')
    require(np.array_equal(expected,t['sensores_pendientes'][:,:2]) and np.all(t['sensores_pendientes'][:,2]==0),'Pending odor differs from live geometry')
def withdrawal(folder,arm):
    p=js(HERE/'PLAN.json');c=p['criteria'];folder=Path(folder);base=NATIVE/('full_'+arm+'_01')
    r=js(folder/'RESULT.json');w=js(folder/'WITHDRAWAL.json');contract=js(folder/'EXPERIMENT_CONTRACT.json')
    require(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Incomplete withdrawal')
    require(r['completed_trial_ms']==400 and r['completed_preparation_ms']==40 and r['engine']=='causal_cuda' and r['odor']==arm,'Wrong withdrawal arm/horizon')
    require(r['wall_total_s']<=p['budget']['wall_each_s_max'],'Withdrawal wall budget')
    require(contract['plan_sha256']==sha(HERE/'PLAN.json'),'Wrong withdrawal plan')
    require(w['active'] and w['clamped_physical_calls']==16000 and w['prepared_physical_calls']==1600,'Wrong withdrawal timing')
    require(w['max_requested_yaw_rad_s']>0,'No requested angular signal to withdraw')
    require(js(folder/'FROZEN.json')==js(base/'FROZEN.json') and js(folder/'INTERVENTION.json')==js(base/'INTERVENTION.json'),'Other operator/source intervention')
    a,b=arrays(folder/'traces.npz'),arrays(base/'traces.npz')
    require(set(a)==set(b) and all(len(v)==440 for v in a.values()),'Wrong trace schema')
    require(all(np.array_equal(a[k][:40],b[k][:40]) for k in a),'Preparation observations differ')
    prepared=compare(base/'prepared_state',folder/'prepared_state',names=('session','prosthesis','published'))
    require(prepared['exact'],'Preparation full state differs')
    t={k:v[40:] for k,v in a.items()};clock(t,1,400)
    require(np.all(t['command_yaw_rate_rad_s']==0),'Angular command not removed')
    dq=t['DN_q_usada']-t['DN_baseline'];forward=np.clip(.2+np.mean(dq[:,:2],axis=1),0.,.5)
    require(np.array_equal(forward,t['command_forward_mm_s']),'Forward reader law changed')
    yaw=pose_yaw(t,a['qpos'][39,3:7]);field=js(folder/'final_state/boundary.json')
    require(field==js(base/'final_state/boundary.json'),'Field moved or changed')
    exposure(t,field)
    s=support(folder,c);require(s['passed'],'Withdrawal support failed')
    flow=js(folder/'flow/RESULT.json');require(flow['samples']==440 and flow['max_target_error']<=1e-9 and flow['max_rate_error']<=1e-12,'Native flow reconstruction failed')
    result={'arm':arm,'yaw_400_deg':float(yaw[-1]),'preparation':prepared,'support':s,
            'angular_command_zero_all_samples':True,'forward_law_unchanged':True,'coupled_feedback_preserved':True,
            'traces_sha256':sha(folder/'traces.npz'),'status':'COMPLETE_VALIDATED','stage3_admission':False}
    save(folder/'WITHDRAWAL_RESULT.json',result);return result
def withdrawal_pair(left,right):
    c=js(HERE/'PLAN.json')['criteria'];a=withdrawal(left,'odor_left');b=withdrawal(right,'odor_right')
    odd=(a['yaw_400_deg']-b['yaw_400_deg'])/2
    result={'arms':[a,b],'odd_yaw_deg':odd,'limit_deg':c['angular_reader_withdrawal_odd_yaw_deg_max'],
            'passed':abs(odd)<=c['angular_reader_withdrawal_odd_yaw_deg_max'],'scope':'Necessity of effective angular readout in this coupled trajectory, not necessity of a uniquely identified biological neuron type'}
    save(HERE/'WITHDRAWALS_RESULT.json',result);require(result['passed'],'Coupled withdrawal retains directional yaw');return result
def continuation(folder):
    folder=Path(folder);source=HERE/'reference_odor_right_01';p=js(HERE/'PLAN.json');c=p['criteria'];r=js(folder/'RESULT.json')
    require(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'] and r['completed_continuation_ms']==300,'Incomplete continuation')
    require(r['wall_total_s']<=p['budget']['wall_each_s_max'],'Continuation wall budget')
    initial=compare(source/'state_100ms',folder/'state_100ms');require(initial['exact'],'Restored initial state differs')
    actual=arrays(folder/'traces.npz');ref=arrays(source/'traces.npz');ref={k:v[140:] for k,v in ref.items()}
    require(set(actual)==set(ref) and all(len(v)==300 for v in actual.values()),'Continuation traces incomplete')
    clock(actual,101,400);require(np.array_equal(actual['CNS_time_ns'],ref['CNS_time_ns']),'Continuation clock mismatch')
    require(np.array_equal(actual['DN_baseline'],ref['DN_baseline']),'Continuation baseline changed')
    field=js(folder/'final_state/boundary.json');require(field==js(source/'final_state/boundary.json'),'Continuation field changed');exposure(actual,field)
    require(np.array_equal(actual['sensores_usados'],ref['sensores_usados']),'Continuation sensory exposures differ')
    dq=actual['DN_q_usada']-actual['DN_baseline'];omega=np.tanh(250*(dq[:,2]-dq[:,3]))*np.deg2rad(5.)
    require(np.array_equal(omega,actual['command_yaw_rate_rad_s']),'Continuation reader changed')
    with np.load(source/'traces.npz',allow_pickle=False) as z:origin=z['qpos'][39,3:7].copy()
    pose_yaw(actual,origin)
    yaw=float(np.max(np.abs(actual['yaw_delta_deg']-ref['yaw_delta_deg'])))
    cmd=float(np.rad2deg(np.sum(np.abs(actual['command_yaw_rate_rad_s']-ref['command_yaw_rate_rad_s']))*.001))
    result={'engine':'reference_cuda','same_profile':True,'initial':initial,'yaw_sup_deg':yaw,'command_L1_deg':cmd,
            'passed':yaw<=c['resume_yaw_sup_deg_max'] and cmd<=c['resume_command_L1_deg_max'],
            'errors_by_trace':{k:float(np.max(np.abs(actual[k]-ref[k]))) for k in actual if actual[k].dtype.kind in 'f'},
            'discrete_mismatches':{k:int(np.count_nonzero(actual[k]!=ref[k])) for k in actual if actual[k].dtype.kind in 'biu'},
            'traces_sha256':[sha(folder/'traces.npz'),sha(source/'traces.npz')],
            'scope':'Cold continuation at100ms of this reference-profile organism; not all profiles or arbitrary checkpoints'}
    save(HERE/'CONTINUATION_RESULT.json',result);require(result['passed'],'Functional continuation differs');return result
if __name__=='__main__':
    if sys.argv[1]=='withdrawal':print(json.dumps(withdrawal(Path(sys.argv[2]),sys.argv[3])))
    elif sys.argv[1]=='pair':print(json.dumps(withdrawal_pair(Path(sys.argv[2]),Path(sys.argv[3]))))
    elif sys.argv[1]=='continuation':print(json.dumps(continuation(Path(sys.argv[2]))))
    else:raise ValueError('Unknown check')
