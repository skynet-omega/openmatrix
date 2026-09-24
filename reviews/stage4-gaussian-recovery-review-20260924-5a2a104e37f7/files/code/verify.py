"""Independent CPU reconstruction of the scoped Gaussian input/command gate.

No simulator imports, no GPU operations, no navigation verdict. Missing evidence
is incomplete, while contradictory boundary evidence belongs to rival C.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAMPAIGN=HERE
PLAN=json.loads((HERE/'reference/PLAN.json').read_text())
GEOMETRY=json.loads((HERE/'reference/GEOMETRY.json').read_text())
LOCK=json.loads((HERE/'reference/SOURCE_LOCK.json').read_text())
DT=1_000_000
EPS=1e-12
IDS=(10176,10208,10360,523769,10065,10118)
ROWS=(159,189,332,131957,52,104)


class IncompleteEvidence(ValueError):pass


def need(ok,message):
    if not ok:raise ValueError(message)


def read_json(path):
    with Path(path).open() as f:
        return json.load(f,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def finite(value,label):
    a=np.asarray(value)
    need(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'Invalid numeric '+label)
    return a


def same(actual,expected,label,atol=0.):
    a=finite(actual,label);b=finite(expected,label)
    need(a.shape==b.shape,'Shape '+label)
    need(np.array_equal(a,b) if atol==0 else np.allclose(a,b,rtol=0,atol=atol),'Mismatch '+label)


def specs_from_geometry(g=GEOMETRY):
    a=np.asarray(g['prepared_antennae_mm']);q=np.asarray(g['prepared_qpos_root'])
    baseline=np.linalg.norm(a[0,:2]-a[1,:2]);left=(a[0,:2]-a[1,:2])/baseline
    forward=np.array([left[1],-left[0]]);w,x,y,z=q[3:]
    if forward@np.array([1-2*(y*y+z*z),2*(w*z+x*y)])<0:forward=-forward
    return {name:{'source_mm':(a[:,:2].mean(0)+sign*.5*baseline*forward+2*baseline*left).tolist(),
                  'sigma_mm':float(2*baseline),'geometry_sha256':PLAN['fixed_inputs']['geometry_sha256']}
            for name,sign in (('plus',1),('minus',-1))}


def concentration(antennae,spec):
    return np.exp(-np.sum((np.asarray(antennae)[...,:2]-spec['source_mm'])**2,axis=-1)/(2*spec['sigma_mm']**2))


def yaw(q):
    w,x,y,z=np.asarray(q)[...,3:7].T
    return np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def check_trace(t,arm):
    need(arm in ('plus','minus'),'Field identity')
    for key,value in t.items():
        if key!='fase':finite(value,'trace '+key)
    q=np.asarray(t['qpos'])
    need(q.ndim==2 and q.shape[1]>=7,'Quaternion pose layout')
    # Existing Stage3 pose_yaw domain, unchanged and checked before yaw.
    need(float(np.max(abs(np.sum(q[:,3:7]*q[:,3:7],axis=1)-1)))<1e-8,
         'Quaternion norm-squared domain')
    n=len(t['fase']);need(40<=n<=440,'Trace must include preparation and bounded trial prefix')
    phases=['preparacion']*40+['ensayo']*(n-40)
    need(t['fase'].tolist()==phases,'Trace phase/order')
    same(t['paso'],np.r_[np.arange(1,41),np.arange(1,n-39)],'trace steps')
    times=t['CNS_time_ns'];need(times.dtype.kind in 'iu','Noninteger CNS clock')
    same(np.diff(times),np.full(n-1,DT),'clock stride')
    for key in ('PN_time_ns','body_time_ns'):same(t[key],times,key)
    start=int(times[0])-DT
    spec=specs_from_geometry()[arm]
    same(t['qpos'][39,:7],GEOMETRY['prepared_qpos_root'],'prepared root',PLAN['preflight']['prepared_root_qpos_max_abs_native'])
    same(t['antenas_mm'][39],GEOMETRY['prepared_antennae_mm'],'prepared antennae',PLAN['preflight']['prepared_antenna_max_abs_mm'])
    initial={k:concentration(t['antenas_mm'][39],v) for k,v in specs_from_geometry().items()}
    initial_error=float(np.max(abs(initial['plus']-initial['minus'])))
    need(initial_error<=PLAN['preflight']['initial_plus_minus_concentration_max_abs'],'Initial source pair')
    for key in ('sensores_usados','sensores_pendientes'):
        need(t[key].shape==(n,3),'Sensor shape '+key)
        same(t[key][:40],np.zeros((40,3)),'sham preparation '+key)
    same(t['concentracion_campo'][:40],np.zeros((40,2)),'sham field')
    after=concentration(t['antenas_mm'][40:],spec)
    before=np.vstack((initial[arm],after[:-1])) if n>40 else np.empty((0,2))
    same(t['concentracion_campo'][40:],after,'Gaussian formula',EPS)
    same(t['sensores_pendientes'][40:,:2],after,'pending Gaussian',EPS)
    same(t['sensores_usados'][40:,:2],before,'one-step sensory lag',EPS)
    same(t['sensores_usados'][:,2],np.zeros(n),'used reinforcement')
    same(t['sensores_pendientes'][:,2],np.zeros(n),'pending reinforcement')
    same(t['DN_q_usada'][1:],t['DN_q_actual'][:-1],'one-step DN lag')
    same(t['DN_baseline'],np.broadcast_to(t['DN_baseline'][0],t['DN_baseline'].shape),'fixed DN baseline')
    dq=t['DN_q_usada']-t['DN_baseline']
    command=np.tanh(250*(dq[:,2]-dq[:,3]))*np.deg2rad(5.)
    same(t['command_yaw_rate_rad_s'],command,'angular decoder',EPS)
    forward=np.clip(.2+np.mean(dq[:,:2],axis=1),0,.5)
    same(t['command_forward_mm_s'],forward,'forward decoder',EPS)
    trial_yaw=(yaw(t['qpos'][40:])-yaw(t['qpos'][39])+180)%360-180
    same(t['yaw_delta_deg'][40:],trial_yaw,'body yaw from quaternion',EPS)
    return {'steps':n-40,'complete':n==440,'start_ns':start,'initial_pair_abs':initial_error,
            'initial':{k:v.tolist() for k,v in initial.items()},'concentration':after,
            'command':command[40:],'yaw':trial_yaw,
            'signed_command_deg':float(np.rad2deg(command[40:].sum()*.001)),
            'absolute_command_deg':float(np.rad2deg(abs(command[40:]).sum()*.001))}


def check_intervals(data,t,arm,m):
    field=data['field'];spec=specs_from_geometry()[arm]
    need(field['arm']==arm and field['sampling_order']==['L','R'],'Interval field/order')
    same(field['source_mm'],spec['source_mm'],'fixed source')
    same(field['sigma_mm'],spec['sigma_mm'],'fixed sigma')
    need(field['geometry_sha256']==PLAN['fixed_inputs']['geometry_sha256'],'Field geometry hash')
    need(field['installed_ns']==int(t['CNS_time_ns'][39]),'Field installation clock')
    rows=data['intervals'];need(len(rows)==m['steps'],'Interval count')
    for j,r in enumerate(rows):
        i=j+40
        need(r['sample_used_ns']==int(t['CNS_time_ns'][i])-DT and r['sample_pending_ns']==int(t['CNS_time_ns'][i]),'Auditor clocks')
        for name,key in (('used','sensores_usados'),('pending','sensores_pendientes'),('PN_available_NOT_consumer_witness','PN_general_transmission')):
            same(r[name],t[key][i],'Auditor '+name)
        same(r['expected_80c_used_NOT_consumer_witness'],80*t['sensores_usados'][i,:2],'Auditor 80c')
        same(r['command_yaw_rate_rad_s'],m['command'][j],'Auditor command',EPS)
    guard=data['prepared_pose_guard']
    need(guard['geometry_sha256']==PLAN['fixed_inputs']['geometry_sha256'],'Pose guard hash')
    same(guard['qpos_root_max_abs_native'],np.max(abs(t['qpos'][39,:7]-GEOMETRY['prepared_qpos_root'])),'Pose receipt root')
    same(guard['antennae_max_abs_mm'],np.max(abs(t['antenas_mm'][39]-GEOMETRY['prepared_antennae_mm'])),'Pose receipt antenna')
    # Diagnostic booleans are cross-checked, never used to derive completion.
    need(data['complete'] is m['complete'],'Auditor complete flag contradicts raw rows')
    need(data['stage4_admission'] is False,'False navigation claim')


def check_descriptor(d):
    need(d['dtype']=='float64' and len(d['shape'])==1,'Boundary dtype/shape')
    size=d['shape'][0];need(type(size) is int and 0<size<=2_000_000,'Boundary size')
    need(d['bytes']==size*8,'Boundary bytes')
    h=d['sha256'];need(isinstance(h,str) and len(h)==64 and all(x in '0123456789abcdef' for x in h),'Boundary digest')
    ix=d['indices'];need(len(ix)==len(set(ix)) and all(type(i) is int and 0<=i<size for i in ix),'Boundary indices')
    need(finite(d['values'],'boundary values').shape==(len(ix),),'Boundary selection shape')


def check_drive(data,index,t,expected_orn=None):
    selected=PLAN['observation']['bounded_drive_epochs'];ordinals=PLAN['observation']['bounded_drive_ordinals']
    need(data['selected_epochs']==selected and index['capture_epochs']==selected,'Drive capture schedule')
    need(data['source']['adapter_sha256']==LOCK[str(ROOT/'campanas/etapa3_motor_nuevo_20260922/organism_adapter.py')],'Drive adapter source')
    need(data['source']['observer_sha256']==LOCK[str(ROOT/'motor_nuevo/adapter_drive_observer_20260923/observer.py')],'Drive observer source')
    limits=read_json(ROOT/'motor_nuevo/adapter_drive_observer_20260923/PLAN.json')['capture_limits']
    need(data['capture_limits']==limits,'Drive observation limits changed')
    rows=index['selected_drive_rows'];left=index['ORN_DM1_L_all_rows'];right=index['ORN_DM1_R_all_rows']
    need(left and right and len(set(left+right))==len(left+right),'ORN laterality/duplicates')
    need(rows==left[:4]+right[:4],'ORN selected rows')
    if expected_orn is not None:
        need(left==expected_orn['L'] and right==expected_orn['R'],'ORN metadata identity')
    labels=[str(f)+':'+str(int(k)) for f,k in zip(t['fase'],t['paso'])]
    epochs=data['epochs'];need([r['label'] for r in epochs]==labels,'Drive epoch coverage')
    for r in epochs:
        need(2<=r['calls']<=32,'Drive per-epoch budget')
        need(r['captured']==(2 if r['label'] in selected else 0),'Drive epoch capture count')
    expected_keys=[(label,o) for label in labels if label in selected for o in ordinals]
    records=data['records'];need([(r['epoch'],r['ordinal']) for r in records]==expected_keys,'Drive record coverage/order')
    pointers=None
    for r in records:
        i=labels.index(r['epoch']);start=int(t['CNS_time_ns'][i])-DT
        need(type(r['start_time_ns']) is int and start<=r['start_time_ns']<start+DT,'Drive start clock')
        need(type(r['duration_ns']) is int and 0<r['duration_ns']<=DT and r['start_time_ns']+r['duration_ns']<=start+DT,'Drive duration')
        if r['ordinal']==0:need(r['start_time_ns']==start,'First observed drive clock')
        for name in ('drive_argument','drive_before','drive_after','pn_refresh','pn_before','pn_after'):check_descriptor(r[name])
        need(r['drive_argument']==r['drive_before']==r['drive_after'],'Drive boundary identity/hash')
        need(r['pn_refresh']==r['pn_before']==r['pn_after'],'PN boundary identity/hash')
        need(r['drive_argument']['indices']==rows and r['pn_before']['indices']==[],'Captured row map')
        # Actual captured arguments are already float64 before validation;
        # preserve their arithmetic instead of inserting an unobserved cast.
        need(r['argument_dtype_before_validation']=='float64','Actual drive argument dtype')
        expected=np.array([80*t['sensores_usados'][i,0]]*len(left[:4])+[80*t['sensores_usados'][i,1]]*len(right[:4]),dtype=np.float64)
        same(r['drive_argument']['values'],expected,'Actual bilateral ORN drive')
        p=r['captured_device_pointers'];need(len(p)==2 and all(type(x) is int and x>0 for x in p),'Device pointers')
        if pointers is None:pointers=p
        need(p==pointers,'Rebound graph boundary')
        need(1<=r['host_pn_reads']<=2,'PN actual refresh count')
        need(r['status']=='OBSERVED_EXACT_BOUNDARIES' and r['argument_unchanged'] is True and r['buffers_unchanged_during_advance'] is True,'Observer contradiction')
    need(data['coefficient_build_calls_observed']==18,'Coefficient build count')
    need(type(data['actual_pn_reader_calls_during_build']) is int and
         data['actual_pn_reader_calls_during_build']==18 and data['pn_capture_binding_observed'] is True,
         'Peripheral PN reader count/binding')
    need(data['open_epoch'] is None and data['failure'] is None and data['failed'] is False,'Incomplete observer')
    return {'captured_calls':len(records),'epochs':len(epochs),'full_buffer_data_archived':False,
            'meaning':'Matching descriptor hashes and eight ORN rows at declared calls; held PN buffer binding during 18 coefficient builds. Route attribution additionally requires frozen layout and prepared/final manifest checks; no all-kernel read proof.'}


def check_pn_route(folder):
    """Separate the disabled generic replacement from preserved peripheral feedback."""
    name='29__legacy_sources__gpu_coefficient_layout.py'
    digest='2598415d108e5ab595351ad2f3a5a841f6212c999026981f6aa38278ca554f9f'
    frozen=read_json(folder/'FROZEN.json')
    need(frozen.get(name)==digest and sha(folder/'executed_sources'/name)==digest,
         'PN route frozen layout identity')
    intervention=read_json(folder/'INTERVENTION.json')
    need(intervention['operation']=='disable left PN629 general replacement' and
         intervention['only_changes']==['general_outputs.enabled','record_sha256'] and
         intervention['dynamic_466_enabled'] is True and intervention['left_pn_id']==10208 and
         intervention['anatomy_changed'] is False and intervention['motor_decoder_changed'] is False,
         'PN route intervention identity')
    states={}
    for state_name in ('prepared_state','final_state'):
        path=folder/state_name/'session.json'
        inventory=read_json(folder/state_name/'MANIFEST.json')['files']['session.json']
        need(path.is_file() and not path.is_symlink() and path.stat().st_size==inventory['bytes'] and
             sha(path)==inventory['sha256'],'PN route saved session integrity')
        # Parse the semantic path, never search arbitrary nested historical text.
        saved=read_json(path)
        manifest=saved['hybrid']['pn_online_manifest']
        need(manifest['general_outputs']['enabled'] is False,'General PN629 replacement enabled')
        need(manifest['electrical_outputs']['enabled'] is True,'Dynamic 466 route changed')
        peripheral=manifest['orn_peripheral_terminal']
        need(type(peripheral['local_PN_feedback_pairs']) is int and peripheral['local_PN_feedback_pairs']==31 and
             peripheral['recurrent_connected'] is True,'Peripheral PN feedback manifest changed')
        need(manifest['record_sha256']==intervention['after_hash'],'PN intervention/session manifest identity')
        states[state_name]={'session_sha256':inventory['sha256'],'general_outputs_enabled':False,
                            'dynamic_466_enabled':True,'local_PN_feedback_pairs':31,'recurrent_connected':True,
                            'manifest_record_sha256':manifest['record_sha256']}
        del saved,manifest,peripheral
    return {'layout_sha256':digest,'saved_states':states,
            'meaning':'Frozen layout has one peripheral reader per coefficient with generic replacement disabled; buffer is passed to the terminal kernel. Counts plus saved flags explain the observed binding, without hardware callsite tracing or proof of pathway efficacy.'}


def load_operator(folder):
    path=folder/'prepared_state/effective_operator'
    meta=read_json(path.with_suffix('.json'));need(meta['schema']=='effective_operator_v2','Operator schema')
    identity={'schema':meta['schema'],'bindings':meta['bindings'],'manifest':meta['manifest']}
    need(hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()==meta['identity_sha256'],'Operator identity')
    values={}
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        for key in ('gain','theta','tau','cuda_gain','cuda_theta','cuda_tau'):
            a=z[meta['values'][key]['__array__']];finite(a,'operator '+key)
            fingerprint={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
            need(fingerprint==meta['manifest'][key],'Operator field hash '+key)
            values[key]=a
    for name in ('gain','theta','tau'):same(values[name],values['cuda_'+name],'CPU/CUDA operator '+name)
    return values


def check_flow(records,arrays,metadata,t,operator):
    need(len(records)==len(t['fase']),'Flow sample count')
    same(arrays['time_ns'],t['CNS_time_ns'],'Flow NPZ time')
    need(arrays['phase'].tolist()==t['fase'].tolist(),'Flow NPZ phase')
    same(arrays['ids'],IDS,'Flow IDs')
    need(metadata['kernel_sha256']==kernel_hashes(),'Native flow kernel provenance')
    for j,(identity,row) in enumerate(zip(IDS,ROWS)):
        m=metadata['rows'][str(identity)]
        need(m['row']==row,'Flow metadata row')
        need(m['type']==('DM1_lPN' if j<2 else 'DNa02' if j<4 else 'DNb05'),'Flow cell identity')
        need(m['side']==('R' if j%2==0 else 'L'),'Flow laterality')
        if identity in IDS[:2]:need(m['operator']=='orn_pn_synaptic' and m['type']=='DM1_lPN','PN kernel ownership')
    errors=[];rates=[]
    for i,r in enumerate(records):
        need(r['phase']==str(t['fase'][i]) and r['ms']==int(t['paso'][i]) and r['time_ns']==int(t['CNS_time_ns'][i]),'Flow time/phase')
        need(r['graph_build_coefficient_calls']==18,'Flow coefficient build count')
        need(set(r['rows'])==set(map(str,IDS)),'Flow rows')
        raw=np.array([r['rows'][str(k)]['raw_signed'] for k in IDS]);drive=np.array([r['rows'][str(k)]['drive'] for k in IDS])
        target=np.array([r['rows'][str(k)]['target'] for k in IDS]);rate=np.array([r['rows'][str(k)]['rate'] for k in IDS])
        theta=np.array([r['rows'][str(k)]['theta'] for k in IDS])
        same(theta,operator['theta'][list(ROWS)],'Flow theta')
        expected=np.maximum(0,np.tanh(operator['gain'][list(ROWS)]*(raw+drive-theta)))
        expected_rate=1/operator['tau'][list(ROWS)]
        same(target,expected,'PN/DN coefficient from raw net',1e-9)
        same(rate,expected_rate,'PN/DN rate',1e-12)
        same(arrays['raw_signed'][i],raw,'Flow raw NPZ/JSONL')
        same(arrays['target'][i],target,'Flow target NPZ/JSONL')
        errors.append(float(np.max(abs(expected-target))));rates.append(float(np.max(abs(expected_rate-rate))))
        same(r['target_max_error'],errors[-1],'Flow claimed error',EPS)
        same(r['rate_max_error'],rates[-1],'Flow claimed rate error',EPS)
    return {'samples':len(records),'target_max_error':max(errors),'rate_max_error':max(rates),
            'scope':'Last accepted fine-stage signed PN/DN net reproduces coefficient. Not ORN-only current, all graph stages, or PN629/466 efficacy.'}


def kernel_hashes():
    sources={}
    for key,file,name in (('general','gpu_visual_brain.py','CUDA_SOURCE'),('pn','prosthetic_olfactory_brain.py','ORN_PN_CUDA_SOURCE')):
        p=ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources'/file
        values=[ast.literal_eval(n.value) for n in ast.parse(p.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        need(len(values)==1 and isinstance(values[0],str),'Kernel literal source')
        sources[key]=values[0]
    original=dict(sources)
    replacements={
        'general':(('double* target, double* rate) {','double* target, double* rate, double* debug_raw) {'),
                   ('target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));','debug_raw[row]=a; target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));')),
        'pn':(('double slow_area, double normalization, double* target) {','double slow_area, double normalization, double* target, double* debug_raw) {'),
              ('if(lane==0) target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row])));','if(lane==0) { debug_raw[row]=current; target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row]))); }'))}
    for key,pairs in replacements.items():
        for before,after in pairs:
            need(sources[key].count(before)==1,'Kernel tap insertion')
            sources[key]=sources[key].replace(before,after)
    return {key+'_'+kind:hashlib.sha256(values[key].encode()).hexdigest() for key in sources for kind,values in (('original',original),('tap',sources))}


def json_lines(path):
    with path.open() as f:return [json.loads(s,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSONL'))) for s in f if s.strip()]


def check_budget_progress(progress,result,t):
    need(len(progress)==len(t['fase']),'Progress coverage')
    elapsed=[];rss=[];gpu=[]
    for i,r in enumerate(progress):
        need(r['phase']==str(t['fase'][i]) and r['step']==int(t['paso'][i]) and r['clock_ns']==int(t['CNS_time_ns'][i]),'Progress alignment')
        for name in ('elapsed_s','step_wall_s','rss_peak_sample_gib','gpu_device_used_sample_gib','gpu_pool_used_sample_gib'):
            need(np.isfinite(r[name]) and r[name]>=0,'Resource numeric '+name)
        need(r['step_wall_s']<=r['elapsed_s'],'Step timing')
        same(r['yaw_delta_deg'],t['yaw_delta_deg'][i],'Progress yaw')
        elapsed.append(r['elapsed_s']);rss.append(r['rss_peak_sample_gib']);gpu.append(r['gpu_device_used_sample_gib'])
    need(np.all(np.diff(elapsed)>=0),'Progress time ordering')
    wall=result['wall_total_s'];need(np.isfinite(wall) and wall>=elapsed[-1] and wall<=PLAN['budget']['wall_each_s_max'],'Wall budget')
    need(max(rss)<=PLAN['budget']['RAM_GiB_max'] and max(gpu)<=PLAN['budget']['GPU_GiB_max'],'Sampled memory budget')
    return {'wall_s':wall,'rss_peak_at_last_sample_gib':max(rss),'gpu_device_used_sample_max_gib':max(gpu),
            'memory_limit_scope':'RSS lifetime high-water sampled per interval; GPU device usage snapshots (includes other processes), no bound on unsampled GPU peaks or cleanup allocation'}


def source_check(campaign):
    repair_lock=read_json(HERE/'REPAIR22_LOCK.json')
    need(repair_lock['schema']=='gaussian_repair22_source_lock_v1','Repair source lock schema')
    for name,digest in repair_lock['files'].items():
        need(Path(name).name==name and sha(HERE/name)==digest,'Repair source hash '+name)
    need(set(repair_lock['files'])=={'verify.py','test_verify.py','REPAIR22_CONTRACT.json'},'Repair source coverage')
    need(sha(campaign/'PLAN.json')==sha(HERE/'reference/PLAN.json'),'Frozen PLAN changed')
    need(sha(campaign/'SOURCE_LOCK.json')==sha(HERE/'reference/SOURCE_LOCK.json'),'Frozen SOURCE_LOCK changed')
    for path,digest in LOCK.items():need(sha(path)==digest,'Frozen source hash '+path)
    need(sha(HERE/'reference/GEOMETRY.json')==PLAN['fixed_inputs']['geometry_sha256'],'Reference geometry hash')
    return {'sources':len(LOCK),'source_lock_sha256':sha(campaign/'SOURCE_LOCK.json'),'plan_sha256':sha(campaign/'PLAN.json'),
            'repair22_lock_sha256':sha(HERE/'REPAIR22_LOCK.json'),'repair22_contract_sha256':sha(HERE/'REPAIR22_CONTRACT.json')}


def check_prepared_receipt(path,left,right,engine):
    comparator=ROOT/'motor_nuevo/gaussian_prepared_semantics_20260923/compare_prepared.py'
    digest='4c17a41d9b213ccc66565f6e1dc874206900b5841b5a6496caf14c8592b91a09'
    need(sha(comparator)==digest,'Prepared comparator changed')
    r=read_json(path)
    need(r['comparator_sha256']==digest,'Prepared receipt comparator hash')
    need(r['left']==str(left.resolve()) and r['right']==str(right.resolve()) and r['expected_engine']==engine,'Prepared receipt inputs')
    need(r['missing']==[] and r['invalid']==[],'Prepared receipt incomplete')
    for side,folder in (('left',left),('right',right)):
        manifest=folder/'MANIFEST.json';m=read_json(manifest)
        need(sha(manifest)==r[side+'_integrity']['manifest_sha256'],'Prepared receipt manifest binding')
        need(sha(folder.parent/'RUN_CONTRACT.json')==r[side+'_profile']['contract_sha256'],'Prepared receipt contract binding')
        need(r[side+'_profile']['identity']['engine']==engine,'Prepared receipt engine')
        need(set(m['files'])==set(r[side+'_integrity']['inventory']),'Prepared inventory')
        for name,v in m['files'].items():
            need(Path(name).name==name,'Prepared member path')
            p=folder/name
            need(p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'Prepared file integrity '+name)
    need(r['left_profile']['identity']==r['right_profile']['identity'],'Prepared profiles differ')
    need(r['left_integrity']['metadata_semantic_sha256']==r['right_integrity']['metadata_semantic_sha256'],'Prepared manifest metadata differ')
    need(set(('session','prosthesis','published','effective_operator','boundary'))<=set(r['sections']),'Prepared receipt missing section')
    for name,section in r['sections'].items():
        need('left_metadata_sha256' in section,'Unsupported comparator extra section '+name)
        need(section['left_metadata_sha256']==section['right_metadata_sha256'],'Prepared metadata differ '+name)
        need(section['array_count']==len(section['arrays']),'Prepared receipt array count')
        for a in section['arrays']:
            need(a['structure_left']==a['structure_right'],'Prepared array layout')
            need(a['payload_sha256_left']==a['payload_sha256_right'] and a['nonfinite_left']==a['nonfinite_right']==0,'Prepared array content/nonfinite')
    # Reconstructed above; reject contradictory diagnostic flags too.
    need(r['status']=='EXACT' and r['scientific_state_exact'] is True,'Prepared receipt contradicts reconstructed sections')
    return {'receipt_sha256':sha(path),'comparator_sha256':digest,'recomputed_section_equality':True,
            'scope':'Separate exact semantic comparator receipt linked to both full raw prepared states; its semantic parse is not repeated here'}


def inspect_arm(folder,expected_orn=None):
    contract=read_json(folder/'RUN_CONTRACT.json');arm=contract['field'];engine=contract['engine']
    need(arm in ('plus','minus') and engine in ('causal_cuda','reference_cuda'),'Run identity')
    need(contract['plan_sha256']==sha(HERE/'reference/PLAN.json'),'Run PLAN identity')
    need(contract['source_identity']['source_lock_sha256']==sha(HERE/'reference/SOURCE_LOCK.json'),'Run lock identity')
    need(contract['source_identity']['sources_count']==len(LOCK),'Run source count')
    need(contract['preparation_ms']==40 and contract['trial_ms']==400,'Run horizon')
    need(contract['drive_capture_epochs']==PLAN['observation']['bounded_drive_epochs'],'Run observer schedule')
    result=read_json(folder/'RESULT.json')
    if result.get('status')=='INTERRUPTED' and (result.get('error') or {}).get('type') in ('InterruptedError','KeyboardInterrupt'):
        raise IncompleteEvidence('Externally interrupted attempt retained; no complete measurement verdict')
    frozen=read_json(folder/'FROZEN.json');need(len(frozen)==len(LOCK),'Frozen source count')
    need(sorted(frozen.values())==sorted(LOCK.values()),'Executed source multiset')
    for name,digest in frozen.items():
        need(Path(name).name==name,'Executed source path')
        need(sha(folder/'executed_sources'/name)==digest,'Executed source corruption')
    with np.load(folder/'traces.npz',allow_pickle=False) as z:t={k:z[k] for k in z.files}
    m=check_trace(t,arm)
    check_intervals(read_json(folder/'GAUSSIAN_INTERVALS.json'),t,arm,m)
    same_spec=read_json(folder/'GAUSSIAN_SPEC.json');need(same_spec==specs_from_geometry(),'Run fixed sources')
    initial=read_json(folder/'GAUSSIAN_INITIAL_GUARD.json')
    for k in ('plus','minus'):same(initial[k],m['initial'][k],'Initial concentrations',EPS)
    same(initial['pair_max_abs'],m['initial_pair_abs'],'Initial pair receipt',EPS)
    same(initial['limit'],PLAN['preflight']['initial_plus_minus_concentration_max_abs'],'Initial limit')
    route=check_pn_route(folder)
    drive=check_drive(read_json(folder/'DRIVE_OBSERVATION.json'),read_json(folder/'DRIVE_INDEX_MAP.json'),t,expected_orn)
    with np.load(folder/'flow/FLOW.npz',allow_pickle=False) as z:flow_arrays={k:z[k] for k in z.files}
    flow=check_flow(json_lines(folder/'flow/FLOW.jsonl'),flow_arrays,read_json(folder/'flow/METADATA.json'),t,load_operator(folder))
    need(result['field']==arm and result['engine']==engine,'RESULT run identity')
    need(result['runtime']['profile']==engine,'Installed numerical profile')
    need(result['completed_trial_ms']==m['steps'] and result['completed_preparation_ms']==40,'RESULT counts contradict raw rows')
    need(result['stage4_pass'] is None,'Runner navigation claim')
    resources=check_budget_progress(json_lines(folder/'PROGRESS.jsonl'),result,t)
    need(result['error'] is None and result['cleanup_errors']==[],'Execution/cleanup failure')
    need(result['status']=='COMPLETE' if m['complete'] else result['status']!='COMPLETE','Runner status contradicts rows')
    return {'arm':arm,'engine':engine,'metrics':m,'drive':drive,'pn_route':route,'flow':flow,'resources':resources,'folder':str(folder),
            'raw_hashes':{name:sha(folder/name) for name in ('traces.npz','GAUSSIAN_INTERVALS.json','DRIVE_OBSERVATION.json','flow/FLOW.npz','flow/FLOW.jsonl','RESULT.json')}}


def pair_metrics(plus,minus):
    need(plus['complete'] and minus['complete'],'Incomplete pair')
    # 300..400 inclusive, array offset 299 in the 400-row trial.
    return {'late_concentration_mean_abs':float(np.mean(abs(plus['concentration'][299:400]-minus['concentration'][299:400]))),
            'command_L1_deg':float(np.rad2deg(abs(plus['command']-minus['command']).sum()*.001)),
            'yaw_sup_deg':float(np.max(abs(plus['yaw']-minus['yaw'])))}


def decide(runs,invalid=False):
    if invalid:return {'rival':'C','classification':'BLOQUEADO','reason':'Contradictory boundary, consumer, identity or budget evidence'}
    keys=[('causal_cuda',s) for s in ('plus','minus')]
    if any(k not in runs or not runs[k]['metrics']['complete'] for k in keys):
        return {'rival':None,'classification':'INCOMPLETO','reason':'Native pair missing or interrupted; no scientific conclusion'}
    native=pair_metrics(*(runs[k]['metrics'] for k in keys))
    threshold=PLAN['thresholds']
    if native['late_concentration_mean_abs']<=threshold['late_300_400ms_mean_concentration_pair_abs_min_exclusive']:
        return {'rival':'C','classification':'DESCARTADO_EN_ESTE_CONTRATO','reason':'Insufficient input separation','native_pair':native}
    if native['command_L1_deg']<=threshold['integrated_absolute_command_pair_deg_min_exclusive']:
        return {'rival':'B','classification':'DESCARTADO_EN_ESTE_CONTRATO','reason':'Separated input without material command contrast in 400 ms','native_pair':native}
    references=[('reference_cuda',s) for s in ('plus','minus')]
    if any(k not in runs or not runs[k]['metrics']['complete'] for k in references):
        return {'rival':None,'classification':'PROMETEDOR_NO_CONFIRMADO','reason':'Material native contrast; both references required for A','native_pair':native}
    reference=pair_metrics(*(runs[k]['metrics'] for k in references));parity={}
    for arm in ('plus','minus'):
        a=runs['causal_cuda',arm]['metrics'];b=runs['reference_cuda',arm]['metrics']
        parity[arm]={'yaw_sup_deg':float(np.max(abs(a['yaw']-b['yaw']))),
                     'command_L1_deg':float(np.rad2deg(abs(a['command']-b['command']).sum()*.001))}
    okay=(reference['late_concentration_mean_abs']>threshold['late_300_400ms_mean_concentration_pair_abs_min_exclusive'] and
          reference['command_L1_deg']>threshold['integrated_absolute_command_pair_deg_min_exclusive'] and
          all(v['yaw_sup_deg']<=threshold['native_reference_yaw_sup_deg_max'] and v['command_L1_deg']<=threshold['native_reference_command_L1_deg_max'] for v in parity.values()))
    return {'rival':'A' if okay else None,'classification':'CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA' if okay else 'BLOQUEADO',
            'reason':'Scoped synthetic intensity/history-to-command effect only' if okay else 'Reference contrast or numerical parity failed; no A/B biological inference',
            'native_pair':native,'reference_pair':reference,'parity':parity}


def compute(campaign=CAMPAIGN):
    started=time.perf_counter();campaign=Path(campaign);errors=[];missing=[];runs={};sources=None
    try:sources=source_check(campaign)
    except (ValueError,OSError,KeyError) as e:errors.append({'scope':'sources','error':str(e)})
    contracts=sorted(campaign.glob('*/RUN_CONTRACT.json'))
    anatomy=None
    try:
        anatomy=read_json(HERE/'reference/ORN_INDEX_MAP.json')
        need(sha(anatomy['source_path'])==anatomy['source_sha256'],'Anatomical table changed')
    except (ValueError,OSError,KeyError) as e:errors.append({'scope':'anatomy','error':str(e)})
    if len(contracts)>PLAN['budget']['organism_runs_max']:errors.append({'scope':'budget','error':'Too many run contracts'})
    exposure_keys=[]
    for p in contracts:
        try:
            c=read_json(p);exposure_keys.append((c['engine'],c['field']))
        except (ValueError,OSError,KeyError) as e:errors.append({'scope':'run inventory','error':str(e)})
    if len(exposure_keys)!=len(set(exposure_keys)):errors.append({'scope':'budget','error':'Duplicate arm/profile attempt; preserve exposure, no silent retry'})
    for engine,limit in (('causal_cuda',PLAN['budget']['native_runs_max']),('reference_cuda',PLAN['budget']['reference_runs_max'])):
        if sum(k[0]==engine for k in exposure_keys)>limit:errors.append({'scope':'budget','error':'Run count exceeded '+engine})
    known={p.parent for p in contracts}
    for p in campaign.iterdir():
        if p.is_dir() and p not in known and any((p/n).exists() for n in ('RESULT.json','PROGRESS.jsonl','traces.npz')):
            missing.append({'folder':str(p),'missing':'RUN_CONTRACT.json; unbound attempt evidence'})
    for path in contracts:
        try:
            run=inspect_arm(path.parent,None if anatomy is None else anatomy['ORN_DM1']);key=(run['engine'],run['arm'])
            need(key not in runs,'Duplicate arm/engine exposure');runs[key]=run
        except FileNotFoundError as e:missing.append({'folder':str(path.parent),'missing':str(e.filename)})
        except IncompleteEvidence as e:missing.append({'folder':str(path.parent),'interrupted':str(e)})
        except (ValueError,KeyError,TypeError,IndexError,OSError) as e:errors.append({'folder':str(path.parent),'error':str(e)})
    preparation_checks={}
    for engine in ('causal_cuda','reference_cuda'):
        if all((engine,a) in runs for a in ('plus','minus')):
            try:
                preparation_checks[engine]=check_prepared_receipt(
                    campaign/('PREPARED_COMPARE_'+engine+'.json'),
                    Path(runs[engine,'plus']['folder'])/'prepared_state',
                    Path(runs[engine,'minus']['folder'])/'prepared_state',engine)
            except FileNotFoundError as e:missing.append({'scope':'prepared pair '+engine,'missing':str(e.filename)})
            except (ValueError,OSError,KeyError,TypeError) as e:errors.append({'scope':'prepared pair '+engine,'error':str(e)})
    wall=sum(r['resources']['wall_s'] for r in runs.values())
    if wall>PLAN['budget']['aggregate_wall_s_max']:errors.append({'scope':'budget','error':'Aggregate wall exceeded'})
    decision=decide(runs,invalid=bool(errors))
    if not errors and missing:
        decision={**decision,'classification':'INCOMPLETO','rival':None,'reason':'Missing or interrupted evidence prevents a complete campaign verdict'}
    if not errors and any(all((engine,a) in runs for a in ('plus','minus')) and engine not in preparation_checks for engine in ('causal_cuda','reference_cuda')):
        decision={'rival':None,'classification':'INCOMPLETO','reason':'Prepared comparison receipt missing; metrics retained without paired causal inference'}
    summarized={}
    for key,r in runs.items():
        summarized['/'.join(key)]={k:v for k,v in r.items() if k != 'metrics'}
        summarized['/'.join(key)]['metrics']={k:v for k,v in r['metrics'].items() if not isinstance(v,np.ndarray)}
    return {'schema':'independent_gaussian_gate_v1','sources':sources,'decision':decision,'prepared_comparisons':preparation_checks,'errors':errors,'missing':missing,'runs':summarized,
            'stage4_navigation_admitted':False,'biological_equivalence':False,'aggregate_completed_run_wall_s':wall,
            'verifier_wall_s':time.perf_counter()-started,'verifier_rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations':['No new organism or GPU execution; this is raw readback, not reproduction.',
                'No cold checkpoint restore or independent MuJoCo reconstruction of antennae from all joints.',
                'Drive full buffers were not archived: their hashes establish agreement among recorded descriptors, not recoverable full data.',
                'ORN index identity is checked against a separately pinned CPU anatomical map; no physiological ORN calibration.',
                'PN observation is signed total net at one fine-stage per interval, not isolated ORN current or efficacy of PN629/466.',
                'No independent hardware memory trace; GPU samples do not bound intersample peaks.',
                'SOURCE_LOCK is checked, but is not a complete transitive organism dependency manifest.',
                'Stage3 eligibility receipts and event-owner logs are not independently re-audited here.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=CAMPAIGN);p.add_argument('--write',type=Path);a=p.parse_args()
    result=compute(a.campaign);text=json.dumps(result,indent=2,allow_nan=False)
    if a.write:
        with a.write.open('x') as f:f.write(text+'\n')
    print(text)
    return 2 if result['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
