"""Reconstruct bounded recruitment evidence; never infer navigation admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def need(ok,msg):
    if not ok:raise ValueError(msg)


def read(path):return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def operator(path):
    m=read(path.with_suffix('.json'));out={}
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        for name,ref in m['values'].items():
            a=z[ref['__array__']];need(np.isfinite(a).all(),'Nonfinite operator')
            got=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            need(got==m['manifest'][name],'Operator metadata/raw mismatch: '+name);out[name]=got
    return out


def validate_arm(folder,condition,plan,donor,reference_inputs=None):
    folder=Path(folder);r=read(folder/'RESULT.json');c=read(folder/'RUN_CONTRACT.json')
    need(r['status']=='COMPLETE' and not r['error'] and not r['cleanup_errors'],'Incomplete arm')
    need(r['condition']==c['condition']==condition,'Condition metadata differs')
    need(r['completed_preparation_ms']==40 and r['completed_trial_ms']==200,'Completion count')
    need(r['stage4_admission'] is False and r['stage5_admission'] is False,'Unsupported stage admission')
    need(c['plan_sha256']==sha(HERE/'DYNAMIC_PLAN.json'),'Run plan differs')
    z=arrays(folder/'traces.npz');p=arrays(folder/'probe/PANEL.npz');m=read(folder/'probe/METADATA.json')
    need(set(z)==set(donor)|{'neural_command_shadow'},'Trace column set')
    need(z['fase'].tolist()==['preparacion']*40+['ensayo']*200,'Trace phase')
    need(z['paso'].tolist()==list(range(1,41))+list(range(1,201)),'Trace steps')
    for key,a in z.items():
        need(len(a)==240,'Trace length: '+key)
        if a.dtype.kind!='U':need(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'Invalid trace: '+key)
    for k in donor:
        need(np.array_equal(z[k][:60],donor[k][:60]),'Preparation/prepulse donor differs: '+k)
        if condition=='sham':need(np.array_equal(z[k],donor[k][:240]),'Sham differs: '+k)
    for k in ('qpos','qvel','command_forward_mm_s','command_yaw_rate_rad_s','sensores_usados','sensores_pendientes','antenas_mm','generalized_force_native'):
        need(np.array_equal(z[k],donor[k][:240]),'Paired physical input differs: '+k)
    need(np.array_equal(z['DN_q_usada'][40:],z['DN_q_actual'][39:-1]),'Readout lag')
    dq=z['DN_q_usada'][40:]-z['DN_baseline'][40:]
    shadow=np.column_stack((np.clip(.2+np.mean(dq[:,:2],axis=1),0,.5),np.tanh(250*(dq[:,2]-dq[:,3]))*np.deg2rad(5)))
    need(np.allclose(z['neural_command_shadow'][40:],shadow,rtol=0,atol=1e-15),'Shadow readout equation')
    need(np.array_equal(p['phase'],z['fase']) and np.array_equal(p['step'],z['paso']) and np.array_equal(p['time_ns'],z['CNS_time_ns']),'Panel clock')
    need(p['ids'].tolist()==m['ids'] and p['q'].shape==(240,22) and p['raw'].shape==(240,4,22),'Panel shape')
    for k in ('q','raw','maximum','counts'):need(np.isfinite(p[k]).all(),'Nonfinite panel')
    need(np.all((p['q']>=0)&(p['q']<=1)),'Panel domain')
    need(p['counts'].dtype.kind=='u' and np.all(p['counts'][:,0]>0) and np.all(p['counts'][:,1]<=p['counts'][:,0]),'Query counts')
    gain,theta=np.asarray(m['gain']),np.asarray(m['theta'])
    reconstructed=np.maximum(0,np.tanh(gain*(p['raw'][:,0]+(p['raw'][:,1]+p['raw'][:,2])-theta)))
    target_error=float(np.max(abs(reconstructed-p['raw'][:,3])))
    need(target_error<=1e-12,'Native equation differs')
    need(m['source_ids']==plan['sources'][condition],'Pulse identities changed')
    wanted=np.zeros((240,22))
    dose=None
    if m['source_ids']:
        dose=read(folder/'probe/DOSE.json')
        need(dose['ids']==[i for i in m['ids'] if i in m['source_ids']],'Dose identity')
        ix=np.asarray([m['ids'].index(i) for i in dose['ids']])
        initial_input=np.asarray(dose['raw_input'])+dose['preexisting_drive']
        u=gain[ix]*(initial_input-theta[ix]);t0=np.maximum(0,np.tanh(u));goal=t0+.25*(1-t0)
        delta=(np.arctanh(goal)-u)/gain[ix]
        need(np.isfinite(delta).all() and np.all(delta>0) and np.all(goal<1),'Unresolved dose')
        need(np.allclose(delta,dose['additional_drive'],rtol=1e-12,atol=1e-12),'Dose formula changed')
        need(np.allclose(goal,dose['instantaneous_target'],rtol=1e-10,atol=1e-12),'Dose target changed')
        wanted[60:100,ix]=np.asarray(dose['additional_drive'])[None,:]
    need(np.array_equal(p['raw'][:,2],wanted),'Wrong native dose, window or cell')
    inputs=[json.loads(x) for x in (folder/'INPUTS.jsonl').read_text().splitlines()]
    need(len(inputs)==240,'Input witness count')
    for index,x in enumerate(inputs):
        need(x['phase']==z['fase'][index] and x['step']==int(z['paso'][index]),'Input witness order')
        need([v['ns'] for v in x['adapter']]==[62500,125000]*8,'Input cadence')
        need(bool(x['body'])==(index>=40),'Physical witness coverage')
    if reference_inputs is not None:need(inputs==reference_inputs,'Body/copied input mismatch')
    prepared=operator(folder/'prepared_state/effective_operator');final=operator(folder/'final_state/effective_operator')
    need(prepared==final,'Biological operator changed during probe')
    return z,p,m,inputs,dict(condition=condition,target_reconstruction_max_error=target_error,
        raw_parameter_arrays_unchanged=True,wall_s=float(r['wall_total_s']),dose=dose,
        trace_sha256=sha(folder/'traces.npz'),panel_sha256=sha(folder/'probe/PANEL.npz'),input_sha256=sha(folder/'INPUTS.jsonl'))


def summarize(sham,intervention,metadata,plan):
    q=intervention['q'][60:];base=sham['q'][60:];delta=q-base
    records=[]
    for j,identity in enumerate(metadata['ids']):
        max_q=float(np.max(abs(delta[:,j])));net=intervention['raw'][60:,0,j]-sham['raw'][60:,0,j]
        maximum=float(np.max(intervention['maximum'][60:,j]));source=identity in metadata['source_ids']
        threshold=plan['screen']['source_material_q_effect'] if source else plan['screen']['panel_material_q_effect']
        classification=('MATERIAL_SOURCE_RESPONSE' if source else 'MATERIAL_PANEL_RESPONSE') if max_q>=threshold else (
            'SUBTHRESHOLD_AT_ALL_EXECUTED_QUERIES' if maximum<=0 else 'NO_MATERIAL_Q_EFFECT_IN_THIS_WINDOW')
        records.append(dict(id=identity,source=source,max_abs_delta_q=max_q,final_delta_q=float(delta[-1,j]),
            min_delta_q=float(np.min(delta[:,j])),max_delta_q=float(np.max(delta[:,j])),
            baseline_max_q=float(np.max(base[:,j])),pulse_end_delta_q=float(delta[39,j]),max_abs_delta_synaptic_input=float(np.max(abs(net))),
            maximum_argument_over_all_queries=maximum,classification=classification,
            tau_s=float(metadata['tau'][j]),gain=float(metadata['gain'][j]),theta=float(metadata['theta'][j])))
    return records


def motor_effect(sham,intervention):
    yaw=np.rad2deg(intervention['neural_command_shadow'][40:,1]-sham['neural_command_shadow'][40:,1])
    forward=intervention['neural_command_shadow'][40:,0]-sham['neural_command_shadow'][40:,0]
    return dict(min_delta_command_deg_s=float(yaw.min()),max_delta_command_deg_s=float(yaw.max()),
        integral_delta_command_deg=float(yaw.sum()*.001),max_abs_delta_forward_mm_s=float(abs(forward).max()),
        scope='Unapplied neural command difference; integral is not actual yaw or evidence of navigation.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    need(not a.out.exists(),'Refuse overwrite')
    plan=read(HERE/'DYNAMIC_PLAN.json');donor=arrays(HERE/plan['donor']);need(sha(HERE/plan['donor'])==plan['donor_sha256'],'Donor hash')
    sz,sp,sm,si,sr=validate_arm(HERE/'sham_01','sham',plan,donor)
    conditions=[sr];wall=sr['wall_s']
    for condition in plan['conditions'][1:]:
        z,p,m,inp,result=validate_arm(HERE/(condition+'_01'),condition,plan,donor,si)
        need(m['ids']==sm['ids'] and m['theta']==sm['theta'] and m['gain']==sm['gain'] and m['tau']==sm['tau'],'Paired model differs')
        result['cells']=summarize(sp,p,m,plan)
        result['max_abs_neural_yaw_command_difference_deg_s']=float(np.rad2deg(np.max(abs(z['neural_command_shadow'][40:,1]-sz['neural_command_shadow'][40:,1]))))
        result['max_abs_neural_forward_command_difference_mm_s']=float(np.max(abs(z['neural_command_shadow'][40:,0]-sz['neural_command_shadow'][40:,0])))
        result['shadow_motor_effect']=motor_effect(sz,z)
        conditions.append(result);wall+=result['wall_s']
    need(wall<=plan['budget']['aggregate_wall_s'] and all(r['wall_s']<=plan['budget']['per_run_wall_s'] for r in conditions),'Frozen wall budget')
    result=dict(classification='DESCRIPTIVE_CAUSAL_SCREEN_COMPLETE',conditions=conditions,organism_wall_s=wall,
        stage4_admission=False,stage5_admission=False,
        limitations=['No refined200ms reference allocated; material effects remain diagnostic, not final numerical admission.',
            'Fixed common inputs and body trajectory; shadow neural commands do not move the body.',
            'Functional model pulse, not physiology-calibrated stimulation. Silence conclusions concern22-cell panel and sampled interval only.',
            'All-query maxima include discarded predictors/rejected trials; query counts are not accepted-time occupancy.'])
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(classification=result['classification'],organism_wall_s=wall,arms=len(conditions))))


if __name__=='__main__':main()
