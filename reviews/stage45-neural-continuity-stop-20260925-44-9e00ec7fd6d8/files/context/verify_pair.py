"""Reconstruct the causal contrast from consumed inputs, states and body tapes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from verify_sham import calculate as verify_sham, need, sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    for key,value in data.items():
        if value.dtype.kind in 'fc':need(np.isfinite(value).all(),'Nonfinite '+key)
    return data

def compare_exact(a,b,label):
    need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),'Unmatched '+label)

def calculate(common,virtual):
    sham=verify_sham(HERE/'sham_01')
    need(sham['classification']=='SHAM_EXACT','Sham invalid')
    plan=json.loads((HERE/'PLAN.json').read_text()); floor=plan['effects']['layer_numeric_floor']
    common,virtual=Path(common),Path(virtual)
    import sys
    sys.path.insert(0,str(ROOT/'campanas/etapa45_navigation_wind_20260925_40'))
    from checkpoint_compare_utf8 import compare
    for folder,arm in ((common,'common'),(virtual,'virtual')):
        initial=compare(ROOT/'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/final_state',folder/'prepared_state')
        need(initial['exact'],'Complete paired initial state differs')
        contract=json.loads((folder/'RUN_CONTRACT.json').read_text())
        need(contract['arm']==arm and contract['plan_sha256']==sha(HERE/'PLAN.json')
             and contract['source_lock_sha256']==sha(HERE/'SOURCE_LOCK.json')
             and contract['body_commands_from_donor'] is True and contract['neural_output_delivered'] is False,
             'Paired context/flag changed')
        result=json.loads((folder/'RESULT.json').read_text())
        need(result['stage4_admission'] is False and result['stage5_admission'] is False
             and result['completed_ms']==120 and result['error'] is None and result['cleanup_errors']==[],
             'Execution flags differ from raw verified scope')
    a,b=arrays(common/'traces.npz'),arrays(virtual/'traces.npz')
    s=arrays(HERE/'sham_01/traces.npz')
    need(set(a)==set(b)==set(s),'Trace keys differ')
    t=arrays(ROOT/'campanas/etapa45_orientation_observability_20260925_43/tapes_01/TAPES.npz')
    for key in a:
        end=19 if key in ('sensores_pendientes','concentracion_campo') else 20
        compare_exact(a[key][:end],s[key][:end],'common prefix '+key)
        compare_exact(b[key][:end],s[key][:end],'virtual prefix '+key)
    compare_exact(a['sensores_usados'][20:],t['common_control_used'],'common consumed tape')
    compare_exact(b['sensores_usados'][20:],t['virtual_intervention_used'],'virtual consumed tape')
    need(np.max(np.abs(np.mean(a['sensores_usados'][20:,:2],axis=1)-
                       np.mean(b['sensores_usados'][20:,:2],axis=1)))<=plan['effects']['common_match_max_abs'],'Input means differ')
    body_keys=('qpos','qvel','position_mm','antenas_mm','natural_concentration','upright',
               'contact_active','normal_force_N','contact_force_N','generalized_force_native',
               'energy_motor_J','command_forward_mm_s','command_yaw_rate_rad_s','wind_torque_native')
    for key in body_keys:
        compare_exact(a[key],s[key],'common body '+key);compare_exact(b[key],s[key],'virtual body '+key)
    forces=[arrays(p/'substep_body.npz') for p in (HERE/'sham_01',common,virtual)]
    for key in forces[0]:
        compare_exact(forces[1][key],forces[0][key],'common substep '+key)
        compare_exact(forces[2][key],forces[0][key],'virtual substep '+key)
    inputs_a,inputs_b=arrays(common/'actual_inputs.npz'),arrays(virtual/'actual_inputs.npz')
    schemas=[json.loads((p/'INPUT_SCHEMA.json').read_text()) for p in (common,virtual)]
    need(schemas[0]==schemas[1],'Input schemas differ')
    left,right=map(lambda k:np.array(schemas[0][k],dtype=int),('ORN_L_indices','ORN_R_indices'))
    mask=np.ones(inputs_a['drive'].shape[1],dtype=bool);mask[left]=False;mask[right]=False
    compare_exact(inputs_a['drive'][:,mask],inputs_b['drive'][:,mask],'nonolfactory actual drive')
    compare_exact(inputs_a['light'],inputs_b['light'],'actual retinal input')
    delta_input=b['sensores_usados']-a['sensores_usados'];gain=schemas[0]['odor_drive']
    residual=[]
    for channel,index in enumerate((left,right)):
        difference=inputs_b['drive'][:,index]-inputs_a['drive'][:,index]
        error=float(np.max(np.abs(difference-gain*delta_input[:,channel,None])))
        need(error<=1e-12,'Actual ORN drive not explained by tape');residual.append(error)
    w_a=json.loads((common/'EXOGENOUS_WITNESS.json').read_text());w_b=json.loads((virtual/'EXOGENOUS_WITNESS.json').read_text())
    need(w_a==w_b and len(w_a)==120,'Exogenous RNG/proprio/light witness differs')
    need(all(w['rng_before']==w['rng_after']==w_a[0]['rng_before'] for w in w_a),'Live RNG changed')
    compare_exact(a['neural_command_raw_rad_s'][:21],b['neural_command_raw_rad_s'][:21],'motor causal latency')
    layers={}
    keys=('sensores_usados','ORN_q_L','ORN_q_R','ORN_filters','PN_q_legacy',
          'PN_general_transmission','PN_gamma_nS','PN_additional_nS','DN_q_actual','DN_q_usada',
          'neural_command_raw_rad_s','motor_filter_state_rad_s','motor_filter_applied_rad_s')
    for key in keys:
        d=b[key].astype(float)-a[key].astype(float);result={}
        for first,last in plan['effects']['windows_steps']:
            sel=(a['paso']>=first)&(a['paso']<=last);part=d[sel]
            result[str(first)+'_'+str(last)]={'rms':float(np.sqrt(np.mean(part**2))),
                                            'max_abs':float(np.max(np.abs(part))),
                                            'above_numeric_floor':bool(np.max(np.abs(part))>floor)}
        layers[key]=result
    raw_delta=np.rad2deg(b['neural_command_raw_rad_s'][20:]-a['neural_command_raw_rad_s'][20:])
    abs_integral=float(np.sum(np.abs(raw_delta))*0.001)
    net=float(np.sum(raw_delta)*0.001);mean_last=float(np.mean(raw_delta[-50:]))
    material=(abs_integral>=plan['effects']['material_raw_abs_integral_deg_min'] or
              abs(mean_last)>=plan['effects']['material_raw_mean_last50_deg_s_min'])
    detectable=layers['neural_command_raw_rad_s']['1021_1120']['above_numeric_floor']
    return {'schema':'stage45_neural_matched_pair_verified44_v1','classification':
            'PROMETEDOR_NO_CONFIRMADO' if material else 'DESCARTADO_EFECTO_MATERIAL_100MS',
            'plan_sha256':sha(HERE/'PLAN.json'),'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json'),
            'tape_sha256':sha(ROOT/'campanas/etapa45_orientation_observability_20260925_43/tapes_01/TAPES.npz'),
            'trace_sha256':{'common':sha(common/'traces.npz'),'virtual':sha(virtual/'traces.npz')},
            'all_exogenous_and_physical_controls_exact':True,'actual_drive_residual_max':residual,
            'layers':layers,'raw_delta_abs_integral_deg':abs_integral,'raw_delta_net_integral_deg':net,
            'raw_delta_last50_mean_deg_s':mean_last,'raw_detectable':detectable,'raw_material':material,
            'stage4_admission':False,'stage5_admission':False,
            'scope':'Effect of finite lateral tape redistribution at matched input mean on computed neural output. No free body feedback, angular error code or learning established.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--common',type=Path,required=True);p.add_argument('--virtual',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=calculate(a.common,a.virtual)
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('classification','raw_delta_abs_integral_deg','raw_delta_last50_mean_deg_s')}))

if __name__=='__main__':main()
