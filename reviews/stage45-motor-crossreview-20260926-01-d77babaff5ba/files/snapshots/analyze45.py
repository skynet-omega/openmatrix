"""Paired acute-response analysis; never calibrate a model unit as firing Hz."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'motor_nuevo/dynamics_12s_20260926_13'))
from observations import save,sha,need


def read_arm(name):
    folder=HERE/name
    result=json.loads((folder/'RESULT.json').read_text())
    need(result['status']=='COMPLETE' and result['persisted_ms']==4000,'Incomplete arm '+name)
    pieces=[]
    for block in sorted((folder/'blocks').glob('*ms')):
        manifest=json.loads((block/'MANIFEST.json').read_text())
        need(all(sha(block/k)==v for k,v in manifest['hashes'].items()),'Changed block '+str(block))
        with np.load(block/'traces.npz',allow_pickle=False) as z:
            pieces.append({key:z[key].copy() for key in z.files})
    trace={key:np.concatenate([x[key] for x in pieces]) for key in pieces[0]}
    need(np.array_equal(trace['paso'],np.arange(1,4001)),'Discontinuous trace')
    return result,trace


def arm_metrics(trace,plan):
    pos=trace['position_mm'][:,:2]
    speed=np.linalg.norm(trace['qvel'][:,:2],axis=1)*10.
    # MuJoCo's free-joint rotational velocity is body-local. Its norm is
    # invariant to that frame; it is not falsely labelled world yaw speed.
    angular=np.rad2deg(np.linalg.norm(trace['qvel'][:,3:6],axis=1))
    i,j=plan['rest_window_ms']
    rest_span=float(np.linalg.norm(pos[j-1]-pos[i-1]))
    yaw=trace['yaw_delta_deg'][i-1:j]
    rest=dict(speed_max_mm_s=float(speed[i-1:j].max()),net_displacement_mm=rest_span,
        angular_speed_max_deg_s=float(angular[i-1:j].max()),yaw_span_deg=float(np.ptp(yaw)))
    rest['criterion_pass']=(rest['speed_max_mm_s']<=plan['rest_speed_max_mm_s'] and
        rest_span<=plan['rest_net_displacement_max_mm'] and
        rest['angular_speed_max_deg_s']<=plan['rest_angular_speed_max_deg_s'] and
        rest['yaw_span_deg']<=plan['rest_yaw_span_max_deg'])
    return dict(rest=rest,displacement_after_onset_mm=float(np.linalg.norm(pos[-1]-pos[999])),
        path_after_onset_mm=float(np.linalg.norm(np.diff(pos[999:],axis=0),axis=1).sum()),
        speed_max_mm_s=float(speed.max()),forward_applied_max_mm_s=float(trace['command_forward_mm_s'].max()),
        ORN_mean_L=float(trace['ORN_q_L'].mean()),ORN_mean_R=float(trace['ORN_q_R'].mean()),
        DN_release_range=trace['DN_q_actual'].ptp(axis=0).tolist())


def main():
    plan=json.loads((HERE/'PLAN.json').read_text())
    r0,s=read_arm('sham');r1,o=read_arm('odor')
    need(r0['initial_ns']==r1['initial_ns'] and r0['parameters']==r1['parameters'],'Different neural setup')
    need(r0['stored_weight_digest']==r1['stored_weight_digest'] and r0['factors_digest']==r1['factors_digest'],'Different frozen brain')
    prefix=all(np.array_equal(s[key][:1000],o[key][:1000]) for key in s if key not in ('sensores_pendientes','concentracion_campo'))
    need(prefix,'Common prefix diverged before the first consumed odor sample')
    expected=np.zeros((4000,3));expected[1000:3000,:2]=plan['odor_amplitude']
    need(np.array_equal(o['sensores_usados'],expected) and np.all(s['sensores_usados']==0),'Wrong actual consumed pulse')
    need(np.array_equal(s['DN_baseline'],o['DN_baseline']),'Different or drifting DN baseline')
    need(np.all(s['command_yaw_rate_rad_s']==0) and np.all(o['command_yaw_rate_rad_s']==0),'Applied yaw changed')
    metrics={name:arm_metrics(trace,plan) for name,trace in [('sham',s),('odor',o)]}
    dn=o['DN_q_actual']-s['DN_q_actual']
    command=o['command_forward_mm_s']-s['command_forward_mm_s']
    response={}
    for name,lo,hi in [('stimulus_consumed',1000,3000),('poststimulus_consumed',3000,4000)]:
        response[name]=dict(DNg100_difference_LR_max_abs=np.max(abs(dn[lo:hi,:2]),axis=0).tolist(),
            DNg100_difference_LR_integral_model_unit_s=np.sum(dn[lo:hi,:2],axis=0).tolist(),
            command_difference_max_mm_s=float(command[lo:hi].max()),
            command_difference_integral_mm=float(command[lo:hi].sum()*.001))
        response[name]['DNg100_difference_LR_integral_model_unit_s']=(np.sum(dn[lo:hi,:2],axis=0)*.001).tolist()
    velocity_o=np.linalg.norm(o['qvel'][:,:2],axis=1)*10.
    velocity_s=np.linalg.norm(s['qvel'][:,:2],axis=1)*10.
    speed_delta=velocity_o-velocity_s
    duration=plan['movement_sustained_ms']
    eligible=speed_delta>=plan['movement_difference_mm_s']
    sustained=np.convolve(eligible.astype(int),np.ones(duration,dtype=int),mode='valid')==duration
    # The sample at k describes the end of interval k. Start/first consumed
    # boundaries are distinct, and the test cannot precede consumed odor.
    first_motion=np.flatnonzero(sustained & (np.arange(len(sustained))>=1000))
    first_command=np.flatnonzero(command>0)
    displacement_difference=metrics['odor']['displacement_after_onset_mm']-metrics['sham']['displacement_after_onset_mm']
    initiated=all(v['rest']['criterion_pass'] for v in metrics.values()) and bool(len(first_motion)) and displacement_difference>=plan['movement_displacement_difference_mm']
    result=dict(status='COMPLETE_PAIRED_ANALYSIS',simulated_ms_each=4000,common_prefix_exact=True,
        programmed_odor_onset_ms=1000,first_odor_consumed_interval=1001,last_odor_consumed_interval=3000,
        first_positive_command_difference_interval=None if not len(first_command) else int(first_command[0]+1),
        first_sustained_motion_difference_interval=None if not len(first_motion) else int(first_motion[0]+1),
        metrics=metrics,response=response,displacement_difference_mm=displacement_difference,
        operational_initiation_criterion_pass=bool(initiated),neural_robustness_threshold=None,
        interpretation='One pair under a prosthesis with yaw command disabled; quantitative model response, no population statistics, no natural walking or learning claim',
        result_sha256={name:sha(HERE/name/'RESULT.json') for name in ('sham','odor')})
    save(HERE/'RESULTADOS.json',result)
    (HERE/'RESULTADOS.md').write_text('''# Prueba olfativa de iniciación — resultados

Dos brazos de 4 s, olor bilateral DM1 virtual 0,5 vs sham, sin avance basal ni pulso motor; plasticidad deshabilitada.
El soporte y el freno a mando cero pertenecen a la prótesis. Yaw no aplicado en ambos brazos; orientación libre y salida DNb05 registrada.
'''+f'\nCriterio operativo de iniciación: **{initiated}**. Diferencia de desplazamiento: {displacement_difference:.6g} mm.\n'+
        '\nLas unidades neuronales no tienen umbral biológico calibrado. Un fallo del criterio no demuestra ausencia general de respuesta al olor. Consultar RESULTADOS.json para amplitudes, duración, retardos y estado de reposo.\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
