"""Read late behavior and direct PN/DN signed inputs from available PN629-disabled arms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


H=Path(__file__).resolve().parent
ARMS={'sham':'full_sham_01','odor_left':'full_odor_left_01',
      'odor_right':'full_odor_right_01','uniform':'full_uniform_01'}
WINDOWS=(250,300,320,350,400)
IDS=(10176,10208,10360,523769,10065,10118)  # PN R/L, DNa02 R/L, DNb05 R/L.


def read(name):
    folder=H/name
    result=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'traces.npz',allow_pickle=False) as z:
        traces={k:z[k].copy() for k in z.files}
    flow_result=json.loads((folder/'flow/RESULT.json').read_text()) if (folder/'flow/RESULT.json').is_file() else None
    flow=None
    if (folder/'flow/FLOW.npz').is_file():
        with np.load(folder/'flow/FLOW.npz',allow_pickle=False) as z:
            flow={k:z[k].copy() for k in z.files}
        if tuple(int(x) for x in flow['ids'])!=IDS:raise ValueError('Focal native ID order changed')
        lines=[json.loads(line) for line in (folder/'flow/FLOW.jsonl').open()]
        if len(lines)!=len(flow['time_ns']):raise ValueError('Flow JSON and NPZ lengths differ')
        net=[]
        for i,line in enumerate(lines):
            if int(line['time_ns'])!=int(flow['time_ns'][i]) or line['phase']!=str(flow['phase'][i]):
                raise ValueError('Flow JSON and NPZ clocks differ')
            rows=[line['rows'][str(k)] for k in IDS]
            if not np.array_equal(np.asarray([r['raw_signed'] for r in rows]),flow['raw_signed'][i]):
                raise ValueError('Flow JSON and NPZ raw values differ')
            if not np.array_equal(np.asarray([r['target'] for r in rows]),flow['target'][i]):
                raise ValueError('Flow JSON and NPZ targets differ')
            net.append([r['raw_signed']+r['drive']-r['theta'] for r in rows])
        flow['net_before_gain']=np.asarray(net,dtype=np.float64)
    for key,value in traces.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError('Nonfinite trace: '+key)
    if flow is not None and not np.array_equal(flow['time_ns'],traces['CNS_time_ns']):
        raise ValueError('Flow/organism clocks differ')
    if len(traces['fase'])!=result['completed_preparation_ms']+result['completed_trial_ms']:
        raise ValueError('Trace and receipt lengths differ')
    expected={'sham':(0.,0.,0.),'odor_left':(1.,0.,0.),
              'odor_right':(0.,1.,0.),'uniform':(1.,1.,0.)}[result['odor']]
    if not np.all(traces['sensores_usados'][traces['fase']=='preparacion']==0):
        raise ValueError('Odor was present during common preparation')
    if not np.array_equal(traces['sensores_usados'][traces['fase']=='ensayo'],
                          np.tile(expected,(result['completed_trial_ms'],1))):
        raise ValueError('Trial odor exposure differs from assigned condition')
    if flow is not None and len(flow['phase'])!=len(traces['fase']):
        raise ValueError('Native flow and trace lengths differ')
    if not np.array_equal(traces['DN_q_usada'][1:],traces['DN_q_actual'][:-1]):
        raise ValueError('Motor reader lag differs from the recorded neural state')
    dq=traces['DN_q_usada']-traces['DN_baseline']
    command=np.tanh(250*(dq[:,2]-dq[:,3]))*np.deg2rad(5)
    if not np.array_equal(command,traces['command_yaw_rate_rad_s']):
        raise ValueError('Recorded motor command differs from the DNb05 decoder')
    return result,traces,flow_result,flow


def sample(traces,flow,ms):
    indices=np.flatnonzero((traces['fase']=='ensayo')&(traces['paso']==ms))
    if len(indices)!=1:return None
    i=int(indices[0]);x=traces
    r={'yaw_delta_deg':float(x['yaw_delta_deg'][i]),
       'command_yaw_rate_rad_s':float(x['command_yaw_rate_rad_s'][i]),
       'command_integral_deg':float(np.rad2deg(np.sum(x['command_yaw_rate_rad_s'][(x['fase']=='ensayo')&(x['paso']<=ms)])*.001)),
       'concentration_L_R':x['concentracion_campo'][i].tolist(),
       'ORN_mean_q_L':float(np.mean(x['ORN_q_L'][i])),
       'ORN_mean_q_R':float(np.mean(x['ORN_q_R'][i])),
       'PN_q_legacy_L_R':x['PN_q_legacy'][i].tolist(),
       'DN_q_actual':x['DN_q_actual'][i].tolist(),
       'DNb05_used_minus_baseline_L_R':(x['DN_q_usada'][i,2:]-x['DN_baseline'][i,2:]).tolist(),
       'DNb05_used_delta_L_minus_R':float((x['DN_q_usada'][i,2]-x['DN_baseline'][i,2])-
                                       (x['DN_q_usada'][i,3]-x['DN_baseline'][i,3]))}
    r['body_minus_command_deg']=r['yaw_delta_deg']-r['command_integral_deg']
    if flow is not None:
        if flow['phase'][i]!='ensayo':raise ValueError('Flow phase differs from trace')
        raw=flow['raw_signed'][i]
        r['native_raw_synaptic_subtotal_by_ID']={str(k):float(v) for k,v in zip(IDS,raw)}
        r['native_net_before_gain_by_ID']={str(k):float(v) for k,v in zip(IDS,flow['net_before_gain'][i])}
        r['native_PN_target_L_minus_R']=float(flow['target'][i,1]-flow['target'][i,0])
        r['native_DNa02_target_L_R']=[float(flow['target'][i,3]),float(flow['target'][i,2])]
        r['native_DNb05_target_L_minus_R']=float(flow['target'][i,5]-flow['target'][i,4])
        r['native_DNb05_raw_L_minus_R']=float(raw[5]-raw[4])
        r['native_PN_L_minus_R']=float(raw[1]-raw[0])
        r['native_DNa02_L_minus_R']=float(raw[3]-raw[2])
    return r


def main():
    arms={k:read(v) for k,v in ARMS.items() if (H/v/'RESULT.json').is_file()}
    result={'schema':'stage3_pn629_late_diagnostic_v1',
            'windows_ms_from_odor_on':list(WINDOWS),'stage3_admission':False,
            'model_confounds':['PN10208 general refinement disabled, 466 dynamic routes retained',
                               'DN baseline not reset at odor onset',
                               '400-ms numerical fidelity not certified'],
            'arms':{},'preparation_exact':{},'contrasts_to_sham':{}}
    if 'sham' in arms:
        base=arms['sham'][1]
        for name,(_,trace,_,_) in arms.items():
            changed=[]
            if set(base)!=set(trace):raise ValueError('Trace schema differs between arms')
            for key in base:
                if not np.array_equal(base[key][:40],trace[key][:40],equal_nan=base[key].dtype.kind in 'fc'):
                    changed.append(key)
            result['preparation_exact'][name]={'equal':not changed,'changed_fields':changed}
    for name,(receipt,trace,flow_receipt,flow) in arms.items():
        windows={str(ms):sample(trace,flow,ms) for ms in WINDOWS}
        trial=trace['fase']=='ensayo'
        command_integral=float(np.rad2deg(np.sum(trace['command_yaw_rate_rad_s'][trial])*.001))
        dn_delta=(trace['DN_q_usada'][:,2]-trace['DN_baseline'][:,2])-(
                  trace['DN_q_usada'][:,3]-trace['DN_baseline'][:,3])
        result['arms'][name]={'status':receipt['status'],
                             'completed_preparation_ms':receipt['completed_preparation_ms'],
                             'completed_trial_ms':receipt['completed_trial_ms'],
                             'wall_total_s':receipt['wall_total_s'],
                             'error':receipt['error']['message'] if receipt['error'] else None,
                             'native_flow_samples':flow_receipt['samples'] if flow_receipt else None,
                             'native_target_max_error':flow_receipt['max_target_error'] if flow_receipt else None,
                             'native_rate_max_error':flow_receipt['max_rate_error'] if flow_receipt else None,
                             'motor_reader_ids_DNb05_L_R':[10118,10065],
                             'motor_command_and_one_ms_lag_exact':True,
                             'odor_exposure_exact':True,
                             'command_integral_trial_deg':command_integral,
                             'DNb05_used_delta_L_minus_R_mean_250_to_400':float(np.mean(dn_delta[trial&(trace['paso']>=250)&(trace['paso']<=400)])) if np.any(trial&(trace['paso']>=250)&(trace['paso']<=400)) else None,
                             'DNa02_native_target_nonzero_samples':int(np.count_nonzero(flow['target'][:,2:4])) if flow is not None else None,
                             'windows':windows}
    if 'sham' in result['arms']:
        for name in result['arms']:
            if name=='sham':continue
            contrasts={}
            for ms in WINDOWS:
                a=result['arms'][name]['windows'][str(ms)]
                b=result['arms']['sham']['windows'][str(ms)]
                if a is not None and b is not None:
                    contrasts[str(ms)]={'yaw_minus_sham_deg':a['yaw_delta_deg']-b['yaw_delta_deg'],
                                        'command_integral_minus_sham_deg':a['command_integral_deg']-b['command_integral_deg'],
                                        'DNb05_used_delta_LR_minus_sham':a['DNb05_used_delta_L_minus_R']-b['DNb05_used_delta_L_minus_R'],
                                        'native_PN_target_LR_minus_sham':a['native_PN_target_L_minus_R']-b['native_PN_target_L_minus_R']
                                        if 'native_PN_target_L_minus_R' in a and 'native_PN_target_L_minus_R' in b else None,
                                        'native_DNa02_LR_minus_sham':a['native_DNa02_L_minus_R']-b['native_DNa02_L_minus_R']
                                        if 'native_DNa02_L_minus_R' in a and 'native_DNa02_L_minus_R' in b else None}
            result['contrasts_to_sham'][name]=contrasts
    (H/'LONG_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'arms':{k:(v['status'],v['completed_trial_ms']) for k,v in result['arms'].items()},
                      'preparation_exact':result['preparation_exact'],
                      'yaw_320':{k:(v['windows']['320']['yaw_delta_deg'] if v['windows']['320'] else None) for k,v in result['arms'].items()}},ensure_ascii=False))


if __name__=='__main__':main()
