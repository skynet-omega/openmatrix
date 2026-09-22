"""Reconstruct fixed common-window behavior and interface contrasts from traces."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
CONDITIONS=['sham','uniform','odor_left','odor_right']

def read(condition):
 folder=HERE/('pilot_'+condition+'_01')
 result=json.loads((folder/'RESULT.json').read_text())
 with np.load(folder/'traces.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
 for k,x in trace.items():
  if x.dtype.kind in 'fi' and not np.isfinite(x).all():raise ValueError('Nonfinite trace '+condition+'/'+k)
 prep=trace['fase']=='preparacion';trial=trace['fase']=='ensayo'
 if prep.sum()!=40 or trial.sum()<100 or np.any(trace['sensores_usados'][prep]!=0):raise ValueError('Incomplete common-window preparation/trajectory')
 if not np.array_equal(trace['paso'][trial],np.arange(1,trial.sum()+1)):raise ValueError('Trial row/clock order changed')
 clocks=trace['CNS_time_ns']
 if not np.array_equal(clocks,trace['PN_time_ns']) or not np.array_equal(clocks,trace['body_time_ns']) or not np.all(np.diff(clocks)==1_000_000):raise ValueError('Physical clock mismatch')
 prep_hash={name:hashlib.sha256((folder/'prepared_state'/name).read_bytes()).hexdigest() for name in ['session.json','session.npz','prosthesis.json','prosthesis.npz','published.json','published.npz','boundary.json']}
 return result,{k:x[trial][:100] for k,x in trace.items()},prep_hash

def main():
 runs={k:read(k) for k in CONDITIONS};data={k:x[1] for k,x in runs.items()}
 common_prep=all(runs[k][2]==runs['sham'][2] for k in CONDITIONS)
 if not common_prep:raise ValueError('Prepared state differs; resolve before paired causal interpretation')
 rows={}
 for k,(result,t,_) in runs.items():
  y=float(t['yaw_delta_deg'][-1]);y0=float(data['sham']['yaw_delta_deg'][-1])
  rows[k]={'yaw_100ms_deg':y,'difference_from_sham_deg':y-y0,'directional_angular_check':(y>=.02 if k=='odor_left' else y<=-.02 if k=='odor_right' else abs(y)<=.02 if k=='sham' else None),'actual_exposure_sum_per_antenna':t['sensores_usados'][:,:2].sum(axis=0).tolist(),'last_command_yaw_rad_s':float(t['command_yaw_rate_rad_s'][-1]),'min_upright':float(t['upright'].min()),'completed_trial_ms':result['completed_trial_ms'],'requested_trial_ms':result['requested_trial_ms'],'status':result['status'],'step_wall_s':result['step_wall_s'],'process_wall_s':result['wall_total_s']}
 signals=['ORN_q_L','ORN_q_R','ORN_filters','PN_general_transmission','PN_gamma_nS','PN_additional_nS','central_q','central_transmission','DN_q_actual','DN_q_usada','command_yaw_rate_rad_s','yaw_delta_deg']
 contrasts={}
 for key in signals:
  x=data['odor_left'][key]-data['odor_right'][key]
  contrasts[key]={'left_right_max_abs':float(np.max(abs(x))),'left_right_rms':float(np.sqrt(np.mean(x*x))),'left_sham_max_abs':float(np.max(abs(data['odor_left'][key]-data['sham'][key]))),'right_sham_max_abs':float(np.max(abs(data['odor_right'][key]-data['sham'][key]))),'uniform_sham_max_abs':float(np.max(abs(data['uniform'][key]-data['sham'][key])))}
 result={'common_window_ms':100,'same_prepared_state_hashes':common_prep,'conditions':rows,'interface_contrasts':contrasts,'stage3_admission':False,'historical_threshold_only':{'turn_deg':.02,'sham_abs_deg':.02,'caution':'Only historical angular checks. Current stimulus begins at trial0ms versus5ms in the older preparation protocol. Engineering reader/tau interventions remain. Not a full stage3 admission contract.'},'scientific_scope':'One deterministic life/checkpoint per condition; no independent biological seeds or biological validation; short numerical transport does not certify cumulative long-time accuracy.'}
 (HERE/'PILOT_VERIFIED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10})
 fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
 labels={'sham':'Sin olor','uniform':'Uniforme','odor_left':'Izquierda','odor_right':'Derecha'}
 for k,t in data.items():
  axes[0].plot(t['paso'],t['yaw_delta_deg'],label=labels[k]);axes[1].plot(t['paso'],t['command_yaw_rate_rad_s']);axes[2].plot(t['paso'],t['DN_q_usada'][:,2]-t['DN_q_usada'][:,3])
 axes[0].axhline(.02,color='gray',ls=':',lw=.8);axes[0].axhline(-.02,color='gray',ls=':',lw=.8)
 axes[0].set_ylabel('Giro acumulado (grados)');axes[0].legend(ncol=4,loc='best')
 axes[1].set_ylabel('Mando de giro (rad/s)');axes[2].set_ylabel('DNb05 consumida: L−R');axes[2].set_xlabel('Tiempo desde inicio del ensayo (ms)')
 fig.suptitle('Etapa 3 — mismo organismo preparado, cuatro condiciones\nEjecución integrada; ventana común de 100 ms')
 for ax in axes:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(HERE/'ETAPA3_TRAYECTORIAS.png',dpi=160);plt.close(fig)
 print(json.dumps({'same_preparation':common_prep,'conditions':rows,'contrasts':contrasts},allow_nan=False))

if __name__=='__main__':main()
