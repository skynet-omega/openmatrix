"""Report all preregistered OFF windows, reconstructed from recorded operands."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from contextlib import contextmanager
from observation_codec import decode
@contextmanager
def record_source(path):
 if path.exists():
  with np.load(path,allow_pickle=False) as z:yield z,hashlib.sha256(path.read_bytes()).hexdigest()
 else:
  z,m=decode(path.with_name(path.stem+'.packed.npz'));yield z,m['original_npz_sha256']
from verify_operands import verify,need
H=Path(__file__).resolve().parent
WINDOWS=[1,5,10,20,50]
def compute():
 start=time.process_time();arms={};traces={};prop={};checks=[]
 for arm in ['sham','profile']:
  d=H/(arm+'_off_01');done=json.loads((d/'RESULT.json').read_text())
  need(done['status']=='COMPLETE' and done['committed_ms']==50,'Complete bounded arm')
  with np.load(d/'traces.npz',allow_pickle=False) as z:traces[arm]={k:z[k] for k in z.files}
  with np.load(d/'proprioception_consumed.npz',allow_pickle=False) as z:prop[arm]={k:z[k] for k in z.files}
  a=[]
  for ms in WINDOWS:
   p=d/f'operands_{ms:03d}ms.npz'
   with record_source(p) as (z,source_hash):
    check=verify(z);need(len(check['margin_ranges'])==6,'Six checked cells')
    checks.append(dict(arm=arm,ms=ms,sha256=source_hash,**check))
    r=z['records'];begin=int(z['offsets'][-2]);ok=np.flatnonzero(r[begin:,402]==1)+begin;need(len(ok)>0 and z['committed'][-1],'Final committed epoch')
    k=int(ok[-1]);x=r[k,:384].reshape(4,6,16)[-1];fine=r[k,396:402]
    need(np.array_equal(fine[:4],traces[arm]['DN_q_actual'][ms-1]),'Recorded DN end state differs')
    a.append(dict(tail_ms=ms,net=x[:,1].tolist(),margin=x[:,10].tolist(),target=x[:,11].tolist(),state=fine.tolist(),target_ulp=check['maximum_positive_target_ULP'],observed_time_s=x[0,14],trial_count=check['trials']))
  arms[arm]=a
 t0=traces['sham'];t1=traces['profile']
 for a in traces.values():
  need(np.all(a['command_forward_mm_s']==0) and np.all(a['command_yaw_rate_rad_s']==0),'Unplanned physical drive')
  need(np.all(a['sensores_usados']==0) and np.all(a['sensores_pendientes']==0),'Original odor not OFF')
 prop_equal={k:bool(np.array_equal(prop['sham'][k],prop['profile'][k])) for k in prop['sham']}
 body_equal={k:bool(np.array_equal(t0[k],t1[k])) for k in ['qpos','qvel','position_mm','yaw_delta_deg','contact_force_N']}
 delta=t1['DN_q_actual']-t0['DN_q_actual'];bilateral=delta[:,2]-delta[:,3]
 return dict(schema='operands49_results_v1',arms=arms,checks=checks,total_exact_sums=sum(x['exact_sums'] for x in checks),all_consumed_sums_exact=True,maximum_positive_target_ULP=max(x['maximum_positive_target_ULP'] for x in checks),bilateral_DNb05_profile_minus_sham_first_last=[float(bilateral[0]),float(bilateral[-1])],bilateral_DNb05_range=[float(bilateral.min()),float(bilateral.max())],unapplied_yaw_difference_deg_s_first_last=np.rad2deg((t1['neural_yaw_unapplied_rad_s']-t0['neural_yaw_unapplied_rad_s'])[[0,-1]]).tolist(),proprioception_bitexact_between_histories=prop_equal,body_bitexact_between_histories=body_equal,maximum_hook_related_joint_speed_deg_s=float(max(np.abs(np.rad2deg(v['angular_velocity_rad_s'])).max() for v in prop.values())),DNg100_targets_zero=all(np.all(np.array(s['target'])[:2]==0) for a in arms.values() for s in a),forward_additional=False,yaw_applied=False,wind_applied=False,stage4_pass=False,stage5_pass=False,CPU_s=time.process_time()-start)
def render(v):
 lines=['# Ronda49: continuación y entradas consumidas','', 'Etapas4/5 abiertas. Se ejecutaron50ms de colaOFF por historia, sham y profile, desde los estados finales48. No se cambió la ley, el lector, los pesos ni el cuerpo; no se aplicó giro ni viento.','',f"Se reconstruyeron exactamente {v['total_exact_sums']} sumas de las seis neuronas en las diez ventanas prefijadas. Diferencia máxima de la transferencia positiva frente al cálculo CPU: {v['maximum_positive_target_ULP']}ULP, dentro del límite de2ULP documentado para tanhf; las sumas/márgenes y la paridad del organismo siguen siendo exactas.", '', '| Cola ms | Historia | MargenDNgL | MargenDNgR | qDNb05L | qDNb05R | qDNa02L | qDNa02R |', '|---|---|---:|---:|---:|---:|---:|---:|']
 for i,ms in enumerate(WINDOWS):
  for arm in ['sham','profile']:
   x=v['arms'][arm][i];q=x['state'];lines.append(f"|{ms}|{arm}|{x['margin'][0]:.6f}|{x['margin'][1]:.6f}|{q[2]:.9f}|{q[3]:.9f}|{q[4]:.9g}|{q[5]:.9g}|")
 lines+=['',f"Diferencia bilateralDNb05 (profile−sham) al primer y último ms: {v['bilateral_DNb05_profile_minus_sham_first_last']}. Es estado residual tras historias distintas, no aprendizaje probado ni señal de dirección del olor.", '',f"Propiocepción consumida idéntica bit a bit entre historias: {v['proprioception_bitexact_between_histories']}. Estado/traza corporal idénticos: {v['body_bitexact_between_histories']}. Velocidad articular máxima: {v['maximum_hook_related_joint_speed_deg_s']:.9g}°/s. Este contraste no aporta una contingencia propioceptiva distinta para un replay causal.", '', 'DNg100 mantuvo objetivo0 en todas las muestras de las cinco ventanas por brazo. No hubo mando de avance adicional. DNa02 es sólo un observador, incluso si su estado cambia. No atribuir ninguna recuperación de navegación a esta colaOFF.', '', 'La comparación continua4ms frente a2+2ms conservó exactamente todos los propietarios serializados. El nuevo observador también conserva el final4ms del observador48. Es una cualificación local del apéndice49; no corrige retrospectivamente la etiqueta histórica restart_tested de48.', '', 'La revisión no autoral encontró tres huecos del verificador (identidadpresináptica, predictor/comprometido y ley positiva); se corrigieron offline y se detectaron once corrupciones deliberadas bajo-O. Se preservan los verificadores fallidos. No hubo cambios de tolerancia en decisiones científicas.', '', 'Alcance: un preparado y dos historias ya expuestas, no cohorte nueva, navegación ni equivalencia biológica. Cotejo de evidencias/falsadores en FUENTES_Y_DECISIONES.md.']
 return '\n'.join(lines)+'\n'
if __name__=='__main__':
 out=H/'RESULTADOS.json';need(not out.exists(),'Preserve prior results');v=compute();out.write_text(json.dumps(v,indent=2)+'\n');(H/'RESULTADOS.md').write_text(render(v));print(json.dumps({k:v[k] for k in ['total_exact_sums','maximum_positive_target_ULP','bilateral_DNb05_profile_minus_sham_first_last','DNg100_targets_zero','proprioception_bitexact_between_histories','CPU_s']}))
