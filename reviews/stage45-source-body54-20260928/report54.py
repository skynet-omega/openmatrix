from pathlib import Path
import json,sys,os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify54 import compute,yaw
H=Path(__file__).resolve().parent

def main():
 r,data=compute(H/'repair02');existing=json.loads((H/'repair02/RESULTADOS.json').read_text())
 if r!=existing:raise ValueError('unverified report inputs')
 if (H/'RESULTADOS.md').exists():raise ValueError('preserve report')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'parent':'#296d9c','I':'#b54c36'};styles={'none':':','L':'-','R':'--'}
 for arm,v in data.items():
  law,side=arm.split('_');t=v['t'];x=np.arange(1,90);col=colors[law];sty=styles[side]
  ax[0,0].plot(x,np.rad2deg(t['command_yaw_rate_rad_s']),color=col,ls=sty,label=arm)
  q0=t['spatial_sample_qpos'][0];ax[0,1].plot(x,np.rad2deg(yaw(t['qpos'])-yaw(q0)),color=col,ls=sty)
  ax[1,0].plot(x,t['command_forward_mm_s'],color=col,ls=sty)
 for a,title,label in [(ax[0,0],'Giro neural aplicado','°/s'),(ax[0,1],'Rotación corporal desde inicio','°'),(ax[1,0],'Avance solicitado','mm/s')]:
  a.set(title=title,xlabel='Tiempo adicional (ms)',ylabel=label);a.grid(alpha=.15)
 ax[0,0].legend(ncol=2,fontsize=8)
 for j,law in enumerate(['parent','I']):
  b=r['comparisons'][law]['body_error_benefit_vs_none_L_R_deg'];ax[1,1].bar(np.arange(2)+(.18 if j else -.18),b,.34,color=colors[law],label=law)
 ax[1,1].set(title='Beneficio del olor frente a sin olor',xticks=[0,1],xticklabels=['Fuente izquierda','Fuente derecha'],ylabel='Reducción del error angular (°)');ax[1,1].axhline(0,color='#333',lw=.8);ax[1,1].legend();ax[1,1].ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax[1,1].grid(axis='y',alpha=.15)
 fig.suptitle('Campaña54 · seis vidas de89ms · ley padre e I (control de corriente)\nCuerpo y sensores acoplados; no prueba de admisión4/5',fontsize=14)
 fig.savefig(H/'COMPARACION.png',dpi=165);plt.close(fig)
 lines=['# Campaña54 — olor espacial y cuerpo con giro aplicado','',
 '**Etapas4/5 abiertas.** Piloto de desarrollo desde el mismo estado48, con fuente fija en el mundo, lectura bilateral de antenas cada1ms, avance y giro neurales aplicados. Dos leyes (padre e I/control de corriente) y tres condiciones (sin olor, fuente izquierda, fuente derecha).89ms por brazo, dos cualificaciones repetidas de2ms. No es una prueba de recuperación ni una cohorte confirmatoria.','',
 '![Comparación](COMPARACION.png)','',
 '|Brazo|Avance medio(mm/s)|Giro medio(°/s)|Rotación corporal(°)|Máximo objetivoDNg100|',
 '|---|---:|---:|---:|---:|']
 for arm,v in r['arms'].items():lines.append(f"|{arm}|{v['mean_forward_mm_s']:.9g}|{v['mean_yaw_deg_s']:.9g}|{v['physical_yaw_change_deg']:.9g}|{v['DNg_target_max_all_RHS']:.9g}|")
 lines+=['','Medias de mando en51–89ms; rotación entre inicio y fin; máximoDNg100 incluye evaluaciones intermedias/rechazadas del integrador. DNg100 es estado/objetivo normalizado, no un recuento de espigas.','',
 '|Ley|SemidiferenciaL−R DNb05(q)|SemidiferenciaL−R yaw(°/s)|BeneficioL vs none(°)|BeneficioR vs none(°)|Criba para vida larga|',
 '|---|---:|---:|---:|---:|---|']
 for law,v in r['comparisons'].items():
  b=v['body_error_benefit_vs_none_L_R_deg'];label='PROMETEDOR_NO_CONFIRMADO' if v['longer_life_screen_supported'] else 'DESCARTADO en esta criba'
  lines.append(f"|{law}|{v['half_L_minus_R_DNb_q']:.9g}|{v['half_L_minus_R_yaw_deg_s']:.9g}|{b[0]:.9g}|{b[1]:.9g}|{label}|")
 lines+=['','Positivo en beneficio significa menor error que el control sin olor de la misma ley, evaluado contra la misma fuente. No basta una sola fuente favorable. Mínimos neuronales conservados:1,6e−5q y0,02°/s. Cambiar dirección y cantidad totalORN sigue parcialmente confundido por323ORN izquierdas frente a371 derechas; no se normalizó después de observar resultados.','',
 'Saturación nueva elegible I frente a padre: '+', '.join(f"{s}={v['new_saturated_eligible_fraction']:.6g}" for s,v in r['saturation'].items())+'. El detalle de objetivos base fuera de0..1 queda en RESULTADOS.json; no se confunde ese objetivo previo a overrides con el estado final.', '',
 'El primer intento conserva un fallo de unidades en el adaptador nuevo: entregaba rad/s a una entrada de giro normalizada. Se detuvo antes del primer ms comprometido del brazo científico. La reparación sólo codifica la señal de entrada correcta y reproduce exactamente360 mandos observados; la ganancia histórica5°/s permanece. Se repitieron ambas cualificaciones y se redujo cada brazo de90a89ms para respetar el presupuesto agregado.', '',
 f"Coste de ambos intentos: {r['budget']['attempted_CNS_ms']}msCNS intentados, {r['budget']['committed_CNS_ms']}ms comprometidos; {r['budget']['worker_CPU_s']:.3f}sCPU de trabajadores, {r['budget']['queue_wall_s']:.3f}s de cola. Topes originales544ms,3300sCPU,3000s de cola. Análisis/entregaCPU separados y sin nueva integración.", '',
 'Verificación: todos los campos comunes de cualificación iguales a52; lector, reloj, latencia, geometría, tasas nominales y métricas reconstruidos desde arrays. '+r['verification_limit'], '',
 'Reproducción corta: `python -B -O check_delivery54.py --corruptions`. Reproducción completa del análisis guardado: `python -B -O verify54.py` (incluida también en la comprobación corta). Los estados completos se conservan en la cápsula local; no se cualificó reanudaciónGPU portable. No relanzar run_queue.py en estos directorios.', '',
 'Datos de cada instante, cantidades y errores absolutos: [repair02/RESULTADOS.json](repair02/RESULTADOS.json). Contrato y fallo original intactos; contrato reparado SHA25686ddc8e645357b9063aeee39bd57238865422f77aef7e3837821ab152d48ae04. Motivo de parada: exposición acotada completada, sin ampliar hasta obtener un resultado favorable. Interpretación y próxima ruta: [DECISION.md](DECISION.md).']
 (H/'RESULTADOS.md').write_text('\n'.join(lines)+'\n')
 print('figure and mechanically generated report saved')
if __name__=='__main__':main()
