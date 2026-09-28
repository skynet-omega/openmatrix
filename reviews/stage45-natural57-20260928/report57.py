"""Generate numeric report and scientific figure only from verified57 outputs."""
from pathlib import Path
import json,time,os
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'cache/matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def main():
 start=time.process_time();r=json.loads((H/'RESULTADOS.json').read_text());cost=json.loads((H/'VERIFY_COST.json').read_text());p=json.loads((H/'PLAN.json').read_text());m=r['means_primary_window'];c=r['contrasts'];signals=r['signals'];b=r['budget']
 if (H/'RESULTADOS.md').exists():raise ValueError('immutable report')
 verdict='Supera la criba diagnóstica congelada' if r['screen_supported'] else 'No supera la criba diagnóstica congelada'
 rows='\n'.join('| '+s+' | '+f"{m['natural_'+s]['mean_yaw_deg_s']:.9f} | {m['natural_'+s]['mean_forward_mm_s']:.9f} | {r['arms']['natural_'+s]['DNg_target_max_all_RHS']:.9g} | "+', '.join(f'{x:.9f}' for x in m['natural_'+s]['final_error_L_R_deg'])+' |' for s in ['none','L','R'])
 lines=[]
 for name,v in signals.items():
  def fmt(x):return 'no definido' if x is None else f'{x:.7g}'
  lines.append(f"| {name} | {v['early']['mean_lateral_norm']:.7g} | {v['late']['mean_lateral_norm']:.7g} | {fmt(v['projection_late_over_early'])} | {fmt(v['cosine_early_late'])} |")
 text=f'''# Campaña57 — señal sensorial natural y giro

**{verdict}. {r['classification']}, exclusivamente para promoción desde este preparado de 384 ms. Etapas 4/5 abiertas.** No hubo viento ni comparación reservada de feedback. La intervención independiente es la posición de la fuente; observar una frontera no prueba por sí solo su mediación causal.

Tres vidas emparejadas desde 48/sham, 384 ms cada una y prefijo basal de 10 ms. Ley padre, cuerpo, lector y perfil sensorial de 54 conservados. Cualificación de 2 ms exacta frente a 56; tres prefijos de 89 ms exactos frente a 54. Nuevo registro sólo lectura de 686 PN y 694 ORN en la frontera CSR, actividad comprometida, filtros DM1 y salidas finas.

## Resultado descendente y corporal

Medias257–384ms. Los errores finales se evalúan siempre respecto de ambas fuentes, incluida la trayectoria sin olor.

| Fuente | Giro solicitado(°/s) | Avance solicitado(mm/s) | Máximo targetDNg100, todosRHS | Error finalL,R(°) |
|---|---:|---:|---:|---|
{rows}

SemidiferenciaL−R: DNb05 **{c['half_L_minus_R_DNb_q']:.10g}q**, giro **{c['half_L_minus_R_yaw_deg_s']:.10g}°/s**. Mínimos conservados1.6e−5q y0.02°/s; materialidad conjunta={c['neural_and_yaw_material']}. Componente común de giro frente a none: **{c['common_yaw_source_minus_none_deg_s']:.10g}°/s**. Signo del contraste compatible con fuentes={c['direction_compatible_half_contrast']}.

Beneficio de error frente a la trayectoria sin olor, para cada fuente: **L {c['body_error_benefit_vs_none_L_R_deg'][0]:+.10g}°; R {c['body_error_benefit_vs_none_L_R_deg'][1]:+.10g}°**. Ambos positivos={c['both_mirror_benefits_positive']}. Un contraste motor material no sustituye este resultado corporal. El mando de avance y el desplazamiento físico son observables distintos: todas las poses, fuerzas y desplazamientos están conservados.

## Señal natural registrada

Contraste fuenteL−fuenteR dividido por2. La plantilla es su vector medio51–89ms; se proyecta después257–384ms. Son estadísticas descriptivas de la misma vida, sin selección de células ni lector operativo. Las normas de fronteras con unidades distintas no son ganancias fisiológicas comparables.

| Frontera | Norma media temprana | Norma media tardía | Proyección tardía/temprana | Coseno de vectores medios |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

Se publican también el componente común[(L+R)/2−none], contrastes individualesL−none/R−none, curvas y resúmenes discretos por1ms. Los testigosCSR contienenfirst/last/min/max/count, incluyendo evaluaciones predictoras/rechazadas;8llamadas adicionales de calentamiento sólo en el primer ms. No se reconstruye una integral física de todas las entradas, ni se equiparaCSR con el consumidor especializado dePN. El proxy periférico utiliza concentraciones geométricas: no es transducción química calibrada. La anatomía323L/371R conserva una diferencia inicial de dosis total aproximada0.77%; no se aísla dirección pura de cantidad.

## Comprobaciones y presupuesto

{b['attempted_CNS_ms']}msCNS intentados y{b['committed_CNS_ms']} comprometidos; {b['worker_CPU_s']:.6f}sCPU de trabajadores y{b['queue_wall_s']:.6f}s de cola, dentro1200/6000/5000. El tiempo total de investigación y cierre es mayor y se registra por separado. Reloj corporal verificado por40sumas exactas25µs por ms. Extensión del guard periférico3200→3384; no se amplió el cargador49 ni se cualificó reanudaciónGPU desde el final.

El verificador reconstruyó resultados y rechazó **{len(cost['tests'])}corrupciones deliberadas bajo−O**. Dos reparaciones analíticas están documentadas: incluir las 8 llamadas iniciales de calentamiento, y comparar la geometría latente del prefijo contra su propia fuente en vez de exigir igualdad entre fuentes distintas. La concentración consumida y todos los estados del prefijo sí coinciden exactamente. Se conservaron fallos y fuentes previas, sin cambiar el contrato ni repetir CNS. ContratoSHA`{json.loads((H/'FREEZE.json').read_text())['plan_sha256']}`.

Una preparación expuesta, sin cohorte nueva, no permite atribuir robustez entre organismos ni equivalencia biológica. Las revisiones conceptuales de los dos ChatGPT y la revisión de archivos de Motor se identifican por separado. [Decisión y alternativas](DECISION.md), [fuentes y límites](FUENTES_Y_DECISIONES.md), [reproducción](REPRODUCIR.md).

![Comparación de las tres fuentes](COMPARACION.png)
'''
 (H/'RESULTADOS.md').write_text(text,encoding='utf-8')
 with np.load(H/'CURVES.npz') as z:curves={k:z[k].copy() for k in z.files}
 traces={};neural={}
 for s in ['none','L','R']:
  with np.load(H/('natural_'+s)/'traces.npz') as z:traces[s]={k:z[k].copy() for k in ['spatial_concentration_used','command_yaw_rate_rad_s']}
  with np.load(H/('natural_'+s)/'PN_consumed.npz') as z:neural[s]=z['first'].astype(float)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained');x=np.arange(1,385);colors={'none':'#666666','L':'#007b9a','R':'#d25a24'}
 for s in ['L','R']:
  v=traces[s]['spatial_concentration_used'];axs[0,0].plot(x,v[:,0]-v[:,1],label='Fuente '+s,color=colors[s])
 axs[0,0].set(title='Diferencia entre antenas',ylabel='Concentración L−R');axs[0,0].legend(frameon=False)
 norm=np.linalg.norm(neural['none'],axis=1)
 for k,label,col in [('lateral_norm','Norma de (L−R)/2','#007b9a'),('common_norm','Norma común vs basal','#8f4a8b')]:axs[0,1].plot(x,100*curves['PN_generic_first_RHS__'+k]/norm,label=label,color=col)
 axs[0,1].set(title='PN genérica realmente observada',ylabel='% de norma del vector basal');axs[0,1].legend(frameon=False)
 for s in ['none','L','R']:
  axs[1,0].plot(x,curves['natural_'+s+'__DNb_difference'],label=s,color=colors[s]);axs[1,1].plot(x,curves['natural_'+s+'__yaw_deg_s'],label=s,color=colors[s])
 axs[1,0].set(title='Diferencia bilateral DNb05',ylabel='qL−qR');axs[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axs[1,0].legend(frameon=False)
 axs[1,1].set(title='Mando de giro aplicado',ylabel='°/s');axs[1,1].legend(frameon=False)
 for i,s in enumerate(['L','R']):
  axs[2,0].plot(x,np.abs(curves['natural_'+s+'__error_signed_deg'][:,i]),label='Fuente '+s,color=colors[s]);axs[2,0].plot(x,np.abs(curves['natural_none__error_signed_deg'][:,i]),ls='--',color=colors[s],label='Sin olor, error a '+s)
 axs[2,0].set(title='Error real hacia cada fuente',ylabel='°');axs[2,0].legend(frameon=False,fontsize=8)
 keys=['ORN_generic_first_RHS','ORN_q_committed','PN_generic_first_RHS','PN_q_committed'];labels=['ORN terminal','ORN actividad','PN terminal','PN actividad'];vals=[signals[k]['projection_late_over_early'] for k in keys]
 axs[2,1].bar(labels,[np.nan if v is None else v for v in vals],color=['#999999','#999999','#007b9a','#007b9a']);axs[2,1].axhline(1,color='black',ls=':',lw=1);axs[2,1].set(title='Conservación del patrón temprano',ylabel='Proyección tardía / temprana');axs[2,1].tick_params(axis='x',labelrotation=15)
 for ax in list(axs.flat)[:-1]:
  ax.axvspan(51,89,color='grey',alpha=.07);ax.axvspan(257,384,color='#007b9a',alpha=.07);ax.set_xlabel('Tiempo desde reanudación(ms)');ax.grid(alpha=.15)
 fig.suptitle('Campaña57 · olor espacial natural, ley padre y cuerpo activo\n384ms por condición · diagnóstico de una preparación expuesta',fontsize=15)
 fig.savefig(H/'COMPARACION.png',dpi=160);fig.savefig(H/'COMPARACION.pdf');plt.close(fig)
 (H/'REPORT_COST.json').write_text(json.dumps(dict(CPU_s=time.process_time()-start))+'\n')
 print(json.dumps(dict(report=str(H/'RESULTADOS.md'),CPU_s=time.process_time()-start)))
if __name__=='__main__':main()
