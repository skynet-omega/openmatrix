"""Render measured results; interpretation remains separately labelled."""
from pathlib import Path
import json,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify55 import load,compute,need
H=Path(__file__).resolve().parent
def main():
 start=time.process_time();d=load(H);r=compute(d);need(r==json.loads((H/'RESULTADOS.json').read_text()),'verified metrics required');c=r['C'];a=r['A'];b=r['budget']
 pn_names=[n for n in r['arms'] if n.startswith('pn_')]
 both_outputs=all(v['neural_and_motor_thresholds_met'] and v['toward_other_donor'] for v in a['contrasts'].values())
 input_text=' / '.join(f"{100*v['consumed_PN_relative_L2']:.3f}%" for v in a['contrasts'].values())
 maximum_target=max(r['arms'][n]['DNg_target_max_all_RHS'] for n in pn_names)
 maximum_forward=max(float(d['arms'][n]['trace']['command_forward_mm_s'].max()) for n in pn_names)
 measured_note=f"Criterios propios de DN/giro y dirección hacia el donante cumplidos en ambos sentidos: **{'sí' if both_outputs else 'no'}**. Los giros medios absolutos siguen siendo positivos: la reciprocidad corresponde a cambios respecto a cada control, no a giros absolutos opuestos. La separaciónPN agregada es {input_text}, frente al mínimo prospectivo de {100*d['plans']['PN']['criteria']['consumed_PN_relative_change_min']:.0f}%. Criba conjunta: **{a['reciprocal_material_transport']}**. Máximo target DNg100 en los cuatro brazosPN: {maximum_target:.9g}; máximo mando de avance: {maximum_forward:.9g}mm/s. Se mantienen los criterios originales; un fallo de promoción no borra un efecto causal medido."
 fig,axs=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True);color=['#2a6f97','#bd5636']
 for name,label,col in zip(['feedback_online','feedback_replay'],['JO según cuerpo actual','JO grabado antes del giro'],color):
  tr=d['arms'][name]['trace'];axs[0,0].plot(np.arange(1,90),np.rad2deg(tr['command_yaw_rate_rad_s']),label=label,color=col)
 axs[0,0].plot(np.arange(1,90),np.rad2deg(d['reference']['JO52_nominal']['yaw_raw_rad_s']),label='Referencia52, giro no aplicado',color='#555',linestyle=':');axs[0,0].axvspan(51,89,color='#ddd',alpha=.3);axs[0,0].set(title='C · Cambiar sólo JO cambia el mando',xlabel='Continuación (ms)',ylabel='Giro solicitado (°/s)');axs[0,0].legend(fontsize=8)
 jo=[d['arms'][name]['neural']['JO_injected'] for name in ['feedback_online','feedback_replay']];sep=np.linalg.norm(jo[0]-jo[1],axis=1)
 axs[0,1].plot(np.arange(1,90),sep,color='#7d4b91');axs[0,1].set(title='C · Separación efectiva de las entradas',xlabel='Continuación (ms)',ylabel='Norma L2 de la diferencia JO (unidad del modelo)')
 for receiver,donor,col in [('sham','profile',color[0]),('profile','sham',color[1])]:
  base=d['arms']['pn_'+receiver+'_from_'+receiver];cross=d['arms']['pn_'+receiver+'_from_'+donor];label=donor+' → '+receiver
  delta=np.rad2deg(cross['trace']['command_yaw_rate_rad_s']-base['trace']['command_yaw_rate_rad_s']);axs[1,0].plot(np.arange(1,129),delta,label=label,color=col)
  delta=np.linalg.norm(cross['consumed']['first']-base['consumed']['first'],axis=1);axs[1,1].plot(np.arange(1,129),delta,label=label,color=col)
 for ax in axs[1]:ax.axvspan(65,128,color='#ddd',alpha=.3);ax.legend(fontsize=8)
 axs[1,0].set(title='A · Efecto del injerto PN parcial',xlabel='Continuación (ms)',ylabel='Giro injerto − propio (°/s)');axs[1,0].axhline(.02,color='#999',linestyle=':',linewidth=.7);axs[1,0].axhline(-.02,color='#999',linestyle=':',linewidth=.7)
 axs[1,1].set(title='A · Salida PN consumida por el operador',xlabel='Continuación (ms)',ylabel='Norma L2: primer RHS de cada ms')
 for ax in axs.flat:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.suptitle('Campaña55 · Intervenciones causales en una preparación expuesta\nEtapas4/5 abiertas; sin ensayo de viento ni admisión de navegación',fontsize=13);fig.savefig(H/'COMPARACION.png',dpi=160);plt.close(fig)
 lines=['# Campaña55 — resultados verificados','',
 '**Las etapas4/5 siguen abiertas.** Se ejecutaron dos instrumentos causales, con controles exactos previos. No se cambió ningún peso, umbral ni ganancia del lector. I es el control de corriente con shunt basal, no una ley instantánea fisiológicamente validada. Se usó sólo en el diagnóstico JO; el intercambio PN conserva la ley original.','',
 '## C: feedback corporal por JO','',
 f'En la ventana51–89ms, cambiar sólo la entrada JO lleva el giro medio de **{c["online_mean_yaw_deg_s"]:.6f} a {c["replay_mean_yaw_deg_s"]:.6f}°/s**. La referencia52, anterior a aplicar el giro al cuerpo, era {c["nominal52_mean_yaw_deg_s"]:.6f}°/s. El cambio es {c["replay_minus_online_yaw_deg_s"]:+.6f}°/s y {c["replay_minus_online_DNb_L_minus_R_q"]:+.9f}q en DNb05 L−R; ambos superan los mínimos diagnósticos congelados.',
 f'La separación relativa L2 de JO fue {c["JO_relative_L2"]:.6f}. Se reduce {100*c["fraction_mean_yaw_gap_recovered"]:.2f}% de la distancia entre estas medias de giro. **Ese porcentaje no es una fracción causal del cerebro ni una validación de navegación.** Queda un residuo de {c["nominal52_residual_yaw_deg_s"]:+.6f}°/s. Los registros52/54 ya diferían ligeramente antes de activar el extraJO.',
 'El control online reproduce los36 campos de54 y las166700 salidas finales exactamente. La cinta procede de52, obtenida antes de la intervención corporal; no es la propia cinta online desde el mismo estado. Sólo se sustituye el extraJO declarado; se conserva la entrada nativa. Otros sensores siguen sus políticas online y pueden cambiar como consecuencia de la divergencia corporal. Dosis y distribución JO cambian conjuntamente: este experimento no identifica selectividad direccional independiente de cantidad.','',
 '## A: frontera PN parcial','',
 'Se intercambiaron una sola vez1372 coordenadas: actividad y filtro común de686ALPN, entre los estados alcanzados por48/sham y48/profile al mismo reloj. Se conservaron PN fina10208, entradas/receptores, memorias PN→KC, DN, cuerpo, pesos y lector del receptor. Los controles vivos de identidad de ambos receptores fueron exactos antes de los cruces.','',
 '| Receptor ← donante | Δgiro medio65–128ms (°/s) | ΔDNb L−R (q) | Diferencia relativa PN consumida | Avance medio con injerto (mm/s) |',
 '|---|---:|---:|---:|---:|']
 for name,v in a['contrasts'].items():lines.append(f'| {name} | {v["yaw_change_deg_s"]:+.8f} | {v["DNb_L_minus_R_change_q"]:+.8g} | {v["consumed_PN_relative_L2"]:.6g} | {v["mean_forward_cross_mm_s"]:.8g} |')
 lines.extend(['',f'Transporte recíproco material según contrato: **{a["reciprocal_material_transport"]}**. Clasificación de esta criba: **{a["classification"]}**. El resultado se refiere sólo a esta frontera y ventana, no a la existencia de toda ruta PN. Los máximos transitorios y sus instantes están en RESULTADOS.json, separados de la ventana primaria.','',
 'El primer injerto cruzado falló antes de completar1ms por una discontinuidad no declarada en siete entradas GABA. La reparación añadió un evento explícito con retardo125µs; conservó ocupación e historia previa y repitió los dos controles de identidad. No cambió la ley neuronal, pesos, criterios ni ventanas. El intento y su coste siguen contabilizados. [Reparación y prueba matemática](REPARACION.md).','', measured_note,'', '## B y decisión','',
 'La cinética ORN→PN existente ya contiene recursos y depresión de dos componentes. Se conserva la alternativa de contrastar transferencia temporal con observables locales, sin añadir una nueva ley por analogía. [Datos, artículo y decisión](B_DATOS.md). Esta ronda no implementa un tercer organismo. [Interpretación y siguiente discriminador](DECISION.md).','',
 '## Evidencia y límites','',
 f'Consumidos **{b["attempted_CNS_ms"]}ms CNS intentados/{b["committed_CNS_ms"]} comprometidos**, {b["worker_CPU_s"]:.3f}sCPU de trabajadores y {b["queue_wall_s"]:.3f}s de cola acumulada. Topes:800ms,4000sCPU,3600scola. Dos instrumentos, seis brazos científicos, seis controles cortos y un intento fallido preservado. No hubo selección por semillas; es una preparación de desarrollo ya expuesta, no confirmación reservada.',
 'Motor C++/CUDA reconstruyó independientemente JO y los cuatro brazos PN desde los registros, sin ejecutar el verificador autoral: cifras, ventana y lector coinciden exactamente. Su revisión distingue el efecto causal medido del fallo de promoción y de la ausencia de avance; véanse aporte_motor/REVISION_JO55.md y aporte_motor/REVISION_PN55.md en el ZIP. ChatGPT ASTRA_V2 y ChatGPT_Motor_V2 contribuyeron restricciones/propuestas; no se les atribuye ejecución local ni modo PRO verificado. Las respuestas y la delimitación de propietarios están conservadas.',
 'El verificador reconstruye cifras, criterios, mandos, entradas registradas, integridad del injerto, decisiones de integración y presupuestos desde arrays y contrato. Incluye corrupción deliberada bajo Python−O. JO conserva contadores por cada evaluación y las últimas muestras; PN conserva primera/última/mínima/máxima salida por intervalo. No son cintas completas de todos los RHS. La igualdad de los targets ORN fue comprobada durante ejecución; sus contadores transitorios no se archivaron por separado.',
 'Hay estados científicos completos y fuentes preservadas. La reproducción CPU desde extracción limpia se verifica en la entrega; no se atribuye una reanudación GPU portable que no se haya ejecutado.',
 '', '![Comparación](COMPARACION.png)','',
 'Verificación local desde la raíz:','',
 '```bash','/home/daroch/miniconda3/envs/GPU/bin/python -B -O campanas/etapa45_transferencia_causal_20260928_55/verify55.py --corruptions','```',''])
 (H/'RESULTADOS.md').write_text('\n'.join(lines))
 decision=['#55 — decisión propia contrastada con los asesores','',
 'Conocíamos40–54 antes de diseñar55. No se presenta esta ronda como confirmación ciega. La propuesta propia A/B/C se registró antes de las nuevas respuestas externas; los contratos exactos se congelaron antes de sus respectivos ensayos.','',
 f'**C aporta evidencia causal local:** sustituir sólo el extraJO cambia el mando {c["replay_minus_online_yaw_deg_s"]:+.6f}°/s en I. La inicialización q0/S/gE0 coincide entre brazos. Motor reconstruyó la entrada desde el movimiento registrado. ChatGPT ASTRA_V2 coincide con el alcance condicionado al protocolo y advierte que el efecto no puede llamarse estabilizador o beneficioso sin criterio espacial. Su revisión de esta cifra fue conceptual; la revisión de archivos fue de Motor.','',
 f'**A, transporte recíproco material:** {a["reciprocal_material_transport"]}; {a["classification"]} para promoción desde esta criba parcial. La discontinuidad del primer intento era del instrumento, no una respuesta neuronal negativa. Quedó reparada mediante un evento fuente→receptor con retardo conservado. No se toca la memoria anterior para forzar continuidad.','',
 measured_note,'', 'ChatGPT_Motor_V2 prefería intervenir primero la señal terminal pura; Motor propuso el estado q+filtro común686ALPN reutilizando la cirugíaCPU51. Elegí esta frontera acotada, con pruebas vivas y estados realmente alcanzados. La alternativa de imponer una señal terminal sostenida no queda evaluada por un injerto inicial de estado. No se fusionaron propuestas por votación.','',
 '| Ruta | Qué decide55 | Próximo discriminador y falsador |','|---|---|---|',
 '| A · Transferencia PN | Intervención parcial cualificada, efecto y persistencia medidos en ambos sentidos. | Distinguir estado inicial de señal terminal sostenida y localizar cuál altera destinos identificados. Si la entrada consumida cambia materialmente y los destinos no responden en ambos sentidos bajo controles, esa frontera no es suficiente en el alcance. No inferir ausencia de todas las rutasPN. |',
 '| B · Cinética y estado local | La depresión ORN→PN ya existe; falta contrastar conversión e historia con observables apropiados. | Protocolo temporal local, primera respuesta y basal emparejados, intervalos reservados; descartar cambios que sólo incrementan amplitud o exigen ajustar cada condición. No introducir hambre global. |',
 '| C · Feedback corporal | JO altera causalmente el mando basal de I. | Separar efecto de cantidad/distribución y comprobar si hay una corrección útil en el preparado candidato. Una influencia sensorial sin mejora atribuible de rumbo no supera4; no extender I sólo porque se mueve. |','',
 'Mi prioridad siguiente es B y una delimitación de A guiada por los efectos realmente observados. C se conserva como control necesario de cualquier candidato acoplado al cuerpo. Ninguna de estas decisiones requiere integrar seis patas ni ajustar neuronas una a una. El predictor y el escáner ayudan a elegir cortes; la identificación causal requiere intervenir y contrastar observables.','',
 'Para la etapa4 todavía falta orientación atribuible a una fuente con ambos espejos y controles de feedback no degenerados, manteniendo el lector declarado. Para5 hace falta además una perturbación física y recuperación causal;55 no aplicó viento. No se adoptan como puertas retrospectivas los umbrales sugeridos por Gemini.','',
 'Motivo de parada: ronda acotada terminada, con dos instrumentos completos y resultados comparables; no continuar ajustando hasta obtenerPASS. La siguiente ronda necesita un contrato prospectivo propio, sin ampliar55. Se mantienen orientación con cuerpo actual, interfaz CNS→VNC acotada después, y músculos/seis patas posteriormente.','']
 (H/'DECISION.md').write_text('\n'.join(decision));(H/'REPORT_COST.json').write_text(json.dumps(dict(CPU_s=time.process_time()-start),indent=2)+'\n')
 print(json.dumps(dict(report=str(H/'RESULTADOS.md'),CPU_s=time.process_time()-start)))
if __name__=='__main__':main()
