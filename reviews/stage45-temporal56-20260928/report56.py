"""Render the report and scientific figure from recomputed results only."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def main():
 start=time.process_time();a=json.loads((H/'A_RESULTADOS.json').read_text());b=json.loads((H/'B_RESULTADOS.json').read_text());v=json.loads((H/'A_VERIFY_COST.json').read_text())
 need(not (H/'RESULTADOS.md').exists(),'immutable report')
 fig,ax=plt.subplots(2,2,figsize=(13.5,8.2),layout='constrained');colors={'sham':'#24599c','profile':'#d67422'};labels={'sham':'Historia control','profile':'Historia perfil'}
 for receiver in colors:
  for mode in ['pulse','hold']:
   values=[]
   for donor in ['sham','profile']:
    with np.load(H/'repair01'/(mode+'_'+receiver+'_from_'+donor)/'traces.npz') as z:values.append(np.rad2deg(z['command_yaw_rate_rad_s']))
   ax[0,0].plot(np.arange(1,129),values[1]-values[0],color=colors[receiver],ls='--' if mode=='pulse' else '-',label=labels[receiver]+(' · pulso 1 ms' if mode=='pulse' else ' · sostenida 128 ms'))
 ax[0,0].axvspan(65,128,color='#e6e6e6',alpha=.6,zorder=-1);ax[0,0].axhline(0,color='gray',lw=.7);ax[0,0].set(xlabel='Tiempo desde intervención (ms)',ylabel='Cambio del mando de giro (°/s)',title='A · Identidad terminal: perfil menos control');ax[0,0].legend(fontsize=8)
 scientific=[n for n in a['means_primary_window']];margins=np.array([a['arms'][n]['DNg_margin_max'] for n in scientific]);x=np.arange(8)
 ax[0,1].bar(x-.18,margins[:,0],width=.36,label='DNg100 izquierda',color='#5869a7');ax[0,1].bar(x+.18,margins[:,1],width=.36,label='DNg100 derecha',color='#94b5c7');ax[0,1].axhline(0,color='black',lw=.8);ax[0,1].set_xticks(x,[n.replace('pulse_','P ').replace('hold_','S ').replace('_from_','←').replace('sham','C').replace('profile','O') for n in scientific],rotation=35,ha='right');ax[0,1].set(ylabel='Máximo de net + drive − umbral\n(unidades internas)',title='A · Margen descendente en todos los RHS');ax[0,1].legend(fontsize=8)
 with np.load(H/'reference/TEMPORAL48.npz') as z:
  for arm,col in [('sham','#555555'),('dm1','#8f3b6f'),('profile','#d67422'),('permuted','#2a958b')]:
   f=z[arm+'__filters'];y=((.186*f[:,2]+.144*f[:,3])*z['caps']/.33).mean(axis=1);ax[1,0].plot(z[arm+'__step']/1000,y,label=arm,color=col)
 ax[1,0].axhline(b['constant_rate_asymptote_equivalent_Hz'],color='gray',ls=':',label='Asíntota estacionaria');ax[1,0].set(xlabel='Tiempo de la campaña 48 (s)',ylabel='Entrada genérica equivalente (Hz del modelo)',title='B · La historia permite exceder la asíntota');ax[1,0].legend(fontsize=8)
 with np.load(H/'B_PROTOCOLS.npz') as z:
  for mode,style in [('dynamic','-'),('frozen','--')]:
   y=z['steady_baseline_50Hz_'+mode][:,1].sum(axis=1);ax[1,1].plot(np.arange(1,len(y)+1)/1000,y,ls=style,label='Recursos '+('dinámicos' if mode=='dynamic' else 'congelados al inicio'))
 ax[1,1].axvspan(0,.5,color='#e6e6e6',zorder=-1);ax[1,1].set(xlabel='Tiempo desde paso 7→50 Hz (s)',ylabel='Conductancia formal del modelo (nS)',title='B · Control matemático, sin ajuste biológico');ax[1,1].legend(fontsize=8)
 fig.suptitle('Campaña 56 · Señal sostenida, historia sináptica y salida motora',fontsize=15);fig.savefig(H/'COMPARACION.png',dpi=160);fig.savefig(H/'COMPARACION.pdf');plt.close(fig)
 e=a['effects_by_receiver'];support=a['joint_persistence_support'];forward=any(v['held_forward_effect_material'] for v in e.values());allzero=all(v['DNg_target_max_all_RHS']==0 for v in a['arms'].values())
 text=['# Campaña 56 — transferencia temporal y mando','',
 '**Etapas 4 y 5 abiertas.** Esta ronda es un diagnóstico causal con dos instrumentos, no una prueba de navegación hacia una fuente ni de recuperación tras viento. La señal sensorial no se adapta a una fuente espacial en estos brazos.', '',
 '## Resultado de la intervención sostenida', '',
 ('La persistencia supera el criterio diagnóstico en ambas historias.' if support else 'La persistencia no supera el criterio diagnóstico conjunto en ambas historias.')+' Clasificación del contraste A: **'+a['classification']+'**, restringida a la suficiencia de esta intervención y esta preparación.', '',
 'Sostener 128 ms la misma muestra también aumenta la exposición acumulada frente al pulso de 1 ms. Este diseño comprueba el efecto de duración, pero no separa una integración ordinaria de una memoria temporal especializada. En estos brazos la entrada ORN está en basal: no demuestra que una salida PN natural bajo olor mantenido sea demasiado breve. El signo es un cambio respecto de control; no indica que el giro apunte hacia una fuente.', '',
 'Dos historias de receptor × dos identidades terminales reales × dos duraciones: ocho brazos de 128 ms, más dos cualificaciones de 2 ms. El pulso actúa sólo el primer ms y la intervención sostenida actúa los 128 ms. Ambos usan el mismo escritor sobre 686 salidas PN consumidas por la suma genérica. Las rutas especializadas quedan con su señal nativa. La fuente fijada es una muestra inicial real, no una reproducción completa de su trayectoria natural. No se cambian pesos, umbrales ni lectores.', '',
 'Ventana primaria congelada: 65–128 ms. Cada efecto es fuente perfil menos fuente control, manteniendo la historia receptora. La interacción es efecto sostenido menos efecto pulso. El target DNg por ms usa la cuarta RHS del último ensayo RK3(2) aceptado en un intervalo comprometido, evaluada en el límite izquierdo de la frontera; no se reevalúa después de ella.', '',
 '| Historia receptora | Δ giro pulso (°/s) | Δ giro sostenido (°/s) | Interacción (°/s) | Cociente sostenido/pulso | Criterio conjunto DN/giro |',
 '|---|---:|---:|---:|---:|---|']
 for r in ['sham','profile']:
  x=e[r];pulse=x['terminal_identity_effect']['pulse'];held=x['terminal_identity_effect']['hold'];ratio=x['checks']['yaw_deg_s']['hold_to_pulse_ratio'];text.append(f"| {labels[r]} | {pulse['yaw_deg_s']:.8g} | {held['yaw_deg_s']:.8g} | {x['duration_by_identity_interaction']['yaw_deg_s']:.8g} | {'rama cero' if ratio is None else f'{ratio:.6g}'} | {'Sí' if x['persistence_supported_at_this_frontier'] else 'No'} |")
 text+=['','El criterio exige, por historia, interacción material tanto en DNb (1,6×10⁻⁵ q) como en giro (0,02°/s), mismo signo y al menos duplicación. Los valores y cada condición lógica se reconstruyen en `A_RESULTADOS.json`; no se seleccionan sólo las historias favorables.', '',
 ('**DNg100 conservó objetivo cero en todas las evaluaciones registradas.**' if allzero else '**DNg100 mostró algún objetivo positivo; revisar su distribución y control emparejado.**')+' '+('Hubo un efecto material sobre avance bajo intervención.' if forward else 'Ninguna historia cumplió el criterio conjunto de cambio de avance y objetivo DNg100.'),'',
 '| Brazo | Objetivo DNg100 máximo | Avance medio 65–128 ms (mm/s) | Margen máximo izquierda / derecha |', '|---|---:|---:|---|']
 for n in scientific:
  x=a['arms'][n];m=a['means_primary_window'][n];text.append(f"| {n} | {x['DNg_target_max_all_RHS']:.7g} | {m['forward_mm_s']:.7g} | {x['DNg_margin_max'][0]:.7g} / {x['DNg_margin_max'][1]:.7g} |")
 sz=a['fixed_input_contrast'];text+=['',f"Las fuentes difieren en {sz['changed_terminals']}/{sz['total_terminals']} terminales. El contraste L2 es {100*sz['relative_to_sham_L2']:.6g}% respecto de control y {100*sz['relative_to_profile_L2']:.6g}% respecto de perfil. Se informa aunque sea pequeño; 56 no incorpora la antigua puerta del 1%, y no cambia retrospectivamente el veredicto de 55.", '',
 '## Transferencia sensorial e historia', '',
 'B reutiliza 3 s por condición de 48 y resuelve las ecuaciones existentes con una solución analítica. Los 16 contrastes contra DOP853 dieron error máximo '+f"{b['numerical_check']['max_absolute_error']:.8g}"+' (tolerancia previa 10⁻⁸). El control congela recursos desde el inicio, con basal y estado inicial emparejados; no iguala el pico posterior. No se ajustaron parámetros a estos resultados.', '',
 '| Registro 48, últimos 500 ms | ORN q·cap medio (Hz del modelo) | Puente genérico equivalente (Hz del modelo) | Recurso rápido | Recurso lento |', '|---|---:|---:|---:|---:|']
 for n in ['sham','dm1','profile','permuted']:
  x=b['recorded'][n]['late'];text.append(f"| {n} | {x['source_proxy_Hz_mean']:.6g} | {x['bridge_equivalent_Hz_mean']:.6g} | {x['A_fast_mean']:.6g} | {x['A_slow_mean']:.6g} |")
 text+=['',f"La asíntota a tasa constante es {b['constant_rate_asymptote_equivalent_Hz']:.8g} Hz equivalentes. **No es una cota transitoria ni una tasa de PN.** La entrada DM1 registrada puede superarla porque conserva historia. Esto demuestra compresión en las ecuaciones actuales, sin demostrar exceso de depresión fisiológica.", '',
 'La revisión de Motor no encontró una segunda normalización de población en el tramo inspeccionado. También confirmó que las salidas finas y genéricas tienen consumidores diferentes y que el estímulo prescrito sustituye explícitamente el objetivo periférico ORN. Los datos biológicos temporalmente emparejados para validar esa transferencia siguen pendientes. Véanse `aporte_motor/FRONTERA.md` y `FUENTES_Y_DECISIONES.md`.', '',
 '## Integridad y coste', '',
 f"{a['budget']['attempted_CNS_ms_measured']} ms CNS intentados y comprometidos medidos en la cola reparada; cargo conservador total {a['budget']['CNS_ms_budget_charge']} ms. CPU de trabajadores medida: {a['budget']['worker_CPU_s_measured_repaired']:.3f} s; cargo con el primer fallo: {a['budget']['worker_CPU_s_budget_charge']:.3f} s. Colas: {a['budget']['queue_wall_s']:.3f} s. Topes previos: 1200 ms CNS, 6000 s CPU y 5000 s de cola.", '',
 'La primera cualificación falló al escribir caracteres Unicode antes del primer paso CNS. No es un negativo neural. Se conservaron log y archivos parciales; se cobró la reserva completa de 170 s CPU y 2 ms CNS porque no había cierre cronometrado. La reparación sólo impone UTF-8 en metadatos. Dos nuevas cualificaciones reprodujeron exactamente la referencia antes de los ocho brazos.', '',
 f"El verificador reconstruye lectores, relojes, contexto RHS, identidad de terminales, intervención consumida, márgenes, objetivos y presupuesto. Rechazó {len(v['tests'])} corrupciones deliberadas bajo Python optimizado. CPU de verificación: {v['CPU_s']:.3f} s. El banco B consumió "+f"{json.loads((H/'B_COST.json').read_text())['CPU_s']:.3f} s CPU; no repitió 48.", '',
 'No hay semillas nuevas ni confirmación ciega: son dos historias del mismo preparado expuesto. Las cualificaciones comprueban transparencia instrumental, no equivalencia biológica. Los archivos completos conservan estados y fuentes; la reproducción limpia del paquete compacto recalcula los registros y el banco CPU, no vuelve a simular el cerebro. El cierre y la publicación tienen recibos separados.', '',
 '![Comparación](COMPARACION.png)', '', '## Decisión y siguiente discriminador', '',
 ('Conservar A como señal causal prometedora para comprobar persistencia natural y alcance de consumidores, sin promoverla a navegación.' if support else 'Descartar el aumento de duración de esta muestra terminal parcial como solución suficiente bajo el criterio de 56. Conservar sus efectos medidos; no concluir que toda salida PN carece de efecto.'), '',
 'Mantener tres alternativas diferenciadas: **A**, ruta y persistencia natural de salida PN; **B**, transferencia ORN→PN con datos y observables emparejados; **C**, estado descendente e iniciación locomotora. Priorizar el discriminador que cambie una decisión de mecanismo, antes de otra vida larga o de integrar patas. No ajustar neuronas una a una, cambiar el lector por selección de un positivo ni aumentar ganancias para forzar un PASS. El siguiente contrato deberá delimitar dos instrumentos como máximo y un presupuesto nuevo antes de ejecutar.', '',
 'La admisión de 4 requiere orientación útil con controles causales pertinentes y la de 5 requiere perturbación y recuperación. Una activación externa de terminales no sustituye esos resultados. La posibilidad de lograrlos sigue siendo una hipótesis experimental; este diagnóstico no prueba imposibilidad ni garantiza éxito.', '', 'La priorización posterior y el contraste con los asesores quedan en [DECISION.md](DECISION.md).', '']
 (H/'RESULTADOS.md').write_text('\n'.join(text),encoding='utf-8');(H/'REPORT_COST.json').write_text(json.dumps({'CPU_s':time.process_time()-start,'result_sha256':hashlib.sha256((H/'A_RESULTADOS.json').read_bytes()).hexdigest()},indent=2)+'\n')
 print(json.dumps({'classification':a['classification'],'joint_persistence_support':support,'forward_effect_material':forward,'all_DNg_targets_zero':allzero,'CPU_s':time.process_time()-start}))
if __name__=='__main__':main()
