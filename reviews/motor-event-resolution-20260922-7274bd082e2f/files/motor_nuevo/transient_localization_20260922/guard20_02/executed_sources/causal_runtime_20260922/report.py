"""Recompute comparison decisions from saved state arrays and frozen limits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1]/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(P))
from verify_transport import verify,read_state,flatten

def exact(a,b):
 a,b=dict(flatten(a)),dict(flatten(b));changes=[]
 for key in sorted(set(a)|set(b)):
  if key not in a or key not in b:changes.append(key);continue
  x,y=a[key],b[key]
  same=(isinstance(y,np.ndarray) and x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()) if isinstance(x,np.ndarray) else type(x)is type(y) and x==y
  if not same:changes.append(key)
 return changes

comparisons={
 'five_ms_aligned':verify(H.parent/'native_hybrid_20260922/aligned_01',H/'device_aligned_01'),
 'twenty_ms_aligned':verify(H/'reference20_01',H/'device20_01')}
fine=H/'references/fine5'
if not fine.exists():fine=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/runs/motor14_20260922/event_reference_fine')
comparisons['five_ms_fine']=verify(fine,H/'device_aligned_01')
recovery={k:exact(read_state(H/'recovery_01/control_neural'),read_state(H/f'recovery_01/{k}_neural')) for k in ('cold','fault')}
rows={}
for name in ('device_aligned_01','reference20_01','device20_01'):
 r=json.loads((H/name/'RESULT.json').read_text());t=r['times'];steady=t[1:]
 rows[name]={'status':r['status'],'completed_ms':len(t),'process_s':r['wall_total_s'],'steps_s':sum(v['whole_step_s'] for v in t),'steady_mean_s_per_ms':float(np.mean([v['whole_step_s'] for v in steady])),'steady_components_s_per_ms':{k:float(np.mean([v[k] for v in steady])) for k in ('spatial_membranes_and_events_s','global_CNS_s','PN_and_source_s','body_advance_s')},'native_membrane_s':r['cell']['native_wall_s']}
ref,candidate=rows['reference20_01'],rows['device20_01']
total_speed=ref['steady_mean_s_per_ms']/candidate['steady_mean_s_per_ms'];membrane_speed=ref['native_membrane_s']/candidate['native_membrane_s']
result={'comparisons':comparisons,'recovery_changed_paths':recovery,'runs':rows,'same_uncompressed_operator_twenty_ms_speedup':total_speed,'native_membrane_speedup':membrane_speed,'criteria_sha256':hashlib.sha256((P/'verification_vendor/INTEGRATED_ADAPTIVE_CONTRACT.json').read_bytes()).hexdigest(),'full_second_measured':False,'classification':'PROMETEDOR_NO_CONFIRMADO','stage3_admitted':False}
(H/'VERIFIED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
lines=['# Resultado: controlador por bloques en GPU','','**El motor completo aún no alcanza 1 segundo simulado por minuto. Etapa 3 sigue abierta.**','','Se implementó un controlador adaptativo genérico dentro de CUDA, con propuesta, rechazo y confirmación privados por bloque. El adaptador real conserva las 51 matrices originales, todos los voltajes/compuertas, fronteras de intercambio, tolerancias y regla de eventos. No depende de la cota de agrupación anterior.','','| Organismo real | Tiempo simulado | Avance acumulado | Proceso con carga/guardado |','|---|---:|---:|---:|']
for name in ('reference20_01','device20_01'):
 r=rows[name];lines.append(f"| {name} | {r['completed_ms']} ms | {r['steps_s']:.3f} s | {r['process_s']:.3f} s |")
lines.extend(['',f"La media de pasos 2–20 mejora **{total_speed:.3f}×**; el trabajo nativo de membranas mejora **{membrane_speed:.3f}×**, de {ref['native_membrane_s']:.3f} a {candidate['native_membrane_s']:.3f} s acumulados. Son mediciones de 20 ms del organismo, no un segundo ejecutado. No se repitieron para escoger el mejor tiempo.",'','## Fiabilidad medida','','La comparación de 20 ms usa la referencia con fronteras de eventos y operador original sin compresión. Las decisiones se reconstruyen desde arrays con el verificador y contrato congelados, no desde el booleano del runner.'])
m=comparisons['twenty_ms_aligned']['metrics']
lines += [f"PN: {m['PN_voltage_max_abs_mV']:.6g} mV; estado normalizado: {m['normalized_state_max_abs']:.6g}; compuertas: {m['gates_max_abs']:.6g}; guiñada: {m['yaw_trace_max_abs_deg']:.6g} grados. Conteos, relojes, metadatos y banderas exactos: {not comparisons['twenty_ms_aligned']['changed_exact_fields']}. Criba: {comparisons['twenty_ms_aligned']['screen_pass']}.",'','También se comparó la ventana de 5 ms con la referencia fina conservada. Esa referencia comparte algunos mecanismos heredados: la concordancia no sustituye una prueba independiente de toda la neurofisiología.','','La discrepancia al reconstruir se localizó: el cargador histórico perdía los valores efectivos de tau/theta preparados. El registro explícito del operador repone valores, sin recalcular intervenciones. Los estados guardados de continuación fría y después del fallo tienen, respectivamente, '+str(len(recovery['cold']))+' y '+str(len(recovery['fault']))+' diferencias respecto del control de 1 ms. La sesión corporal fallida continúa bloqueada; no se implementó recuperación automática del cuerpo.','','Un modelo adicional de tres estados usa el mismo controlador CUDA. La prueba de células reales verifica independencia de orden/lote, no publicación tras un fallo y reintento exacto. Esto acredita infraestructura reutilizable en esos casos, no soporte universal de cualquier cerebro.','','## Rival de puertos y límite pendiente','','Se implementó por separado la respuesta de bloques afines a saltos y cambios de operador ordenados. Pasa referencias de exponencial matricial, eventos tardíos, constantes iguales/casi iguales, prefijos y conductancias no conmutativas. Admite un modelo de siete estados sin cambiar el núcleo. **No se conectó como sustituto del CNS no lineal:** falta un control del error de su recurrencia y receptores.','','ChatGPT aportó un falsador relevante: un pico entre muestras puede perderse aunque coincida el voltaje final. Los relojes locales no solucionan automáticamente ese problema del detector heredado; el caso se conserva en `DEVICE_CHECK.json`. No se amplió ninguna tolerancia para admitir la candidata.','','## Autocrítica y decisión','','Bajar de nivel no elimina un algoritmo que repite trabajo global ante eventos locales. En esta ejecución, CNS ocupa '+f"{candidate['steady_components_s_per_ms']['global_CNS_s']:.3f} de {candidate['steady_mean_s_per_ms']:.3f} s por ms"+'; PN aporta '+f"{candidate['steady_components_s_per_ms']['PN_and_source_s']:.3f}"+' s/ms. Otra aceleración aislada de membranas tiene un techo bajo. La siguiente ronda debe cambiar la planificación de dependencias y los puertos con error controlado; el Schur acoplado permanece como rival matemático, no como una tercera implementación oculta.','','Se conserva A como **PROMETEDOR_NO_CONFIRMADO**: mejora real y criba de 20 ms, sin promesa de segundos ni admisión de etapa 3. B es un componente probado para sistemas afines, sin ganancia global atribuida. C no se implementó. Se cierra este hito de dos prototipos y reparación de identidad para revisión; el objetivo de eficiencia general sigue pendiente.','','ChatGPT revisó fuentes anteriores y su comentario cambió la reparación/pruebas; se le envían fuentes y arrays nuevos. No se atribuye reproducción CUDA a esa revisión. Jev hizo una llamada de clasificación acotada (1064/214 tokens), sin decidir tolerancias ni veredictos. No se usaron subagentes Codex.','','[Contrato y arquitectura](ARCHITECTURE.md) · [Presupuesto previo](PLAN.json) · [Verificación desde arrays](VERIFIED.json).']
(H/'RESULTADOS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'speedup':total_speed,'native_membrane_speedup':membrane_speed,'recovery':recovery,'screens':{k:v['screen_pass'] for k,v in comparisons.items()}}))
