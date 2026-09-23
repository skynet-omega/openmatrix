"""Figures and round decision from saved arrays comparisons and run receipts."""
from pathlib import Path
import json
T=Path(__file__).resolve().parent
r=json.loads((T/'CANDIDATES.json').read_text());p=json.loads((T/'PLAN.json').read_text());runs={}
for name in ('reference_01','causal_01','fine1562_01','guard20_01','guard20_02','fine1562_20'):
 v=json.loads((T/name/'RESULT.json').read_text());runs[name]={k:v.get(k) for k in ('status','ms','wall_s','error')}
wall=sum(v['wall_s'] for v in runs.values())
if len(runs)>p['budget']['organism_loads'] or wall>p['budget']['aggregate_process_seconds']:raise ValueError('Round budget exceeded')
budget={'organism_loads':len(runs),'aggregate_runner_wall_s':wall,'limits':p['budget'],'runs':runs,'numerical_candidates':2,'new_Jev_requests':1,'note':'Failed initialization counts as a load. No extra run after refined-domain failure. GPU/CPU fixture receipts in CLEAN_CHECK.log and local logs; no concurrent organism benchmarks.'}
(T/'BUDGET_USED.json').write_text(json.dumps(budget,indent=2)+'\n')
guard=r['performance']['guard20_02'];fine=r['performance']['fine1562_20'];old=json.loads((T/'OLD_VS_FINE1.json').read_text())['old_causal']['metrics']['normalized_state_max_abs'];one=r['brain_snapshots']['1'];five=r['brain_snapshots']['5'];twenty=r['twenty_ms_against_old_reference']['historical_partial_screen']
verdict={'status':'PROMETEDOR_NO_CONFIRMADO','engine_ready':False,'stage3_admission':False,'one_ms_normalized_difference':one['normalized_state_max_abs'],'five_ms_normalized_difference':five['normalized_state_max_abs'],'one_ms_error_reduction_factor':old/one['normalized_state_max_abs'],'refined_completed_ms':fine['ms'],'refined_failure':r['fine_failure']['message'],'guard_completed_ms':guard['ms'],'guard_advance_s':guard['advance_s'],'measured_speed_gap_to_target':guard['mean_advance_s_per_simulated_ms']/.06,'reasons':['Refined20ms endpoint unavailable due to domain failure.','Whole-organism throughput remains far from approximate60s per simulated second.','Endpoint comparisons do not certify event histories, full-state bounds or stage3 behavior.']}
(T/'VERDICT.json').write_text(json.dumps(verdict,indent=2)+'\n')
text=f'''# Resolución de eventos y estado del motor — 22-09-2026

**{verdict['status']}. Motor no listo; etapa 3 continúa abierta.** La autorización del usuario para continuar al estar listo permanece vigente. No se lanzó una campaña conductual con evidencia numérica insuficiente.

## Resultado medido y decisión

| Comparación real, mismo sham | Diferencia máxima normalizada |
| --- | ---: |
| Padre independiente frente a refinamiento de 1562 ns, 1 ms | {old:.12g} |
| Guardia temporal frente al mismo refinamiento, 1 ms | {one['normalized_state_max_abs']:.12g} |
| Guardia frente a refinamiento, 5 ms | {five['normalized_state_max_abs']:.12g} |
| Guardia frente a referencia antigua, 20 ms | {twenty['metrics']['normalized_state_max_abs']:.12g} |

El límite histórico es 1e-4. La mejora de 1 ms es un factor {verdict['one_ms_error_reduction_factor']:.2f} en esa métrica; en 1/5 ms no cambian campos discretos ni aparecen no finitos. Las instantáneas son cerebrales, no validación de toda la trayectoria ni todos los campos corporales. La referencia antigua también se separa del refinamiento y no es verdad absoluta. A 20 ms la comparación antigua excede el límite; además carece de dos trazas centrales añadidas después, declarado por el verificador.

La guardia completó 20 ms en{guard['advance_s']:.3f}s de avance y{guard['wall_s']:.3f}s totales. Observación:{guard['observation_s']:.6f}s. Media:{guard['mean_advance_s_per_simulated_ms']:.3f}s por ms simulado, {verdict['measured_speed_gap_to_target']:.2f} veces el presupuesto temporal objetivo. No se ejecutó 1 segundo y no se promete una extrapolación exacta. El contraste refinado completó{fine['ms']}ms y falló durante el siguiente paso con `{r['fine_failure']['message']}`. No hay endpoint refinado de 20 ms ni resultado corporal final de ese brazo. Comparar directamente sus tiempos totales con 20 ms sería incorrecto.

## Causa localizada y crítica

Estados iniciales y operador idénticos; instrumentar conserva exactamente los extremos anteriores. La primera diferencia de filtro aceptada se explica al cambiar únicamente las marcas temporales, con inputs y saltos iguales. El replay independiente del puerto 75907 usa SciPy expm 2×2 entre saltos y coincide con CUDA en 166 consultas, residual máximo3.47e-18. Esto comprueba el filtro para las historias registradas, no la exactitud de las emisiones ni toda mediación posterior.

Los conteos finales coincidían, pero no siempre los intermedios; q puede diferir aproximadamente 0.294 en una frontera. Habíamos exigido precisión de voltajes sin controlar suficientemente la fecha de los eventos. También sería un error tratar una referencia más lenta como solución exacta: el refinamiento ahora expuso una excepción no diagnosticada con suficiente detalle. Se añadió al ejecutor la captura del índice, valor y reloj del ensayo fallido antes de rollback. La nueva prueba induce una violación real y comprueba rechazo y restauración. No recupera retrospectivamente los valores del fallo real ya ocurrido, no limita/clampa el estado y no altera aceptación exitosa.

## Alternativas de esta ronda

A, diferencias temporales de eventos: respaldada localmente frente a B, filtro/publicación, y C, estado/operador distinto. La candidata numérica A limita a 1562 ns los ensayos cercanos al umbral existente, con detector muestreado intacto. Mejora acotada, conservada para investigar; no promovida.

La segunda candidata numérica, B, conserva la propuesta nominal tras un corte obligado del CNS. En fixture independiente reduce 10 a 9 pasos, sin saltar eventos; no se midió con el organismo. No se combinó con A para atribuirle un efecto no observado. El límite de 6 cargas incluye el fallo inicial de conexión reparado. No añadir cargas para rescatar un PASS.

Para la próxima decisión macro permanecen tres rutas: A, corregir el defecto de dominio con diagnóstico y el esquema general actual como control; B, integración multirritmo o respuesta analítica de puertos que evite recalcular la red global por cada evento local, conservando realimentación y estimador; C, comparar un backend general alternativo con esas mismas ecuaciones/inputs, sin sustituirlas silenciosamente por LIF. B/C requieren protocolo y falsador propios, no son motores implementados. Mantener propuestas rivales no implica construirlas todas a la vez.

## Revisión externa y límites

ChatGPT examinó fuentes/datos textuales y sostuvo la localización; pidió contrastar con refinamiento y comprobar estados ya por encima del umbral. `check_guard_threshold.py` prueba en CUDA ese falsador, picos de ensayo, axones y límite exacto. ChatGPT no ejecutó los arrays ni aprobó el motor. Jev realizó 1 consulta real, clasificó tareas y no es un verificador científico. No hay subagentes Codex.

## Reproducción

`PLAN.json`, `BUDGET_USED.json`, `CANDIDATES.json`, `LOCALIZATION.json` y `VERDICT.json` contienen criterio, consumo, arrays comparados y veredicto. Archivos originales y fallos conservados. Fuentes efectivamente ejecutadas están en cada `executed_sources`, anteriores a la reparación diagnóstica cuando corresponde.

Desde la raíz del ZIP extraído, con Python 3.11, NumPy 1.26, SciPy 1.15 y CuPy 13.6/CUDA 12:

```bash
python -O motor_nuevo/transient_localization_20260922/analyze_boundaries.py
python -O motor_nuevo/transient_localization_20260922/compare_candidates.py
python -O motor_nuevo/transient_localization_20260922/reconstruct_membranes.py
python motor_nuevo/transient_localization_20260922/check_port_replay.py
python motor_nuevo/transient_localization_20260922/check_guard_threshold.py
python motor_nuevo/transient_localization_20260922/check_guard_attach.py
```

La entrega es autocontenida para reconstrucción offline y fixtures documentados. Incluye matrices locales de esos fixtures, no todos los recursos anatómicos/corporales estáticos del organismo. El modo completo local requiere el histórico conservado y su entorno GPU; no se anuncia reproducción completa desde el ZIP:

```bash
# Desde AXIOMA_ASTRA con el histórico original accesible; salida nueva obligatoria.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=14 timeout240s python motor_nuevo/transient_localization_20260922/benchmark_candidate.py --variant guard --ms20 --out NUEVA_SALIDA
```

El comando completo es documentación; no se ejecutó otra carga tras agotar la ronda. Corregir espacios en opciones sería necesario si se copiara la forma abreviada; los comandos exactos están en el runner.
'''
# Render the actual command, not prose abbreviations.
text=text.replace('timeout240s','timeout 240s').replace('--ms20','--ms 20').replace('Corregir espacios en opciones sería necesario si se copiara la forma abreviada; los comandos exactos están en el runner.','')
(T/'README.md').write_text(text)
print(json.dumps(verdict,indent=2))
