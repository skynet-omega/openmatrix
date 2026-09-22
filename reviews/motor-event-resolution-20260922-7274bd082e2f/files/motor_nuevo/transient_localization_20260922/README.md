# Resolución de eventos y estado del motor — 22-09-2026

**PROMETEDOR_NO_CONFIRMADO. Motor no listo; etapa 3 continúa abierta.** La autorización del usuario para continuar al estar listo permanece vigente. No se lanzó una campaña conductual con evidencia numérica insuficiente.

## Resultado medido y decisión

| Comparación real, mismo sham | Diferencia máxima normalizada |
| --- | ---: |
| Padre independiente frente a refinamiento de 1562 ns, 1 ms | 0.000224397993703 |
| Guardia temporal frente al mismo refinamiento, 1 ms | 2.18344053826e-06 |
| Guardia frente a refinamiento, 5 ms | 1.74167220583e-05 |
| Guardia frente a referencia antigua, 20 ms | 0.000136644939714 |

El límite histórico es 1e-4. La mejora de 1 ms es un factor 102.77 en esa métrica; en 1/5 ms no cambian campos discretos ni aparecen no finitos. Las instantáneas son cerebrales, no validación de toda la trayectoria ni todos los campos corporales. La referencia antigua también se separa del refinamiento y no es verdad absoluta. A 20 ms la comparación antigua excede el límite; además carece de dos trazas centrales añadidas después, declarado por el verificador.

La guardia completó 20 ms en73.108s de avance y89.504s totales. Observación:0.020793s. Media:3.655s por ms simulado, 60.92 veces el presupuesto temporal objetivo. No se ejecutó 1 segundo y no se promete una extrapolación exacta. El contraste refinado completó14ms y falló durante el siguiente paso con `accepted event state outside domain`. No hay endpoint refinado de 20 ms ni resultado corporal final de ese brazo. Comparar directamente sus tiempos totales con 20 ms sería incorrecto.

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
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=14 timeout 240s python motor_nuevo/transient_localization_20260922/benchmark_candidate.py --variant guard --ms 20 --out NUEVA_SALIDA
```

El comando completo es documentación; no se ejecutó otra carga tras agotar la ronda. 
