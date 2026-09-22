# Ejecutores causales por bloques, ronda 22-09-2026

Entrada: `RESULTADOS.md`, `ARCHITECTURE.md`, `PLAN.json`. Candidata experimental;
no activar como motor final ni inferir admisión de etapa3. No hay ensayos en cola.

Requisitos comprobados: Python3.10, NumPy1.26.4, SciPy1.15.3, CuPy13.6,
CUDA12 en RTX4070TiSUPER16GiB. FP64, sin FMA en estos kernels.

## Verificación corta del paquete extraído

Desde la raíz extraída, con esas dependencias y GPU CUDA:

```bash
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/daroch/miniconda3/envs/GPU/bin/python
ROUND=motor_nuevo/causal_runtime_20260922
$PY -B "$ROUND/check_scheduler.py"
$PY -B "$ROUND/check_affine.py"
$PY -B "$ROUND/check_device.py"
$PY -O -B "$ROUND/check_operator_state.py"
$PY -O -B "$ROUND/report.py"
```

Sustituir `PY` por el intérprete con las dependencias si se extrae en otra máquina.
El ejecutor CUDA se compila desde las fuentes incluidas. La reconstrucción del
informe vuelve a comparar los arrays guardados; no simula nuevamente el organismo.
Los comprobadores sobrescriben sus recibos derivados, nunca los datos base.

## Repetición del organismo en este entorno local

El benchmark completo requiere el checkpoint `settled_700ms`, anatomía, modelos
MuJoCo, bibliotecas y cargadores históricos de AXIOMA_FLYWIRE. Estos recursos
estáticos **no se incluyen íntegramente en este paquete de revisión**. Por tanto,
la extracción limpia demuestra las pruebas de componentes y la reconstrucción
de resultados, no reproducción independiente completa del organismo.

Con el árbol histórico local disponible en sólo lectura, desde AXIOMA_ASTRA:

```bash
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=14
export NUMBA_CACHE_DIR=/home/daroch/AXIOMA_ASTRA/campanas/etapa3_motor_nuevo_20260922/numba_cache
PY=/home/daroch/miniconda3/envs/GPU/bin/python
ROUND=motor_nuevo/causal_runtime_20260922
timeout 240 "$PY" -B "$ROUND/benchmark_real.py" --native-cell --event-boundaries --ms 20 --out "$ROUND/repeat_reference20"
timeout 240 "$PY" -B "$ROUND/benchmark_real.py" --device-cell --event-boundaries --ms 20 --out "$ROUND/repeat_device20"
"$PY" -B campanas/etapa3_motor_nuevo_20260922/verify_transport.py "$ROUND/repeat_reference20" "$ROUND/repeat_device20" --out "$ROUND/repeat_check.json"
timeout 240 "$PY" -B "$ROUND/check_operator_recovery.py" recovery_new
```

Los directorios de ejecución deben ser nuevos. Una repetición no es autorización
para ajustar criterios según el resultado. La guardia conserva el bloqueo después
de un fallo de la sesión corporal. Los nombres `repeat_*` son ejemplos no ejecutados.

El historial de la ronda previa está fijado en OpenMatrix, commit
`8ff835ecae45f39e41c2969b84bacec95effc061`; se incluyen las fuentes transitivas
necesarias para la revisión del adaptador y los comprobadores cortos. Las fuentes
congeladas dentro de cada ejecución documentan las versiones que la produjeron.
