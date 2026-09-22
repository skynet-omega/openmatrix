# Ronda nativa del organismo — 22-09-2026

Estado: **PROMETEDOR_NO_CONFIRMADO**. La meta de velocidad y la etapa 3 siguen abiertas. Leer `RESULTADOS.md` y `VERIFIED.json` antes de ejecutar. No hay procesos persistentes de esta ronda.

## Reproducir verificaciones desde una extracción limpia

El ZIP de revisión contiene fuentes, operadores/células reales y estados finales. Es autocontenido para las verificaciones siguientes, con un entorno Python que disponga de NumPy, SciPy y CuPy, compilador C++ y CUDA. No requiere importar fuentes del árbol activo.

```bash
# Situarse en la raíz extraída, que contiene motor_nuevo y campanas.
export PYTHONUTF8=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Sustituir si el entorno GPU tiene otro nombre/ruta.
PYTHON=/home/daroch/miniconda3/envs/GPU/bin/python
H=motor_nuevo/native_hybrid_20260922
P=campanas/etapa3_motor_nuevo_20260922
g++ -O3 -std=c++17 -shared -fPIC "$H/cell_control.cpp" -lcudart -o "$H/libcell_control.so"
g++ -O3 -std=c++17 -shared -fPIC "$H/graph_control_v2.cpp" -lcudart -o "$H/libgraph_control_v2.so"
g++ -O3 -std=c++17 -shared -fPIC "$P/graph_control.cpp" -lcudart -o "$P/libgraph_control.so"
"$PYTHON" -B -O "$H/report.py"
"$PYTHON" -B -O "$H/test_structure.py"
"$PYTHON" -B "$H/check_event_boundary.py"
"$PYTHON" -B "$H/check_transaction.py"
# Comprobaciones adicionales, más costosas:
"$PYTHON" -B "$H/check_basis.py" --certificate
"$PYTHON" -B "$H/check_ros.py"
```

Los verificadores reconstruyen resultados desde los arrays, no desde booleanos del runner. Los directorios originales no se sobrescriben. Estas comprobaciones no equivalen a repetir la simulación del organismo.

## Repetir el organismo completo en el entorno local original

`benchmark_real.py` conserva la anatomía, el checkpoint, cuerpo, campos y ajustes del cargador histórico. **Este paquete de revisión no distribuye todos esos recursos estáticos ni permite repetir el organismo aislado del árbol histórico.** Esa distribución completa sigue pendiente. Su identidad queda en `PARENT.json`, las fuentes congeladas y los manifiestos de estado.

```bash
# En /home/daroch/AXIOMA_ASTRA, con los recursos históricos originales presentes:
PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=14 \
NUMBA_CACHE_DIR=/home/daroch/AXIOMA_ASTRA/campanas/etapa3_motor_nuevo_20260922/numba_cache \
timeout 240 /home/daroch/miniconda3/envs/GPU/bin/python -B \
 motor_nuevo/native_hybrid_20260922/benchmark_real.py \
 --out motor_nuevo/native_hybrid_20260922/nueva_repeticion_unica \
 --ms 5 --native-cell --compressed --event-boundaries
```

`--ros` ejecuta la alternativa B descartada, para reproducción. Sin `--event-boundaries` se reproduce el muestreo temporal heredado, que comparte el defecto del evento tardío: sirve de control, no de modo preciso general. No activar reanudación automática tras fallos; el guardián invalida propietarios obsoletos y la sesión corporal falla de forma cerrada.

## Evidencia y alcance

- `baseline_01`, `native_01`, `compressed_01`, `certified_02`, `aligned_01`: estados y tiempos del organismo real, 5 ms cada uno.
- `ros_01`: negativo de rendimiento/criterio; `ROS_OPERATOR_CHECK.json`: contraste independiente CPU Radau y tableau CPU/GPU.
- `certified_01`: fallo por cota uniforme demasiado conservadora; se sustituyó su cálculo por una cota dependiente de los coeficientes, manteniendo el residuo permitido en 1e-12.
- `recovery_01..04`: fallos de reconstrucción y recibos; la última conserva arrays control/reintento y diferencias materiales, no un PASS de recuperación.
- `CHATGPT_REVIEW_TRANSPORT.md`, `CHATGPT_ARCHITECTURE.md`: propuestas/revisión externa con su alcance declarado.
- `vendor`: fuentes donantes mínimas para comprobaciones portables, con procedencia en `SOURCES.json`.

Se conservan los contratos y criterios previos. Dos prototipos numéricos completos como máximo. No se ajustó fisiología ni se admite etapa 3. La certificación de un motor genérico para otros cerebros sigue pendiente.
