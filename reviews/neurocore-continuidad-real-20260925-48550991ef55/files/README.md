# Núcleo CUDA de continuidad

Una implementación: controlador de graphs residente y punto medio exponencial
adaptativo, FP64. Los callbacks del modelo aportan ecuaciones y eventos. El
núcleo no importa anatomía, sensores, objetivos conductuales ni ganancias.

## Cambio matemático exacto

Una evaluación del modelo entrega `target, rate` para
`y' = rate(y,t)*(target(y,t)-y)`. Un paso de punto medio evalúa coeficientes
al comienzo, obtiene el estado medio con flujo exponencial congelado y evalúa
otra vez en ese estado medio. La referencia compara un paso entero con dos
medios: 2+2+2 evaluaciones. El paso entero y el primer medio comparten exactamente
estado, tiempo, lado e inputs iniciales: aquí se calcula esa evaluación una vez,
2+2+2-1=5. Sus dos arrays se copian antes de que el modelo reutilice sus buffers.
El error sigue siendo |full-fine| / (3*(atol+rtol*max(|full|,|fine|))).

Se conserva orden, fórmulas, paso doble, límites y política de aceptación del
método de referencia. Se fusiona sólo la aritmética de flujo por coordenada,
con FMA desactivado. Esto no garantiza identidad bit a bit con todos los
compiladores ni una aceleración: la pareja real decide esos alcances.

## Contrato reutilizable

- `GraphMidpoint(initial, coefficient, rtol=..., atol=...)`.
- `coefficient(y, time, side)` devuelve dos arrays CuPy FP64 contiguos del
  tamaño del estado. Puede reutilizar sus buffers de salida. Debe ser puro
  respecto del estado físico y usar el stream de captura vigente.
- `time` es un array de dispositivo de un elemento, segundos desde el origen
  de la época. El llamador mantiene el origen absoluto y actualiza inputs entre
  épocas. `side` es `LEFT=-1` o `RIGHT=1`: los eventos exactamente en `time`
  se incluyen sólo por la derecha. No se usa un epsilon temporal.
- `project(y,time,side)`, opcional, devuelve estado FP64 contiguo sin modificar
  `y`. Es puro e idempotente a tiempo/lado fijos; puede devolver `y` sin cambios.
  Los eventos y variables proyectados pertenecen al modelo, no al integrador.
- `advance(ns,next_ns,min_ns,max_ns,boundaries=...,max_attempts=10000)` recibe
  duración/propuestas enteras en ns y fronteras relativas en segundos. El
  extremo autorizado llega directamente del controlador. No se reconstruye
  en la proyección. Cuartos, mitad y tres cuartos deben ser representables y
  estrictamente interiores; de lo contrario la época falla con rollback.
- El límite de intentos es real. No se ofrece un parámetro de tiempo de pared
  que el controlador ignore; el ejecutor del proceso limita la pared.
- `advance` y `close` rechazan operaciones solapadas antes de tocar el respaldo.
  El ABI C de bajo nivel requiere que el llamador conserve y serialice la
  propiedad del handle; la interfaz Python aplica esa exclusión.
- Una época fallida restaura el vector exactamente. No restaura efectos
  laterales de callbacks impuros, que están fuera del contrato.

Este método sirve a la forma de ecuaciones declarada. No se afirma que resuelva
eficientemente cualquier modelo neuronal ni que haya sustituido las masas y
solvers espaciales/PN. `real_model.py` y `event_projection.py` son adaptadores
del organismo; el segundo conserva los filtros q/s y el orden SET/ADD.

## Compilar y comprobar el núcleo

Requiere CuPy y CUDA compatible; en este equipo Python 3.10/CuPy 13.6, NVCC 12.1,
GCC 12, RTX 4070 Ti SUPER (sm_89). `build.py` permite especificar herramientas.
Desde una extracción nueva:

```bash
python build.py --out librebuilt_controller.so
python check_runtime.py --library librebuilt_controller.so
python -O check_runtime.py --library librebuilt_controller.so
```

Las comprobaciones usan solución logística exacta con buffers reutilizados,
relajación con evento, lados izquierdo/derecho y SET/ADD simultáneos en CUDA,
rechazos y restauración tras progreso parcial. No validan por sí solas el organismo.
La biblioteca reconstruida tiene otro nombre para conservar el binario original
y sus hashes. Desde el ZIP de evidencia, `python verify_saved.py` y
`python -O verify_saved.py` verifican hashes y recalculan la comparación real
guardada; no necesitan el modelo histórico ni vuelven a simular el organismo.

## Prueba real acotada

`run_pair.py` ejecuta como máximo dos procesos de 100 ms desde el mismo checkpoint
real, con olor lateral nominal. Depende del organismo/checkpoint locales
AXIOMA_FLYWIRE y de sus adaptadores preservados; el paquete del núcleo no los
reconstruye. No altera archivos o procesos de la sesión estable. Conserva todas
las muestras CNS/cuerpo/axones/membranas, eventos y estados finales PN.

```bash
OPENBLAS_NUM_THREADS=1 /home/daroch/miniconda3/envs/GPU/bin/python -u run_pair.py
python compare_real.py reference100_01 candidate100_01 --out COMPARACION_NUEVA.json
```

`PLAN.md` fija alcance, presupuesto y umbrales anteriores a las corridas.
`RUNS.json` registra salidas de proceso y pared; `PAIR100.json` registra la
comparación científica. Un fallo de comparación no se convierte en PASS al
empaquetar. Los snapshots compactos PN no son reinicios completos del organismo.
