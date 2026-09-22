# Motor general v2 — resultado de la ronda del 22-09-2026

**PROMETEDOR_NO_CONFIRMADO.** Reparaciones de contratos y primer backend implícito general CUDA ejecutados. No representa un cerebro biológico completo ni supera la etapa 3. El explícito sigue siendo la elección demostrada para las cargas suaves evaluadas; este puente implícito no se promueve como reemplazo universal.

## Resultados reconstruidos desde arrays

| Caso | Modo | Avance + lectura (s reales) | Error máximo normalizado | Cumple criterio |
|---|---|---:|---:|---|
| clamp | fast | 0.474772 | 2.22044605e-16 | True |
| clamp | precise | 0.466775 | 2.17690789e-16 | True |
| stiff | fast | 0.918003 | 2.54036106e-06 | True |
| stiff | precise | 10.468656 | 1.33425921e-08 | True |
| hh | fast | 7.696267 | 0.00268866178 | True |
| hh | precise | 54.075725 | 2.07431454e-06 | True |
| mixed | fast | 2.579798 | 9.34465162e-05 | True |
| mixed | precise | 7.626341 | 2.2998371e-08 | True |

clamp y stiff: 1 s simulado; HH: 0,05 s; mixed pequeño: 0,2 s. Error = max |candidata−referencia|/(escala física + |referencia|). Límites externos conservados: rápido 0,01; preciso 1e-5. No es un porcentaje uniforme de error biológico. Referencias HH/mixed: Radau con dos tolerancias; los otros casos tienen solución analítica. Cohorte expuesta al autor, sin ajuste tras resultados.

## Carga grande y decisión

726.900 estados y 23.296.700 conexiones; un segundo continuo. Exactitud inspeccionada en 4.096 sondas × 21 instantes y TODOS los estados finales. Referencias previas DOP853 reutilizadas con hashes: descriptor idéntico, RHS completo idéntico en tres contextos de control; no prueba de equivalencia universal de todos los posibles descriptores.
- fast: 6.578951 s de avance/lecturas, 1.741513 s de preparación del integrador; error 0.000226880644; criterio numérico: True.
- precise: 82.422156 s de avance/lecturas, 1.766835 s de preparación del integrador; error 2.10951099e-08; criterio numérico: True.

El puente preciso supera el minuto objetivo en esta carga. Las cifras son mediciones exploratorias únicas, sin intervalo de confianza ni control exclusivo del equipo. El inicio de la corrida grande rápida coincidió con una prueba pequeña de sesiones; no se usa como estimación de rendimiento aislado. `measured_window_s` incluye preparación del modelo y controles desde el punto declarado, pero no importaciones; donde existe, `subprocess_wall_s` incluye el proceso completo. No llamamos tiempo total del proceso a una ventana interna. Pico RAM medido ~2,05 GiB; no se midió pico VRAM total incluyendo asignaciones de SUNDIALS.

En HH, comparación nueva con exactamente el mismo descriptor y la misma referencia:
- Explícito fast: 0.055631 s, error 0.00182985748, para 0,05 s simulados. El implícito correspondiente figura arriba y es más lento.
- Explícito precise: 0.273242 s, error 8.93156294e-07, para 0,05 s simulados. El implícito correspondiente figura arriba y es más lento.

No elegir un único integrador por preferencia arquitectónica. La interfaz general de modelos es compartida; la selección futura debe responder a rigidez, masa y coste medidos, nunca al nombre anatómico. Este resultado refuta que mover el cálculo a GPU por sí solo garantice velocidad.

## Cambios y límites

- Semántica FP64 de literales en CPU/GPU; identidad incluye orden de almacenamiento; índices validados antes de convertir; factores de unidades no implementados se rechazan.
- Reanudación explícita comprueba perfiles efectivos y huellas de ejecución. Modificaciones inválidas en el estado vivo se rechazan antes de sustituirlo.
- JVP generado sigue todas las conexiones: F_x v + F_u W(H_x v). La diagonal local solo precondiciona; no sustituye al Jacobiano global.
- ARKStep, dos SPGMR y vectores CUDA comparten stream y mantienen vectores grandes en GPU. Control adaptativo, QR pequeña y callbacks siguen coordinados desde CPU/Python. Hay sincronizaciones escalares comprobadas por llamada; no se describe como un bucle enteramente residente.
- Masa dispersa constante admitida solo si tiene diagonal positiva y dominancia estricta por filas, después de aplicar clamps. No DAEs, masas singulares, ruido, retardos, eventos discretos o cuerpo.
- Sesiones implícitas permiten inspección y modificaciones por épocas, añadiendo/eliminando estados y conexiones e inmovilizando variables. `save_restart` guarda estado físico y REINICIA adaptación; no guarda la historia completa de SUNDIALS. No debe anunciarse como checkpoint bit a bit del integrador.
- Verificador anterior reparado en una copia: cohorte exacta 4×2×2 y bandera eligible reconstruida. Las 16 condiciones originales sí estaban completas; había una debilidad de verificación, no evidencia de que aquellos datos fueran falsos.

## Revisión y autocritica

ChatGPT identificó contraejemplos reales en las fuentes y después orientó la integración con SUNDIALS. Ambas respuestas son documentales: no ejecutó el código nuevo. Jev clasificó tareas acotadas en una llamada (1.011 tokens entrada, 213 salida); su salida no es validación numérica. No se usaron subagentes Codex.

La crítica válida es que una arquitectura extensible necesita contratos de identidad, intervención y derivadas acopladas, además de rapidez. La crítica a nuestra propia decisión es que un integrador implícito con muchos intercambios pequeños puede ser mucho más lento incluso conservando estados en GPU. No ocultamos ese negativo ni sustituimos la evidencia por los TFLOPS de la tarjeta. El próximo trabajo útil es medir y reducir la coordinación del puente y mejorar precondicionamiento general cuando la rigidez justifique el método, conservando el explícito competente.

A: implícito global, implementado en esta ronda. B: multirrate con corrección acoplada, condicionado a separación temporal demostrada. C: IMEX con operador lineal global declarado, no implementado; distinto del exponencial diagonal rechazado previamente. Un solo prototipo nuevo completo; no se reabre el negativo mediante retocar tolerancias.

## Procedencia, ejecución y reproducción

SUNDIALS v7.6.0, commit `ddf5daba8397ea89287a0fec6f1b3bc3fe6c548b`, licencia BSD-3 conservada. El paquete incluye fuentes necesarias para construirlo sin descargarlas; requiere Linux, CUDA/nvcc, CMake, GCC/G++12 y Python con NumPy/SciPy/CuPy. Configuración local comprobada: RTX4070 Ti SUPER, CUDA12.0, arquitectura89, doble precisión. La sugerencia posterior de ChatGPT sobre7.9 no se trató como compatibilidad automática: la versión ejecutada está fijada. [Fuentes oficiales](https://github.com/LLNL/sundials/tree/v7.6.0).

Desde una extracción limpia, usando el entorno GPU ya instalado:

```bash
PY=/home/daroch/miniconda3/envs/GPU/bin/python
$PY -B -O verify_package.py
$PY -B -O verify.py verified_again.json
$PY -B build_dependency.py
$PY -B -O test_contracts.py check_contracts
$PY -B -O test_jvp.py check_jvp
$PY -B -O test_session.py check_session
$PY -B -O pilot.py check_clamp --case clamp --profile precise
```

Los nombres de salida deben ser nuevos. El modo corto reconstruye todos los resultados guardados y repite contratos, derivadas, sesiones y un ensayo pequeño; NO repite los 16 solves científicos. Repetición de la campaña completa de esta ronda (incluye construir una copia local, 16 solves, máximo180s por proceso; referencias grandes históricas se conservan y verifican):

```bash
$PY -B reproduce.py nueva_reproduccion
```

`PILOT_FREEZE.json` conserva las fuentes de los pilotos; `LARGE_RUNNER_REPAIR.json` registra una reparación del importador de referencia antes de que comenzara la simulación grande. No cambió ecuaciones ni criterios. `contracts_01/snapshot` y `session_01/restart` son fixtures deliberadamente corrompidos al final de sus pruebas, no puntos válidos de reanudación. Preservados como negativos.

Motivo de parada: hito de integración genérica y contraste empírico completo de una ronda acotada. Sigue pendiente optimizar el puente para regímenes rígidos grandes, incorporar capacidades neurobiológicas aún ausentes y cerrar etapa3. No se promete tiempo real para cualquier cerebro.

Extracción limpia comprobada: ocho comandos aprobados, incluida compilación local de SUNDIALS desde las fuentes incluidas, verificación de hashes antes/después y ejecución pequeña. Ver `CLEAN_VALIDATION.json`. No se repitió la campaña completa solo para empaquetar.
