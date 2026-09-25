# Neurocore: CNS y membranas con modelos separados

Esta etapa incorpora las membranas al núcleo de ejecución reutilizable. El CNS
conserva exactamente el código matemático validado en la etapa anterior. Las
membranas conservan su masa densa, canales, integración y eventos físicos; su
backend ya no importa kernels históricos ni los modifica mediante búsquedas y
sustituciones de texto. PN y el acoplamiento del organismo siguen conservados.

## Componentes y responsabilidad

| Archivo | Responsabilidad |
|---|---|
| `resident_controller.cu`, `graph_runtime.py` | CNS: control residente y punto medio exponencial con cinco evaluaciones |
| `dense_warp.cuh` | Álgebra FP64 de bloques densos de 1 a 32 variables; residual del operador original |
| `independent_blocks.cuh` | Adaptación, aceptación y límite de intentos por bloque independiente |
| `block_executor.py` | Compilación, memoria privada de época, exclusión y publicación sólo si todos los bloques pasan |
| `membrane_model.cu` | Ecuaciones y eventos físicos del modelo de membrana de 17 coordenadas |
| `linear_mass_model.cu` | Otro modelo: `M dy/dt = f-Ky`, con masa completa no identidad |
| `device_cell.py`, `real_model.py` | Puentes del organismo heredado a los nuevos ejecutores |

No hay anatomía ni leyes de canales en los dos headers genéricos. Un modelo
declara `trial(h)` y `commit(used,h)` para el controlador de bloques. El
ensamblado de matrices, las tolerancias y las leyes físicas pertenecen al modelo.
La independencia significa que ningún bloque lee el estado que otro bloque
está avanzando durante una época. Los inputs se mantienen conforme al método de
acoplamiento original. La infraestructura no convierte bloques acoplados en
independientes por suposición.

## Contratos del núcleo

`dense_warp_solve<N>` asigna una fila completa por lane CUDA, admite `1<=N<=32`
y exige participación de los lanes `0..N-1`. Conserva fila y RHS originales.
No pivota: requiere pivotes positivos y residual finito dentro del límite
explícito. Rechaza operadores que incumplen; no sustituye la masa por identidad.
Un residual pequeño no garantiza buen condicionamiento ni error temporal pequeño.

`BlockExecutor(source, name, count, state)` recibe campos CuPy contiguos con una
dimensión inicial por bloque. `run(state, bind_arguments)` copia el estado a
memoria privada, ejecuta y comprueba todos los estados de salida antes de entregar
arrays independientes. Un fallo deja intacta la entrada y permite reintentar con
esa entrada. La función de enlace y el kernel no deben mutar los campos del
llamador ni producir efectos físicos externos. El adaptador del organismo
publica después los eventos; un fallo de esa publicación invalida al adaptador.

La compilación incluye el hash de los headers en la identidad del código para
que cambiar un header no reutilice por accidente un kernel antiguo de la caché.
Se usa FP64, división precisa y FMA desactivado, sin fast math. Esta separación
no promete que un único solver sirva para todas las escalas o matrices neuronales.

## Reproducción desde una extracción nueva

Requiere Python, NumPy, SciPy, CuPy y CUDA compatible. Entorno probado: ver
`ENVIRONMENT.json`. No se necesita el árbol histórico para estos controles:

```bash
python build.py --out librebuilt_controller.so
python check_runtime.py --library librebuilt_controller.so
python check_blocks.py
python -O check_blocks.py
```

Los controles de bloques contrastan N=1/3/17/32 con CPU, rechazo de sistemas
inválidos y de un residual superior al umbral, un modelo de masa densa de tres
coordenadas frente a la exponencial matricial exacta, fallos tras avances
parciales, aislamiento de todos los bloques y reintento limpio. No sustituyen
la comparación del organismo.

El ZIP de evidencia permite recalcular todas las muestras de la pareja real:

```bash
python verify_saved.py
python -O verify_saved.py
```

Reejecutar el organismo completo requiere el loader, modelo y checkpoint locales
originales AXIOMA_FLYWIRE/AXIOMA_ASTRA. Los snapshots PN compactos permiten
comparar su estado final, pero no reiniciar por sí solos todo el organismo.
`run_pair.py` ejecuta la pareja prospectiva una sola vez y rechaza relanzarla.
Para una campaña nueva se requiere otro directorio y un presupuesto previo.

## Desarrollo externo y límites

El usuario eligió la tarea «ChatGPT C++/Cuda»; la app la identifica como Codex.
Desarrolló el header matemático en `external/` por lectura del donante, sin
ejecutar GPU. La copia integrada sólo adapta la inclusión de `math.h` a NVRTC;
se conservan original, nota y hashes. Jev emitió una clasificación de prioridades,
no código ni un veredicto de corrección. Las pruebas y la integración se ejecutan
localmente. El presupuesto y los criterios están fijados en `PLAN.md`.

Este cierre concierne a ingeniería del motor y a compatibilidad numérica en el
alcance ensayado. No acredita equivalencia biológica ni navegación/vuelo.
