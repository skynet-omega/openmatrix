# Reproducción de la campaña 56

El ZIP compacto contiene datos registrados, contratos, verificador, banco temporal y fuentes nuevas. Requiere Python 3 con NumPy y SciPy ya instalados; Matplotlib sólo para regenerar la figura. Las versiones efectivas están en `ENVIRONMENT.json`. Las comprobaciones siguen activas con `-O`.

## Modo corto, desde una extracción nueva

Dentro del directorio `ETAPA45_TRANSFERENCIA_TEMPORAL_20260928_56` extraído:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH='' python3 -B -O check_delivery56.py --corruptions
```

Verifica el manifiesto, reconstruye lectores, señales consumidas, objetivos/márgenes descendentes, efectos factoriales y presupuesto; después comprueba las corrupciones deliberadas. También resuelve nuevamente el banco temporal y sus 16 contrastes independientes de integración. No importa fuentes del árbol de trabajo, no usa CUDA y no ejecuta un nuevo cerebro. El resultado JSON indica explícitamente cero milisegundos CNS nuevos.

No usar `--write` contra resultados archivados: las salidas científicas son inmutables. El informe y la figura se generaron con `report56.py` a partir de esos resultados; para regenerarlos usar otra copia retirando sólo las salidas de presentación de esa copia, sin cambiar los datos.

## Estados completos y fuentes transitivas

El archivo local `ETAPA45_TRANSFERENCIA_TEMPORAL_20260928_56_ESTADOS_Y_FUENTES.zip` conserva los dos estados iniciales de 48, los ocho estados finales de 56, los registros y las dependencias locales identificadas. Almacena una sola copia de cada contenido SHA256 y un índice que reconstruye todos los nombres lógicos. `FULL_CAPSULE.json` identifica el archivo y su hash. Los estados contienen estado neuronal, corporal, reloj, RNG, parámetros y metadatos de intervención.

```bash
python3 -B -O restore_capsule56.py ../ETAPA45_TRANSFERENCIA_TEMPORAL_20260928_56_ESTADOS_Y_FUENTES.zip
python3 -B -O restore_capsule56.py ../ETAPA45_TRANSFERENCIA_TEMPORAL_20260928_56_ESTADOS_Y_FUENTES.zip --restore-to ../ESTADOS56_EXTRAIDOS
```

El primer comando verifica todos los contenidos sin duplicarlos en disco. El segundo exige un destino inexistente y reconstruye los nombres relativos; nunca escribe automáticamente en las rutas absolutas de origen. La expansión ocupa más que el ZIP por los archivos idénticos con nombres diferentes. `SOURCE_INDEX.json` conserva la relación entre fuentes originales y copias. La verificación por lectura de todo el ZIP es preservación de evidencia, no una nueva reanudación GPU.

## Modo completo, nueva vida en el host original

Este modo consume una nueva exposición y **no se ejecutó durante el cierre de 56**. Requiere el entorno CUDA y el árbol histórico exacto de `/home/daroch/AXIOMA_ASTRA` y `/home/daroch/AXIOMA_FLYWIRE/matrix`. La reubicación de todo el cargador GPU a un sistema limpio no está cualificada. El paquete conserva las fuentes; no oculta esta dependencia de rutas.

```bash
OPENBLAS_NUM_THREADS=1 /home/daroch/miniconda3/envs/GPU/bin/python -B -O repeat_frozen56.py --project-root /home/daroch/AXIOMA_ASTRA
```

Primero verifica todos los prerrequisitos, crea un directorio único y ejecuta la cola reparada: dos cualificaciones exactas de 2 ms y ocho brazos de 128 ms, 1028 ms CNS nuevos. Conserva los topes reducidos de la reparación: 1198 ms CNS, 5030 s CPU de trabajadores y 4637,87 s de cola. No repite deliberadamente el fallo de codificación, dependiente del entorno; copia su evidencia histórica y lo declara en `NEW_EXPOSURE.json`. El verificador conserva su cargo histórico conservador, distinguido del nuevo coste medido.

## Qué no reproduce el paquete compacto

El programa original de Motor `aporte_motor/reconstruir_frontera.py` lee estados originales de 48/55 en sus rutas. Su informe y procedencia están incluidos; ese análisis completo no forma parte del modo corto portátil. Tampoco se reproducen aquí datos biológicos nuevos, navegación hacia un olor espacial, viento ni marcha con seis patas. Los enlaces de publicación y las extracciones comprobadas están en los recibos de cierre.
