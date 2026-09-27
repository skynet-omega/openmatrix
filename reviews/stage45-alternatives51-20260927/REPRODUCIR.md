# Reproducción de la evidencia de 51

El ZIP compacto contiene todos los registros de las diez condiciones de 90 ms y las cuatro comprobaciones de 4 ms, contratos, lectores, operandos locales, fuentes de las candidatas y verificadores. El paquete completo adicional conserva estados finales, donantes de 48 y dependencias locales transitivas mediante segmentos deduplicados sin pérdida. Los hashes permiten recuperar los archivos originales byte por byte.

## Modo corto: extracción limpia y recomputación CPU

Requiere Python y NumPy (entorno original en ENVIRONMENT.json). No importa fuentes del árbol de ASTRA, ni usa GPU o conexión externa. Ejecutar desde una carpeta nueva que contenga el ZIP:

```bash
python -m zipfile -e ETAPA45_ALTERNATIVAS_20260927_51.zip evidencia51
cd evidencia51/ETAPA45_ALTERNATIVAS_20260927_51
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python verify_complete51.py
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -O verify_complete51.py
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -O test_corruptions51.py
```

Esto comprueba hashes, identidad, reloj, consumo de entradas, ecuaciones, cualificación, cinco lectores nuevos y contrastes desde arrays. Las corrupciones modifican datos/criterios y actualizan su hash para exigir detección semántica; restauran los bytes al terminar. Las cifras y el veredicto se reconstruyen, no se acepta un booleano PASS guardado. `analyze_pilot.py --out nueva_comparacion.json` permite obtener de nuevo las métricas; la figura se regenera con `make_figure.py` y Matplotlib.

## Modo completo: recuperar todos los originales

El archivo ETAPA45_ALTERNATIVAS_20260927_51_COMPLETO.zip usa recetas de archivos. Copiar `unpack_full51.py` desde el compacto y ejecutar, con espacio suficiente:

```bash
python unpack_full51.py ETAPA45_ALTERNATIVAS_20260927_51_COMPLETO.zip originales51
cd originales51/AXIOMA_ASTRA/campanas/etapa45_alternativas_20260927_51
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -O verify_complete51.py
```

El materializador comprueba tamaño y SHA256 de cada archivo. No descarga paquetes ni ejecuta automáticamente una nueva simulación. La campaña se lanzó originalmente con `run_pilot.py --qualification identity`, las otras tres comprobaciones mediante `qualify_queue.py`, `verify_qualification.py` y `run_queue.py`, con el intérprete y recursos registrados. **No se ha cualificado un relanzamiento GPU de 51 desde rutas reubicadas**, ni la continuación de su nuevo propietario de aire/conductancia. El esquema `pilot51_scientific_state_v1` se distingue deliberadamente del restaurador de 48/49; no eludir su rechazo. Por eso el modo completo acredita conservación y recomputación de la evidencia, no una nueva ejecución física equivalente.

Los scripts de cribas que leen antecedentes originales se incluyen con sus proyecciones pequeñas; algunas rutas originales están fijadas en su código. Los verificadores portables utilizan sólo los registros incluidos. Las pruebas de trasplante PN no abarcan aún eventos/cachés externos vivos. Los resultados de propiocepción son un test de identificabilidad y un antecedente negativo, no un replay nuevo del organismo.
