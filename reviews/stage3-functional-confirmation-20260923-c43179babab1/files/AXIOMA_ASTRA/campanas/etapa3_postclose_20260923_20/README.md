# Verificación poscierre preparada — sin ejecución sobre datos reales

El verificador exige `QUEUE.json` de la reparación19 en `COMPLETE` antes de importar su código o reservar un recibo. Luego ejecuta nuevamente `verify_repair19.verify()` con sus criterios originales, exige aprobación reconstruida y compara **todo** el resultado con `FINAL_RAW_VERIFIED.json`: valores, tipos, listas, banderas, limitaciones y hashes. Un booleano guardado no puede sustituir esta reconstrucción.

La independencia consiste en un proceso poscierre nuevo y comprobaciones de integridad/equivalencia; reutiliza deliberadamente el verificador científico original. No representa una nueva validación biológica independiente ni una ampliación del alcance funcional.

`SOURCE_LOCK.json` fija 18 archivos de código/contrato y sus dependencias pertinentes. Durante la ejecución se registran SHA256 y tamaños de todos los archivos locales realmente abiertos; al terminar se comprueba que no cambiaron. El proceso desactiva escritura de bytecode y bloquea importaciones de GPU/organismo y escrituras fuera de esta carpeta. El verificador heredado ya desactiva sus antiguos guardados de resultados intermedios.

La única salida real será `POSTCLOSE_VERIFIED_01.json`, creada exclusivamente y nunca sobrescrita. Si comienza una verificación y falla, su recibo queda BLOQUEADO/false; un archivo parcial tampoco es admisible. Si la cola está abierta o fallida, el proceso rechaza antes de crear ese recibo. No crea enlaces ni copias en campañas15/16/17/19.

El esquema superior incluye `queue_sha256`, `final_raw_sha256`, `verifier_sha256` (el archivo verify_repair19.py), `contract_sha256` (REPAIR19_CONTRACT.json), `classification`, `functional_stage3_pass`, `rebuilt_raw_result`, `source_hashes` y `read_input_hashes`. El gate posterior debe contrastar esos hashes con campaña19 y mantener el alcance y los negativos históricos del resultado completo.

**Preparación CPU:** 21 pruebas sintéticas por modo normal y `-O`, con guard de cola, excepción científica pese a PASS guardado, datos anidados alterados, bool→int, no finitos, claves duplicadas, cola mutada, escritura fuera de carpeta, lectura modificada, hashes de integración y rechazo de sobrescritura. Las salidas dentro de `fixtures_*` son explícitamente sintéticas; ninguna es el recibo real. Se conservaron las fuentes y recibos anteriores a añadir los cuatro hashes solicitados por el gate. Coste total de fixtures: 0.0408064 s CPU y 0.291443 GiB RSS máximo. READY.json fija versiones y deja `real_recomputations: 0`.

Sólo después del cierre de reparación19, una vez y en un proceso nuevo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/daroch/AXIOMA_FLYWIRE/matrix/.venv/bin/python -B \
/home/daroch/AXIOMA_ASTRA/campanas/etapa3_postclose_20260923_20/verify_postclose.py
```

El presupuesto futuro está congelado en PLAN: una recomputación, 180 s CPU, 300 s pared, 6 GiB de espacio de direcciones, cero GPU/organismos/pasos. Se reserva memoria para comparar los estados serializados completos; no se simula ni reentrena. Un fallo agota ese intento y conserva evidencia.

Para repetir exclusivamente las pruebas sintéticas, elegir una etiqueta nueva y usar los dos modos; esto no llama al verificador real:

```bash
python3 -B campanas/etapa3_postclose_20260923_20/test_postclose_cpu.py comprobacion_nueva
python3 -B -O campanas/etapa3_postclose_20260923_20/test_postclose_cpu.py comprobacion_nueva
```
