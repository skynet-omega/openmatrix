# Atribución de lecturas host del organismo real

24-09-2026. Dos brazos de 2 ms desde el mismo checkpoint asentado, sham, sin observador; sólo el segundo ms del brazo perfilado produjo `cProfile`. Fuentes y presupuesto fijados [antes de ejecutar](NONCNS_PSTATS_SOURCE_LOCK_01.json). No se cambiaron ecuaciones ni propietarios. Cada brazo completó los 2 ms; el control devolvió `COMPLETE`. El runner del perfil devolvió `INCOMPLETE` **después** de guardar `NONCNS_RAW.pstats` porque `python -I` no incorporó la carpeta del extractor a `sys.path`. Su intento de extracción por CLI tuvo además un defecto `pstats.Stats(Path)`; se utilizó el método congelado `extract(str(raw))` en un proceso offline. No se repitió ningún brazo ni se amplió el presupuesto.

El [verificador del bruto conservado](NONCNS_verify_saved.py) comparó todos los campos semánticos de `session`, archivos `prosthesis/published/boundary`, traza corporal y auditoría de eventos: **igualdad exacta** entre brazos. Reconstruyó de nuevo el perfil bruto, comprobó que el conteo de aristas llamador→llamado cierra sobre cada método y que coincide con el perfil plano. Dos corrupciones deliberadas de estado y de total de llamadas fueron detectadas. [Recibo](NONCNS_SALVAGED_VERIFY_01.json). Esto permite una atribución **diagnóstica salvada**, sin cambiar el estado `INCOMPLETE` del runner ni convertirla en una prueba de mejora.

En el segundo ms perfilado hubo 16 avances CNS/KC/PN, **960** llamadas `cupy.ndarray.get` y **204** `numpy.ndarray.tolist`. La distribución `.get` reconstruida de [`NONCNS_ARCS_SALVAGED.json`](noncns_profile_01/NONCNS_ARCS_SALVAGED.json) cierra exactamente:

| Llamador inmediato de `.get` | Llamadas | Tiempo acumulado de esa arista en cProfile |
| --- | ---: | ---: |
| `cupy.asnumpy` | 800 | 0,0879 s |
| `device_cell.advance` | 112 | 0,2492 s |
| PN `StageGraph.run` | 32 | 0,0450 s |
| `organism_adapter.step` | 16 | 0,0058 s |

De las 800 `asnumpy`, 416 vienen de `ProjectedKcBatch.host`, 112 de `advance_graph_resident`, 112 de `axon_gpu` y el resto de PN/otros consumidores. `tolist` aporta 204 conversiones CPU: 112 de PN `advance_graph_resident`, 32 de su comprensión y 16 de `device_cell.advance`; el resto se reparte entre sesiones y sensores. El tiempo propio total de `.get` fue 0,3859 s y el de `tolist` 0,0630 s en el ms perfilado; esos tiempos no miden bytes PCIe y pueden incluir espera del dispositivo. `NativeGraph.advance` registró 1,7417 s propios; el paso perfilado total 3,0826 s. `cProfile` y la sonda de sólo un ms introducen sesgo de medida; el perfil previo de 19 ms sigue siendo la base del cálculo de Amdahl.

**Decisión:** hay fronteras host concretas para rediseñar y medir, en particular las siete lecturas por avance de `device_cell`, los convertidores KC/PN y la decisión adaptativa `graph_control_v2.cpp` que sincroniza por intento. Este perfil no demuestra que trasladarlos a C++ ahorre todo el tiempo fuera de CNS. La siguiente prueba de F debe cerrar la frontera con un estado pareado y una medida integral, no sólo reducir el número de llamadas.
