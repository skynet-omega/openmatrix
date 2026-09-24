# Base nativa limpia: resultado del primer ensayo

**Clasificación: paridad exacta del camino sin rechazos en 1 ms; controlador completo todavía no admitido y sin ganancia de velocidad.** No es un organismo libre de Python ni un motor de 5 s/5–10 min. El modelo del conectoma y los propietarios conservados se prepararon con Python; dentro de cada época CNS de 125 µs, el nuevo [controlador CUDA](resident_controller.cu) ejecutó el grafo neuronal y tomó decisiones de aceptación en la GPU sin sincronizar al host por cada intento. Python todavía acopla PN/KC, cuerpo y sensores cada milisegundo.

El [plan previo](PLAN.json) fijó un fixture CUDA y una pareja real de 1 ms. El [fixture](FIXTURE_SOURCE_LOCK_01.json) pasó: siete relanzamientos nativos en 0,688 s de proceso, sin pretensión de velocidad neuronal. El [primer candidato real](real_resident_01/RESULT.json) se detuvo antes de avanzar porque CuPy destruye `cudaGraph_t` después de compilarlo. Se conservó el fallo, se identificó [la causa y reparación exacta](INTERFACE_FAILURE_01.md), y se congeló un [presupuesto nuevo](REAL_REPLAY_REPAIR_LOCK_02.json) para repetir sólo el candidato.

El [control](real_reference_01/RESULT.json) y el [candidato reparado](real_resident_02/RESULT.json) parten del mismo checkpoint asentado, sham y 1 ms. Ambos completaron 16 épocas CNS, 191 pasos aceptados, cero rechazados y el mismo reloj final. El [verificador](REAL_PAIR_VERIFY_01.json) reconstruyó `session`, `prosthesis`, `published`, `effective_operator` y la frontera física desde sus archivos: **igualdad semántica exacta**, también igual SHA-256 en los archivos auxiliares. Detectó corrupciones deliberadas de escalar y matriz.

| Brazo | Tiempo del paso corporal+neuronal de 1 ms | Tiempo total incluyendo carga y guardado |
| --- | ---: | ---: |
| Referencia | 3,808165 s | 46,246259 s |
| Residente | 3,902560 s | 46,498963 s |

La razón de pared fue 1,02479: el candidato fue 2,48 % **más lento en esta pareja**. Una diferencia tan pequeña de un único milisegundo no prueba una regresión sostenida, pero tampoco justifica promoverlo como aceleración. El controlador residente espera al final de cada época: puede quitar sincronizaciones intermedias sin quitar el trabajo de las 25,58 millones de sinapsis que consume el grafo capturado. Los 191 pasos aceptados no ejercieron la rama de rechazo. Una [prueba sintética congelada de rechazo y frontera de evento](REJECTION_FIXTURE_RESULT_01.json) falló: coincidió el estado final impreso, pero no la secuencia de aceptaciones/rechazos frente al oráculo escalar independiente. No se ajustó ni repitió esa prueba. La paridad de rechazo/rollback permanece abierta.

La [tabla de donantes](DONORS_20260924.md) preserva las rutas multirritmo, QSS, Krylov, eventos dispersos, FP32 y SpMM con sus falsadores. Nada de eso fue descartado por este resultado. Para acercarse al objetivo de velocidad se necesita reducir trabajo recurrente **y** fronteras de propietarios sobre el organismo real, sin atribuir a un microbenchmark una ganancia total. La etapa 4 de navegación y la etapa 5 siguen abiertas.
