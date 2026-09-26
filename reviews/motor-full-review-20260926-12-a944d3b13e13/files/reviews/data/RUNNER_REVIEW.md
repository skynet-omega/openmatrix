# Revisión independiente CPU del runner 12

26-09-2026. Alcance: `run_trial.py`, `PLAN.json`, restauración/identidad inicial, snapshot, inventario/congelación de fuentes y viento. No se importó el organismo, no se ejecutó MuJoCo ni se asignó GPU en esta revisión. No se revisó la aritmética interna del engine, que estaba siendo corregida por otro especialista, ni el comparador todavía pendiente. Fuentes examinadas y sus hashes al cierre: [RUNNER_CPU_V2.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/RUNNER_CPU_V2.json). Se conserva [RUNNER_CPU.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/RUNNER_CPU.json) como evidencia de la primera exposición.

**No encontré un bloqueador pendiente del runner para igualdad de preparación, calendario o interfaz motora de los dos brazos. El hueco material de congelación detectado quedó resuelto en el generador del lock.** La emisión de `SOURCES.json` queda como paso operativo previo a los brazos: aún no existía al cierre de esta revisión y el runner exige leerlo y verificarlo en toda ejecución no diagnóstica. No hace falta cambiar el modelo, la PN ni la física acordada.

## Hallazgo material corregido: lock de inputs

[run_trial.py:106](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/run_trial.py:106) lee el campo desde `navigation_minus_filtered_wind_03/GAUSSIAN_SPEC.json` **después** de `qualify_initial`. [source_inventory.py:10](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/source_inventory.py:10) inventaría código y bibliotecas nativas. El `SOURCE_LOCK.json` histórico sí cubre el código gaussiano y la geometría, pero no ese JSON con fuente/sigma. La función de instalación valida pose, forma y finitud, sin exigir que `source_mm` y `sigma_mm` sean la fórmula congelada de otro archivo. Por ello, sólo congelar `imported()` no excluía un cambio del estímulo entre brazos.

El nuevo [freeze_sources.py:11](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/freeze_sources.py:11) resuelve el hueco reuniendo fuentes realmente ejecutadas por el diagnóstico, lock histórico, fuentes del runner/engine/PN y explícitamente:

- `PLAN.json` de esta ronda.
- El `GAUSSIAN_SPEC.json` consumido.
- `traces.npz` usado para la fila preparada 39.
- `prepared_state/MANIFEST.json` y los archivos `.json/.npz/.py` de ese estado.
- Configuración del loader, `nodes.parquet`, procedencia y archivos `.json/.npz/.py` del checkpoint cerebral.
- Lock histórico y geometría, conservados.

El mapping usa hashes calculados de esos archivos y creación exclusiva (`open('x')`); `verify(source_lock)` corre antes de importar/cargar el organismo. No observé inputs científicos alterados ni condiciones distintas entre dos ejecuciones. No ejecuté la congelación, responsabilidad del padre una vez terminadas las correcciones. El inventario actual captura también ejecución por loaders dinámicos mediante audit hook; `full_snapshot` se importa antes del recibo inicial. Estas correcciones evitan omisiones sin afirmar que un hash demuestre equivalencia numérica.

## Condiciones comunes verificadas por lectura

| Contrato | Evidencia y conclusión |
|---|---|
| Estado de partida | Ambos brazos llaman al mismo loader y a la misma `restore(PREFIX/prepared_state)`; `qualify_initial` compara árboles completos de sesión, prótesis, publicación, operador efectivo y frontera. El runner exige `initial.exact`. [run_trial.py:77](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/run_trial.py:77). |
| Reloj y parámetros | Ambos parten de 44.486.000.000 ns de la preparación y reciben el mismo `next_step_ns` guardado. General PN está explícitamente apagado. La comprobación de IDs/operador fue revisada en DATASET. Ninguno de esos límites de modelo bloquea esta comparación. |
| Diferencia de motor | Sólo `reviewed` instala `HERE/engine/real_model`; `stable` conserva el NativeGraph del adaptador común. Ambos ejecutan `RuntimeSession(h,'causal_cuda')`, mismo propietario de eventos de campaña 15, misma PN y células. La comparación es referencia estable corregida frente a motor revisado, no reproducción binaria de la antigua campaña 40. |
| Reparación PN común | `install_pn_overlay()` sucede antes de `RuntimeSession` en ambos brazos; rechaza nombres PN/consumidores ya importados. Por lectura, no carga un segundo organismo ni modifica el estado científico. [pn/install_pn.py:17](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/pn/install_pn.py:17). La validación del stream corresponde a la revisión CUDA. |
| Olor y lector | Mismo campo `minus`, misma fila de preparación, instalación tras identidad inicial, mismo baseline y filtro motor. El auditor verifica que muestra usada, muestra pendiente y pose sean las del intervalo; también que el mando bruto venga del DN del intervalo previo y que el filtrado llegue al cuerpo. |
| Continuidad | Un solo bucle `range(1,a.ms+1)`, sin restore en 1.000 ms. Los checkpoints de 1.000/1.020/1.500 ms sólo serializan PN y publicación. No reinstalan el motor ni reinician su filtro. |
| Tiempo | Por tick se exige `motor.body_calls=40*k` y `h.time_ns=initial_ns+k*1.000.000`; el capturador/auditor también comprueban PN, CNS, cuerpo y mundo. La vida de 2 s termina a 46.486.000.000 ns. |
| Viento | Mismos valores y clase `MotorWind` para ambos brazos: ticks 1001–1020, torque mundo z `−0,004672697857153467` nativo. `WorldTorque` reconstruye cinemática/COM en `MjData` privado a partir de qpos/mocap actuales; no ejecuta dinámica sobre el estado vivo. |
| Telemetría de dosis | El runner exige conteo acumulado exacto; la clase rechaza un mapeo activo cero y guarda cada aplicación con qpos, tiempo corporal y fuerza generalizada. Se conserva `wind_substeps.npz` al acabar. El campo por tick `wind_torque_native` es calendario; la evidencia física es el registro por subpaso. |

## Sonda independiente ejecutada

[check_runner_cpu.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/check_runner_cpu.py) extrae mediante AST las clases `MotorWind` reales, conservando sus métodos, y las ejecuta con cuerpo/controlador/mapeador falsos. No sustituye el calendario ni el filtro por una reimplementación. Usa una referencia escalar independiente del filtro para verificar continuidad. Presupuesto: 30 s; ejecución: **0,151 s**.

Resultados:

- **80.000** llamadas corporales para 2.000 ticks de 1 ms.
- **800** llamadas de viento, exactamente las **40.001–40.800**.
- Extremos izquierdos relativos de las aplicaciones: **1,000000–1,019975 s**; el último subpaso termina en **1,020000 s**.
- El filtro continúa antes, durante y después de la ventana, sin reset; el modo vuelve a `neural` al retornar cada llamada.
- La pose registrada es la anterior a cada subpaso; todas las aplicaciones falsas son no nulas.
- El recibo de `motor.audit()` tras las 800 aplicaciones se serializa con `allow_nan=False` y vuelve a leer `nonzero_substeps=800`. La revisión V2 verifica el `int(sum(...))` que evita un `np.int64` no serializable cuando hay viento.
- Las fuentes Python examinadas parsean sin errores.

Esto valida **calendario, continuidad de memoria e instrumentación** de esos métodos. No valida fuerzas físicas, integración del organismo ni equivalencia de motores. La prueba física independiente del padre [WIND_CPU_V2.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/WIND_CPU_V2.json) informa igualdad hot/restored durante 40 subpasos y equivalencia con el pipeline posicional completo, en su fixture corporal. Se mantiene su alcance separado.

## Snapshot y evidencia final

[full_snapshot.py:11](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/full_snapshot.py:11) guarda sesión completa, prótesis, publicación/RNG, operador efectivo, memoria del filtro y contadores motores, frontera e intervalos gaussianos; usa directorio `.partial` y manifiesto de hashes antes del renombrado final. El runner sólo lo pide después de completar los 2.000 ticks. La fase declarada, después del paso comprometido y antes de consumir la siguiente muestra, coincide con su ubicación.

`restart_tested=false` es correcto. No hay restaurador completo de este nuevo snapshot ni se necesita para dos vidas continuas; guardar todos esos propietarios no demuestra todavía reanudación entre procesos. La ausencia de esa prueba no debe bloquear la comparación acordada.

## Deuda y obligaciones del comparador; no bloqueadores adicionales del runner

1. **No aceptar `COMPLETE` como equivalencia.** El comparador pendiente debe exigir mismos hashes de plan/inputs/inicialización, 2.000 ticks, relojes, dosis, mandos y contactos, además de los umbrales de `PLAN.json`; informar estados PN/neuronales y eventos por separado. La etiqueta `parameters_changed=false` es una declaración del runner, no una prueba independiente.
2. **Separar predictores de compromisos.** `EVENTS.json` puede incluir ambos. Sus recuentos agregados no son por sí mismos espigas únicas de la vida aceptada. El comparador no estaba disponible en esta revisión y no se certifica aquí.
3. **Finitud de estado completo al cerrar.** La observación por tick y `collect` comprueban sus campos, y los solvers tienen sus propias guardas. El codec `write_state` no rechaza NaN/Inf dentro de todos los ndarray; guardar un snapshot no es por sí solo esa validación. La revisión de estado completo del comparador debe rechazar no finitos en todos los arrays comparados, sin tratar `NaN==NaN` como igualdad.
4. **Presupuesto/medición.** El CLI acepta `--wall-limit`; el lanzador debe pasar los límites del plan (450 s por brazo corto; 8.500/4.800 s largos). La vida diagnóstica de 20 ms lleva cProfile y se etiqueta aparte. `advance_total_s` excluye exportaciones y gran parte de instrumentación; `wall_total_s` las incluye. El límite de disco del plan no se impone en este runner, y el check de VRAM es una muestra por tick, no medición del pico transitorio. Son límites de atribución/control operativo, no evidencia de que se hayan excedido.
5. **Finalizar fuentes antes del lock.** `freeze_sources` incluye los `.py` de la raíz, engine y PN; el inventario de ejecución añade módulos importados/ejecutados y vecinos nativos, sin ampliar a todos los `.py` vecinos. Terminar comparador y auxiliares antes de emitir el lock evita recibos desactualizados. Los informes Markdown y JSON de revisión quedan fuera de ese rastreo.

Veredicto de esta revisión: **runner coherente para condiciones comunes; sin bloqueador material pendiente identificado en el alcance revisado**. La emisión del lock y la revisión del comparador siguen pendientes al cierre, y la equivalencia funcional depende de resultados emparejados. No hay motivo identificado aquí para modificar ciencia, volver a preparar el organismo ni cargar otro dataset.
