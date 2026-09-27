# Revisión del selector y observables de la ronda 50

**No se encontró un fallo que invalide el selector temporal ni la identidad de la pareja PN en el alcance revisado.** Revisión de fuentes y metadatos anatómicos, sin leer resultados neuronales parciales de la ronda 50. No acredita un resultado biológico ni la superación de las etapas 4/5.

Se revisaron `PLAN.json`, `PROPUESTA_PREVIA.md`, `temporal_input.py`, `test_design.py`, `run_arm.py`, el consumidor de `resume49.py` y la captura `matrix_olfactory_diagnostic.py`. Las ocho fuentes contrastadas con el contrato conservaron sus hashes. Se escribieron únicamente los archivos de este directorio; no se modificaron fuentes congeladas, cola ni estados. `test_design.py` se leyó sin ejecutarlo, pues escribe un control canónico fuera de este directorio.

## Entrada realmente seleccionada

Una comprobación CPU con buffers y stream simulados verificó las 1.120 llamadas de las ocho condiciones contra un calendario calculado independientemente. No fueron pasos del organismo ni llamadas CUDA. Los resultados detallados están en [REVISION.json](REVISION.json).

- Cada condición tiene 10 ms basales, 120 ms de estímulo y 10 ms de cola basal; la ventana activa corresponde a los pasos relativos 11–130.
- Cada lado recibe 60 ms en nivel alto por condición. El retardo positivo desplaza circularmente 40 ms el canal derecho: el inicio contiene el final del ciclo desplazado, no una propagación física de olor desde una fuente.
- Los complementos y los índices anatómicos se aplican correctamente. Las 694 ORN seleccionadas se reparten en 323 izquierdas y 371 derechas. La igualdad de exposición es entre condiciones para cada lado; no prueba igualdad de entradas sinápticas efectivas entre antenas.
- `Continued.step` llama a `stimulus.consume` antes de integrar el siguiente milisegundo. El consumidor copia los objetivos y sincroniza el stream antes del paso. Rechaza salir de los 140 intervalos y su restauración recupera el método anterior.
- El selector altera los objetivos externos ORN; no fija instantáneamente su estado ni cambia pesos, ley neuronal o mando corporal. La sustitución de la modulación entrante ORN está declarada en el propietario del estímulo.

La comprobación instrumental consumió 0,036254 s CPU y un máximo de 40.718.336 bytes RSS. Esta cifra corresponde a ese proceso, no al tiempo agregado de lectura y revisión. Cero pasos nuevos neuronales, corporales o GPU.

## Qué mide PN_q_legacy

La captura lee `q[[189,159]]` después de comprobar la correspondencia de IDs al preparar el modelo. Tanto el `node_ids` congelado como los metadatos anatómicos confirman:

| Posición en la pareja | Fila | bodyId | Tipo | Lado anatómico |
|---|---:|---:|---|---|
| 0 | 189 | 10208 | DM1_lPN | L |
| 1 | 159 | 10176 | DM1_lPN | R |

Es un proxy normalizado de salida de estas dos neuronas. No representa toda la población PN, una medición de calcio ni la totalidad de la salida especializada de PN10208. Transmisión filtrada y conductancia adicional se capturan por separado; no deben mezclarse sus unidades.

## Alcance del contraste temporal

El factorial cancela respuestas bilaterales separables, incluso con memoria unilateral. Para mapas binarios conjuntos **estacionarios**, las cuatro funciones indicadoras de las combinaciones de entrada dan media J exactamente cero; por linealidad, lo mismo vale para cualquier función fija de los dos niveles instantáneos.

La estacionariedad importa. Un ejemplo externo al organismo, `X(t)=exp(-t/60 ms)*L(t)*R(t)`, produce J=0,06348259597153814 en este calendario sin depender de entradas pasadas. Su escala es arbitraria y no se compara con el umbral neuronal. Demuestra que J no separa por sí solo una interacción temporal con memoria de una interacción instantánea cuyo estado o ganancia común cambia durante la ventana.

Un J material justificaría estudiar una interacción bilateral sensible al retardo en este estado y época. No identifica un detector de movimiento, il3LN6, un mecanismo fisiológico concreto ni orientación útil. Esta limitación amplía la cautela sobre transitorios ya presente en la propuesta; no requiere cambiar ventana, criterio o adquisición tras conocer efectos.

## Anotaciones para el análisis final

El campo heredado `fase='cola_OFF'` y la concentración cero del mundo antiguo no describen la entrada terminal de la ronda 50. Para interpretarla deben usarse `TEMPORAL_OWNER.json` y `input_and_observers.npz/nominal_Hz`; no debe presentarse toda la adquisición como ausencia de estímulo.

El lector motor consume el último estado DN comprometido, mientras la captura neuronal corresponde al final del paso actual. Esa latencia de un milisegundo debe explicitarse al comparar señal neural y mando; conservar las ventanas y los dos criterios prospectivos, sin realinearlos para mejorar el efecto.

El giro se observa pero no se aplica; tampoco hay viento en esta ronda. El esquema nuevo `temporal50_scientific_state_v1` distingue el checkpoint con propietario temporal de la cola OFF cualificada en 49. No se atribuye a ese nuevo checkpoint una reanudación ya demostrada.

**Conclusión:** conservar el diseño dentro de estas limitaciones y esperar a las ocho condiciones para el análisis conjunto. No abrir otra simulación ni promover etapas con esta revisión de software.

## Revisión posterior del análisis, todavía sin consultar efectos

Se revisaron `analyze50.py`, `verify50.py`, `test_analysis50.py` y `ANALYSIS_PROTOCOL.json`. Los cinco hashes de este último coinciden. La revisión leyó sólo fuentes y ejecutó controles del evaluador: no abrió trazas ni resultados parciales de ninguna condición 50.

La fórmula I, la diferencia J con factor 1/2, su signo, la ventana Python `[10:130]` (muestras 11–130) y la exigencia conjunta de los umbrales absolutos inclusivos 0,000016 y 0,02 °/s cumplen el contrato. El control añadido coloca valores extremos fuera de ventana y verifica que no afecten la media. Las puertas tampoco admiten aprobar sólo por mando o sólo por señal neuronal. El prefijo compara todos los campos guardados con los diez milisegundos comunes de la referencia 49; el lector se reconstruye con DN del paso anterior. Cuerpo y propiocepción se informan sin convertir su identidad en un criterio nuevo.

**Hallazgo concreto, de prioridad secundaria respecto de J: el verificador DNg no reconstruye toda la ley ni el reloj de las evaluaciones.** `check_dng` comprueba el margen, la igualdad de objetivo/tasa base y final, la derivada, los límites de época y sus estados finales. Sin embargo, tres alteraciones incompatibles con la ecuación o el reloj pasan sus controles:

| Alteración en un registro sintético | Resultado del verificador congelado |
|---|---|
| Cambiar tau de 1 a −1, conservando tasa 1 | Aceptada |
| Margen −1 y ganancia +1, pero objetivo base/final +0,5 y derivada correspondiente | Aceptada; informa objetivo máximo 0,5 |
| Desplazar un segundo el tiempo de evaluación de una etapa RHS | Aceptada |

Son fixtures de registros, no estados obtenidos de la simulación. El segundo caso demuestra que la igualdad entre campos registrados no sustituye la rectificación `max(0,tanh(gain*margin))`. No demuestra que los datos reales tengan estas contradicciones. Tampoco modifica el cálculo de J de DNb05 ni exige repetir la adquisición.

Prueba reproducible y acotada: `python3 -B aporte_motor50/probe_analysis.py`. El script preserva cualquier salida previa; para repetir, usar una copia de este directorio sin el archivo de resultado. [ANALYSIS_PROBE.json](ANALYSIS_PROBE.json) registra 0,067806 s CPU y cero pasos CNS/cuerpo/GPU. Los controles originales del analizador, ejecutados sin `--real`, están en [ANALYSIS_STATIC_CONTROLS.json](ANALYSIS_STATIC_CONTROLS.json), con 0,000744 s CPU.

Recomendación transmitida al coordinador: añadir una comprobación complementaria offline, sin editar las fuentes congeladas ni los criterios. Reutilizar las comprobaciones de `49/verify_operands.py`: tau positiva, tasa FP32 recíproca, transferencia rectificada a partir de margen/ganancia y reloj/fracciones RHS; exigir que rechace estos ejemplos. Las tolerancias ya documentadas para la biblioteca matemática no son ajustes de criterios científicos. No condicionar la cola a una auditoría nueva ni atribuir al diagnóstico más alcance que el realmente verificado.
