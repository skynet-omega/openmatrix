# Cierre de iniciación y avance hacia etapas 4/5 — 26 de septiembre

**La campaña45 terminó y el resultado es negativo para iniciar el avance.** El olor sí cambia observables neuronales; las dos vidas producen exactamente las mismas poses, velocidades y fuerzas corporales. Esto no supera navegación ni recuperación de perturbaciones. Conservar el motor numérico y corregir la pregunta experimental es la decisión de esta ronda.

## Entrega visual y datos

- [Comparación narrada](/mnt/f/Videos/Flaywire/ACTUAL_ETAPA45/comparacion.mp4).
- [Control sin olor](/mnt/f/Videos/Flaywire/ACTUAL_ETAPA45/sham.mp4) y [ensayo con olor](/mnt/f/Videos/Flaywire/ACTUAL_ETAPA45/odor.mp4).
- [Figura de datos](/mnt/f/Videos/Flaywire/ACTUAL_ETAPA45/datos.png), [métricas](METRICAS45.json), [suplemento independiente](../iniciacion_sensorimotora_20260926_01/SUPLEMENTO45.json).

Cada vídeo muestra 4 s simulados en 40 s, con narración española, cámara y escalas comunes. El comparativo muestra la diferencia neuronal olor−control; las vistas individuales muestran cambio desde el inicio. Colores identifican sectores y brillo representa magnitud del cambio de salida del modelo, no energía, aprendizaje ni actividad fisiológica calibrada. Se muestran20.342 somas con localización disponible, no todas las166.700 neuronas. Las887 conexiones grises son anatomía, sin flujo inferido.

Se reutilizan poses de1 ms y publicaciones neuronales de100 ms; no se interpolan estados del cerebro. Tres vídeos H.264/AAC1920×1080,24 fps,960 fotogramas cada uno, con decodificación completa verificada. Generación CPU inicial en261,2s; una corrección semántica de la leyenda del control consumió164,8s adicionales sin nueva simulación; cero pasos neuronales/físicos nuevos. Un comando en `instrumentos/neurovideo/paired45.py` produce el conjunto. Se conservaron el vídeo12 s y los históricos.

El protocolo45 usa entrada bilateral uniforme ORN_DM1=0,5 de 1 a 3 s, virtual y sin molécula/concentración física identificada. No tiene fuente puntual ni gradiente espacial: por ello no se dibuja una esfera que sugiera dirección. El apoyo/frenado de la prótesis permanece activo; el pequeño movimiento inicial heredado se asienta antes de la ventana de reposo0,5–1 s. Las MN registradas son observadores; no poseen el torque de las patas en este protocolo.

También queda un [subconjunto portable de observaciones](OBSERVACIONES45.npz), con [procedencia](OBSERVACIONES_PROCEDENCIA.json) y un [recalculador NumPy](recalculate.py). Reproduce las métricas observadas, el lector y el desfase DN→mando; no reinicia el organismo ni reconstruye corrientes internas no guardadas.

## Qué dicen los resultados

Las dos vidas de4 s terminaron en3h14min44 s agregados. Mismos estados iniciales, parámetros, pesos, factores de plasticidad, contexto y prefijo sin estímulo; plasticidad deshabilitada. El suplemento comprueba también el desfase de un intervalo entre entrada muestreada/consumida y salida neural/lector.

| Observable durante1–3 s | Diferencia olor−control | Interpretación |
|---|---:|---|
| ORN media izquierda/derecha | +0,2144 / +0,2171 q | La entrada sí modifica la transducción olfativa del modelo. |
| PN legacy izquierda/derecha | +0,05525 / +0,06548 q | Hay cambio en esas dos salidas; no identifica una mediación exclusiva por ellas. |
| DNb05 izquierda/derecha | +0,0005675 / +0,0002518 q | El efecto alcanza los observables descendentes de giro. Su mando estaba deshabilitado en45. |
| DNg100 izquierda/derecha | 0 / 0 | La señal seleccionada para avance no se modula. |
| Avance crudo y aplicado | 0 mm/s en ambos | El clip no está ocultando una señal positiva. |
| Pose, velocidad y fuerzas | Idénticas entre brazos | No hay iniciación diferencial del movimiento. |

**El olor uniforme también cambia la señal de giro sin aportar una dirección espacial.** El mando neural calculado pero no aplicado cambió, frente al control, una media de+0,393°/s durante1–3 s (integral+0,785°); pasó de−0,482°/s en control a−0,089°/s con olor. Esto no demuestra giro correcto ni una ruta no lineal: puede reflejar asimetría de conectividad, parámetros o estado. Es otro motivo para comparar virtual−common con la misma intensidad media, en vez de atribuir cualquier cambio de giro a información lateral. [Cálculo](UNIFORME_GIRO.json).

Estas unidades q son liberaciones normalizadas del modelo. Los cambios respecto del control no equivalen a cambios respecto del basal: DNb05 puede quedar por debajo de su basal y, aun así, por encima del control. Los vídeos conservan esa distinción. Un solo par determinista no estima variabilidad entre moscas/semillas ni demuestra validez biológica general.

## Localización inicial de la ausencia de avance

[Sonda del preparado](PREPARED_DN.json), sin integración ni evaluación del coeficiente: se restauró el operador guardado, comprobando IDs anatómicos→filas→`release()` frente a las salidas iniciales45. La auditoría se limita a este punto de partida y al alcance de los propietarios inspeccionados. Las filas DNg100 son36/46 y los IDs10045/10056. DNb05 usa10118/10065. La señal de avance está basada en `release()`, no en la publicación FP32; no se atribuye el negativo a un redondeo de visualización.

En ese preparado las filas no pertenecen a las cachés inspeccionadas de sustitución de coeficientes ni reciben posiciones de los escritores dinámicos APL/PN-general. La ley genérica es:

`tau * dq/dt = max(0, tanh(gain * (sum_j W_ij * transmission_j * cap_j + drive_i − theta_i))) − q`.

| DNg100 | Suma excitadora | Suma inhibidora | Entrada neta | Umbral | Preactivación |
|---|---:|---:|---:|---:|---:|
| Izquierda | 6933,2 | −7681,5 | −748,3 | 663,3 | −1411,5 |
| Derecha | 7058,6 | −7804,6 | −746,0 | 559,2 | −1305,2 |

Son unidades internas ponderadas, no corriente fisiológica medida. La entrada directa es0; el objetivo de ambas filas es0. La diferencia no es un decimal dudoso cerca del umbral. Incluso eliminar únicamente el umbral positivo conservaría una entrada neta negativa en este instante. **Esto orienta hacia el balance de entradas y la ley neuronal, no autoriza ajustar umbrales para obtener marcha.** PVLP137 aporta entre las mayores contribuciones excitadoras y GNG127 entre las inhibidoras en el preparado; no se deduce de ello una causa temporal durante el olor.

Límite decisivo: no se guardaron los operandos/objetivos de DNg100 durante el estímulo45. Esta extracción caracteriza el preparado de inicio; no reconstruye ficticiamente las entradas de1–3 s a partir de publicaciones cada100 ms. La evaluación CPU de la suma describe la ley/operador preparado, no certifica identidad bit a bit con cada reducción FP32 durante la vida.

El primer lanzamiento de la sonda falló por una ruta de importación omitida y el segundo por tratar la propiocepción guardada como array en vez de su campo `drive`. Ambos terminaron antes de integrar; logs conservados. La corrección usa el mismo campo que `CyborgVisualSession.step`, completó en23,9s y no cambió estado ni reloj. No hubo barrido de parámetros.

## Qué rescatamos de septiembre y qué evitamos repetir

Esta es una revisión dirigida por el [índice complementario](../../biblioteca_revision_20260923/ejecucion_02/INDICE_COMPLEMENTARIO.md) y por resultados posteriores; no un censo nuevo de todos los archivos de septiembre.

| Antecedente | Qué cambia en la decisión actual |
|---|---|
| [DNg100 directo,13-09](/home/daroch/AXIOMA_FLYWIRE/matrix/reports/DNG100_RECLUTAMIENTO_SIN_APOYO_20260913.md) | Ya hubo reclutamiento neural/MN sin apoyo ni paso completo. Repetirlo no demostraría una ruta olfativa nueva. |
| [Interfaz37](../../campanas/etapa4_motor_interface_20260925_37/README.md) | Avance tónico y canales propulsivos sin contribución ya eran limitaciones estructurales; no basta aumentar ganancia del lector. |
| [Ablaciones41](../../campanas/etapa45_postwind_diagnosis_20260925_41/README.md) y [señal42](../../campanas/etapa45_signal_chain_20260925_42/README.md) | El mando filtrado/relay puede perjudicar la trayectoria; sustituirlo por señal cruda mejoró un replay físico sin recuperar el rumbo. No son ensayos nuevos de feedback neural. |
| [Cintas43](../../campanas/etapa45_orientation_observability_20260925_43/README.md) | El yaw virtual también cambia intensidad común. Ya existe un control que iguala la media conservando la asimetría basal. |
| [STOP44](../../campanas/etapa45_neural_contrast_20260925_44/README.md) | Sólo se ejecutó sham120 ms. Las respuestas common/virtual siguen siendo una pregunta real pendiente, no otro experimento repetido. |
| [Análisis12 s](../iniciacion_sensorimotora_20260926_01/METRICAS.json) | Avance impuesto0,2mm/s; DNg100sin delta. Distancia al foco1,131→0,138→1,192mm en0/6/12 s: acercarse y después alejarse no prueba persecución. Integral de yaw crudo−4,13° frente a aplicado−48,68°; la interfaz domina el giro. |
| [Campaña45](../../campanas/iniciacion_olfativa_20260926_45/RESULTADOS.md) | Retirar el avance impuesto deja claro que el olor uniforme no inicia movimiento por el lector actual. |

**La geometría explica por qué acercar el centro del cuerpo no equivale a oler más.** En la vida de12 s, las antenas están aproximadamente1 mm delante del centro registrado. El olor medio consumido alcanza su máximo en1,026 s y las antenas su menor distancia a la fuente alrededor de1,024–1,025 s. El centro del cuerpo pasa más cerca recién en5,879 s. A6 s el cuerpo está a0,138 mm, pero las antenas están a1,025–1,050 mm. Por tanto, la caída posterior de olor no contradice el campo gaussiano; medir sólo distancia corporal resulta insuficiente para interpretar búsqueda. En el próximo diseño se fijarán y reportarán por separado geometría sensorial, intensidad, rumbo y posición corporal antes de correr. [Cálculo desde la traza](GEOMETRIA_12S.json).

No se mezclan los cuerpos/protocolos anteriores como si fueran una sola réplica; el FAIL histórico de paridad de mando a2s permanece intacto.

## Propuesta y decisión ejecutiva

Tres explicaciones separadas: **A**, la entrada/ley de DNg100 no produce modulación propulsiva; **B**, el lector o su interfaz no representa adecuadamente una acción útil; **C**, la mecánica impide ejecutar una señal efectiva.45 descarta el caso particular deB«clip esconde avance positivo», pero no todaB; no pruebaCporque nunca envía avance no nulo. La sonda preparada acotaAal inicio y confirma la identidad del lector sin identificar la causa temporal del olor.

**Experimento adicional completado: contraste neural lateral pendiente.** Se registró una nueva [ronda46](../../campanas/etapa45_contraste_pendiente_20260926_46/PLAN.json), separada del STOP44, con dos ramas120 ms:20 ms de prefijo común y100 ms de contraste. Se conserva motor original, restauración, cinta corporal, reglas y umbrales científicos44. No se traslada sin verificar el sham a otro motor. Presupuesto nuevo fijado antes: máximo240 ms neuronales, dos procesos,900 s de pared por rama/1800 s agregados,8000 s CPU,18 GiB RAM; sin reintentos ni ajuste de parámetros. CPU basada en3169 s medidos para sham120 ms, por lo que el tope6000 s sugerido inicialmente por ChatGPT sería demasiado ajustado para dos ramas.

La comparación exige entradas realmente consumidas, prefijo/estado corporal y fuerzas subpaso emparejados, RNG/retina/propiocepción iguales. Materialidad histórica conservada: integral absoluta de diferencia de yaw crudo≥0,001° **o** media de últimos50ms≥0,02°/s. Es sensibilidad descendente; no se convierte ese umbral en prueba de giro correcto. Una cinta impone el movimiento corporal durante este diagnóstico y la nueva señal neural no se entrega al cuerpo: no representa arranque autónomo ni feedback cerrado.

El criterio previo exigía detener la ronda ante una discrepancia de contexto/prefijo, un fallo de ejecución o de presupuesto. Un negativo válido termina este contraste de100 ms; no se amplía duración, amplitud o ganancia buscando un positivo. [Estado de ejecución46](../../campanas/etapa45_contraste_pendiente_20260926_46/QUEUE.json).

## Resultado46 y actualización de la decisión

La pareja terminó con todos los controles de contexto/entrada/física satisfechos. **Hay sensibilidad neural al contraste lateral, pero el efecto motor calculado no alcanza la materialidad registrada:** integral absoluta 0.0000165° frente a0,001°; magnitud de media final 0.0002869°/s frente a0,02°/s. Brechas aproximadas61× y70×. El relé calculado sigue idéntico. No es un desacuerdo de decimales al borde de un umbral ni una señal completamente perdida.

Consumo994,4s de pared,6521,2sCPU neuronales; presupuestos1800/8000s respetados. No hay simulación pendiente. [Cierre46](../../campanas/etapa45_contraste_pendiente_20260926_46/README.md). El sham de44se reutilizó como comprobación de montaje, no como nueva réplica; el efecto estimado es virtual−common.

**Cambio en la prioridad:** se cierra la alternativa de que quedaba sólo por ejecutar ese contraste de100ms antes de una vida larga. Los datos no justifican repetir12s, aumentar la ganancia del relé o activar plasticidad. La próxima ronda debe medir directamente entrada/objetivo de las filas motoras bajo estímulo, y comprobar por separado la transferencia física a comandos no nulos como calibración técnica. Esta instrumentación distingue falta de entrada de una ley que mantiene objetivo0; la calibración no se presenta como arranque espontáneo. Antes de cualquier cambio neuronal se requiere una hipótesis de receptor/ley o un fallo de código identificado con evidencia independiente. La evidencia actual no elige entre una reparación del reclutamiento y un lector funcional distinto.

## Camino concreto para admitir etapas4/5

1. **Iniciación:** exigir estímulo externo→modulación de una señal propulsiva válida→desplazamiento atribuible, frente a control, sin offset de avance.45 es negativo. Si el bloqueo persiste, el siguiente cambio debe atacar un mecanismo o un error de implementación identificado, con datos independientes de fisiología/función; no seleccionar retrospectivamente una neurona sólo porque aquí responda más. Mantener plasticidad apagada mientras se identifica esta transferencia aguda.
2. **Etapa4:** una vez disponibles avance y giro útiles, comparar lazo sensorial vivo con cinta de entrada bajo condiciones iguales, en un horizonte suficiente para mover el cuerpo. Medir rumbo/distancia sólo en el evaluador. Exigir ventaja del feedback y del contraste lateral, además de integridad de la trayectoria. El olor uniforme45 no puede evaluar dirección.
3. **Etapa5:** perturbación física común, seguida de recuperación libre online frente a control de feedback. Exigir corrección adicional respecto del retorno/frenado pasivo. Magnitud y duración deben derivarse de la autoridad de giro realmente medida, no de30° arbitrarios. Como ruta alternativa ya formulada, mover la fuente prueba adaptación sensorial, pero no sustituye una perturbación corporal.

El horizonte orientativo es4–6 s para un primer ensayo funcional una vez identificada la transferencia, con 1s de basal y varios segundos de respuesta. No es una constante neurocientífica ni una autorización para saltar los discriminadores: si la autoridad medida requiere más tiempo, se calcula y registra antes. Doce segundos siguen siendo exploración hasta que exista un control causal adecuado. Las primeras admisiones pueden limitarse explícitamente a la prótesis corporal; la marcha natural de seis patas exige otra validación.

## Contraste con asesores y literatura

[ChatGPT](CHATGPT.md) leyó código público45/43/44, pero recibió de Codex los resultados finales45; no accedió a los NPZ locales ni ejecutó la vida. Coincide en la extracción DNg100 y en completar el contraste lateral. Corrige la inferencia excesiva «falta excitabilidad» y recuerda que MN shadow no acciona patas. [Jev](jev_review/response.json), una petición1375 tokens de entrada/188 de salida, prioriza inspección de entrada efectiva entre alternativas explícitas. Sus probabilidades son clasificación del asesor, no incertidumbre científica. No se usaron subagentes Codex.

La identidad DNg100/BDN2 está documentada por el [catálogo MaleCNS](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/DNg100_R.html). El trabajo de [Sapkal y colaboradores](https://www.nature.com/articles/s41586-024-07854-7) distingue supresión de órdenes de marcha y frenado activo; respalda separar contexto neural y ejecución, sin demostrar que el modelo actual reproduzca ambos mecanismos. [Tao y colaboradores](https://www.nature.com/articles/s41467-023-42613-8) estudian cómo la actividad olfativa modula locomoción: identidad/intensidad no equivale a conocer directamente la posición de la fuente. Estas fuentes fundamentan controles y límites; no se importan circuitos por analogía ni se activa plasticidad para compensar una ruta aún no identificada.
