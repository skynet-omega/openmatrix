**Sí: primero transportaría la rama exacta sin reescalado y, una vez conservada su dinámica, distinguiría estas tres hipótesis.** Son explicaciones rivales del fallo observado —pueden coexistir—, no mecanismos ya demostrados ni propuestas de recalibración. **No he revisado el adaptador nuevo ni ejecutado estas pruebas.**

La prioridad es registrar **la señal que realmente cruza cada frontera**, no solamente la actividad de una neurona o el giro final.

## Registro común para las cuatro condiciones

| Frontera | Señales prioritarias |
|---|---|
| **Mundo → antenas** | Concentración recibida por cada antena, posición/orientación de sus sensores, salida del transductor y momento efectivo de consumo. |
| **ORN → PN → cerebro central** | Eventos y liberación presináptica; conductancias receptoras; voltaje y estado adaptativo; **salidas locales efectivas de las PN**, separadas de variables diagnósticas heredadas. |
| **Circuitos centrales → DN → lector** | Entradas excitatorias/inhibitorias por rutas identificadas, voltaje, adaptación y salida transmitida de cada DN; variable efectivamente leída, línea base, filtrado, saturación y comando resultante. |
| **Actuación → cuerpo → sentidos** | Mando solicitado y aplicado, fuerzas/torques y contactos, velocidad angular/lineal, postura y retorno propioceptivo/visual. |

Mantendría un reloj común y la distinción **producido → entregado → consumido**. En señales graduadas, cero espigas no sustituye medir la variable que consume el siguiente componente.

Para cada señal comparable, calcularía por separado el efecto respecto de su sham, la diferencia izquierda–derecha y la respuesta uniforme. **Una componente común no es automáticamente ruido.** Tampoco un promedio bilateral cercano a cero demuestra pérdida de información: hay que conservar los vectores y la estructura temporal. El uniforme controla presencia de olor, pero no necesariamente dosis total; su intensidad debe quedar declarada, sin normalización retrospectiva.

## H1. El contraste espacial se degrada antes de llegar a las salidas olfativas centrales

**Hipótesis:** las antenas reciben exposiciones diferentes, pero la transducción o la transmisión ORN–PN reduce, invierte o mezcla esa diferencia hasta hacerla poco utilizable aguas abajo. La lateralización puede depender de la transmisión sináptica, no solamente de dónde aparece un axón; existe evidencia experimental de liberación asimétrica en la vía olfativa de Drosophila. :chatgpt-content-reference{index="0"}

**Firma que buscaría:** izquierda y derecha son distintas en la entrada realmente consumida, pero dejan de distinguirse en alguna frontera posterior. Localizar **la primera frontera**, usando las salidas efectivas, no solo voltajes somáticos. La comparación debe respetar lateralidad y proyecciones verificadas, sin declarar homólogos dos grupos únicamente por `somaSide`.

**Intervención discriminante:** intercambiar las historias de exposición entre antenas, manteniendo amplitudes, cronología, parámetros y estado inicial. No intercambiar pesos ni imponer simetría interna. El resultado pregunta si la respuesta sigue al estímulo intercambiado o queda fijada al mismo lado del sistema.

**Qué la debilita:** que las salidas PN entregadas al cerebro central conserven respuestas claramente distinguibles y coherentes con el intercambio, mientras el desacople aparece después. Eso no valida toda la periferia, pero reduce su prioridad como explicación principal.

**Límite:** una ausencia detectada únicamente en el promedio de una familia no basta para concluir que el circuito perdió el contraste.

## H2. El contraste llega, pero el estado basal o la dinámica central impiden convertirlo en una salida descendente adecuada

**Hipótesis:** la información olfativa alcanza el cerebro central, pero adaptación, estado recurrente, recursos sinápticos o nivel basal modifican su transformación hacia las DN. No presupone que falte facilitación ni que haya «demasiada inhibición».

Esta separación es biológicamente pertinente: se han observado respuestas de DNa02 a olores lateralizados incluso durante inmovilidad, junto con un desplazamiento de su nivel de actividad asociado al estado locomotor. **Respuesta sensorial y expresión motora no son equivalentes.** :chatgpt-content-reference{index="1"}

**Firma que buscaría:** contraste PN presente, pero respuestas DN dominadas por una componente común, saturación, adaptación sostenida o un retraso dependiente de la historia. Registrar conductancia y corriente por separado: una conductancia inhibitoria elevada no identifica por sí sola la corriente neta ni su efecto sobre el voltaje.

**Intervención discriminante:** preparación original frente a una prolongación **prefijada** en aire limpio, conservando la evolución completa de voltajes, compuertas, calcio y recursos; después, el mismo protocolo de segundos. No reiniciar selectivamente estados ni escoger el instante que produce mejor giro.

Para atribuir la diferencia específicamente al estado neuronal, usaría en el discriminador **las mismas historias aferentes registradas** durante el ensayo. Si cambian también retina, propiocepción o exposición por movimiento corporal, el resultado identifica dependencia del estado del **organismo**, no exclusivamente del CNS.

**Qué la respalda:** con entrada comparable, cambia de manera reproducible la transformación central→DN y ese cambio corresponde a estados dinámicos registrados.

**Qué la debilita:** que las DN ya produzcan una salida direccional consistente en ambas preparaciones y el fallo aparezca entre esa salida y la actuación.

El antecedente **PFG→hDeltaK que comunicas** justifica medir transmisión hacia las siguientes fronteras; un cambio de actividad sin orientación no identifica todavía el mecanismo limitante.

## H3. La señal descendente existe, pero su lectura o expresión mecánica no producen el giro

**Hipótesis:** el problema está después de la generación de una señal descendente utilizable: lector, línea base, transformación a actuadores, restricciones de contacto o estado motor. No presupone que la prótesis sea inválida.

Las mediciones experimentales relacionan diferencias bilaterales de actividad descendente con orientación, pero también muestran respuestas neuronales durante inmovilidad sin consecuencias motoras comparables. Por eso no impondría una equivalencia universal «DN activa = cuerpo gira». :chatgpt-content-reference{index="2"}

**Firma que buscaría:** separar tres posibilidades: la señal DN cambia y el lector no; el comando cambia pero no la actuación efectiva; o aparecen fuerzas/torques pero el cuerpo no responde como exige su contrato mecánico. Comprobar primero unidades, signos y latencias, no aumentar ganancias.

**Intervención discriminante:** reproducir una historia descendente **realmente registrada** en la entrada del lector, desde el mismo estado corporal y del controlador. Cuando sea necesario localizar otra frontera, reproducir su comando registrado directamente en la interfaz de actuación. Son pruebas diagnósticas de reproducción, no soluciones para instalar.

**Qué la respalda:** una pérdida o transformación inconsistente localizada entre dos de esas fronteras.

**Qué la debilita:** que el lector y el cuerpo respondan coherentemente a la historia aplicada y que la señal descendente original carezca ya del contraste requerido.

**Límite esencial:** si el mando aplicado es prácticamente nulo, el resultado no demuestra incapacidad mecánica. Debe informarse la magnitud real de la intervención, sin amplificarla hasta obtener éxito.

## Orden de decisión

**Ejecutaría primero izquierda/derecha/uniforme/sham nativos durante segundos, con este registro común.** La primera frontera donde aparezca el desacople determina cuál discriminador merece ejecutarse; no abriría tres baterías simultáneas.

El transporte comprueba que se conserva el organismo anterior. Los discriminadores comprueban mecanismos dentro de ese modelo. **Ninguno convierte el reescalado retirado en una ley, una reproducción externa en navegación autónoma ni un giro correcto en aprobación de etapa 3.**
