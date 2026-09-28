# Decisión tras la campaña 56

**Conservar la pista de duración como PROMETEDOR_NO_CONFIRMADO y mantener abiertas las etapas 4/5.** El contraste sostenido supera el criterio congelado en ambas historias, sin activar DNg100 ni avance. La siguiente decisión debe conectar esta intervención diagnóstica con la señal sensorial natural; no instalar el escritor externo como controlador.

La diferencia terminal perfil−control pasa de -0.0041357056 a -0.36170537 °/s en la historia control, y de -0.0050535968 a -0.40085563 °/s en la historia perfil. Es una reducción del giro positivo basal, no una demostración de giro correcto hacia olor. Los efectos DNb también cumplen el criterio. DNg100 conserva target cero en todos los registros; los ocho mandos medios de avance son cero.

## Qué cambia y qué no queda identificado

Se descarta la interpretación de una frontera PN genérica incapaz de influir materialmente en el giro a esta amplitud. Una salida real fijada durante 128 ms sí lo hace. No se identifica todavía un defecto de persistencia natural: ORN estaba basal y el protocolo prolonga deliberadamente una muestra de la preparación anterior. También crece la exposición acumulada; el cociente de 79–87 no es una ganancia neuronal intrínseca ni una prueba de memoria especializada. La ruta especializada PN→KC conserva su señal original.

No aparece un bug nuevo de normalización. El banco B y Motor coinciden en que la depresión ya existe y conserva historia; su compresión no es por sí sola patología. Tampoco estos resultados justifican bajar theta, cambiar el lector o adoptar I porque produzca avance sin olor. El fallo original de codificación se corrigió sin cambiar ciencia y quedó conservado.

## Tres alternativas y discriminadores siguientes

| Alternativa | Información legal y operación propuesta | Predicción que puede fallar | Control y límite |
|---|---|---|---|
| **A, prioridad funcional: señal natural y orientación** | Usar la transducción espacial ya existente y la ley padre. Observar las 686 terminales realmente consumidas, DNb, DNg y cuerpo mientras el olor depende de la posición. Antes de simular, comprobar si registros existentes bastan; 54 usó olor espacial pero no este testigo completo de las 686 entradas por RHS. | La señal espacial natural produce una diferencia temporal sostenida que distingue fuente izquierda/derecha y alcanza un mando orientador. | Fuentes espejo y basal, misma historia, lectores y geometría. La señal artificial de 56 es un control de transmisión, no un candidato operativo. Si el olor natural ya persiste pero no orienta, la explicación de mera duración pierde prioridad. |
| **B, transferencia temporal con observador correspondiente** | Reutilizar ecuaciones y recursos observados; contrastar una cinética alternativa sólo si una entrada/medición biológica emparejada la restringe. | Predice transitorios y recuperación no usados para elegir parámetros, además de amplitud. | Basal, primera respuesta y exposición comparables; describir qué se iguala realmente. Curvas normalizadas de observables diferentes no constituyen calibración. La falta de raw numérico no impide seguir con pruebas funcionales etiquetadas. |
| **C, contexto descendente y ley local** | Predefinir un conjunto DN por evidencia independiente y cambiar su contexto inicial, manteniendo estado inicial DNg y entrada ORN. Observar los operandos reales que median el cambio. Distinguir contexto de red, estado local y función de respuesta. | El contexto cambia la entrada consumida por DNg100 y su margen; la iniciación sería una hipótesis adicional, no un supuesto. | No fijar toda la entrada DNg durante el contraste primario: anularía la vía causal. Una cinta completa igualada sirve después como control de mediación. No inventar un estado intrínseco como si ya estuviera implementado ni seleccionar el subconjunto por dar positivo. |

La prioridad de medir señal natural **no exige resolver todo el avance antes de evaluar orientación**, ni descarta C: DNg100 es un canal de avance con respaldo funcional, no la definición completa de la etapa 4. Viento y recuperación se evaluarán cuando una respuesta orientadora del organismo justifique ese contraste. El cuerpo actual se conserva; CNS→VNC sigue como interfaz acotada y las seis patas quedan posteriores.

## Contraste con asesores y datos

ChatGPT ASTRA_V2 propone restricciones separadas de cinética, inhibición presináptica y correspondencia de observable. ChatGPT_Motor_V2 prioriza contexto DN y distingue estado local de propiocepción. Ambos corrigieron los controles tras el contraste con el análisis propio. Sus propuestas previas no son una inspección de archivos56; las respuestas y sus límites se conservan. Motor C++/CUDA reconstruyó los ocho brazos con un programa propio, sin importar el verificador autoral: coincidencia exacta de medias, efectos, criterios y presupuesto; 0,3541571 s CPU, sin nuevos pasos CNS. La revisión numérica quedó cerrada en `aporte_motor/REVISION_FINAL.md`.

Los registros biológicos nuevos no se dan por adquiridos: el índice oficial Braun se recuperó, pero Dataverse devolvió 403. Para los artículos olfativos se localizaron curvas pertinentes sin recuperar series originales emparejadas. [Datos y límites](DATOS_PARA_SIGUIENTE_RONDA.md), [fuentes y análisis de identificabilidad](FUENTES_Y_DECISIONES.md). No se necesitan pesos ajustados a mano ni un nuevo escáner universal para formular el siguiente contraste.

## Cierre de esta ronda

Dos instrumentos completados: banco CPU temporal y factorial GPU. No ampliar el piloto para conseguir una admisión retrospectiva. Conservar estados, fuentes, fallos, métricas y criterios; verificar extracción y publicación antes de entregar. La próxima ronda deberá fijar presupuesto propio y como máximo dos prototipos, conservando A/B/C. Este hito aporta una ruta causal de giro más clara, no equivalencia biológica ni una promesa de superar las etapas.

## Respuestas después de transmitir los resultados

Ambos ChatGPT pasan a priorizar A, después transferencia B y contexto C. Lo justifican por la respuesta material a una señal sostenida, sin inferir que la salida natural sea demasiado breve. Estas respuestas se basan en el resumen numérico enviado, no en una auditoría de archivos; se guardan en `ChatGPT_ASTRA_RESULTADOS.json` y `ChatGPT_Motor_RESULTADOS.json`. Motor C++/CUDA conserva C como siguiente discriminador del problema separado de avance. No se decide por votación: el análisis propio prioriza A porque puede medir directamente si la señal natural aprovecha la ruta que 56 acaba de mostrar; C permanece para iniciación y B para restringir la transferencia.

El cociente bruto de efectos no se dividirá mecánicamente por 128 para atribuir una ganancia por dosis: no hay un control de exposición integrada igualada y, después del pulso, la red y el cuerpo evolucionan libremente. No se reetiqueta este resultado como detector de duración, memoria aprendida o sustrato general.

No se lanzaron los ensayos de la siguiente ronda. Primero se comprobará la cobertura de registros previos y se fijará un nuevo contrato finito; hasta dos instrumentos completos, nunca obligación de fabricar tres integraciones para llenar la terna. La simulación de 56 terminó y la GPU quedó libre.
