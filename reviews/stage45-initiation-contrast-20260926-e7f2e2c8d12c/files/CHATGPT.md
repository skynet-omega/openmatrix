# Revisión externa de ChatGPT · 26-09-2026

Conversación autorizada: 6ab66613-ea8c-83e9-b7eb-7972510f8729. Respuesta recibida por la app. El revisor declara acceso a código público de protocolo45 y 43/44, no a los NPZ finales ni a las rutas locales; resultados cuantitativos suministrados por Codex. No se verificó el modo de razonamiento en esta llamada.

## Dictamen

**Campaña 45 permite cerrar un resultado negativo de iniciación olfativa en esta configuración, no las etapas 4/5.** Mi prioridad sería localizar el cero de DNg100 en su operador efectivo; después, recuperar el contraste direccional pendiente, sin repetir la activación directa ni modificar umbrales.

No accedí a las rutas locales de `SUPLEMENTO45.md` y del informe final. Sí encontré y leí una **copia pública de `protocol45.py`**, además de los planes y el código de 43/44. Los resultados finales de las dos vidas de 4 s proceden de tu mensaje. **No ejecuté simulaciones.**

## 1. La inferencia más débil

La inferencia que evitaría es:

> «DNg100 no cambia; por tanto, le falta excitabilidad y hay que aumentarla».

El resultado identifica ausencia de modulación en **la señal seleccionada para propulsión**. No distingue todavía entrada efectiva insuficiente, ley de transferencia, coordenada equivocada, señal publicada distinta de la señal causal o una elección funcional insuficiente del lector.

Tu exclusión del clipping es correcta: con delta bilateral y avance **crudo** iguales a cero, el clip no está ocultando propulsión positiva. Pero eso no excluye toda la hipótesis B.

Hay además una precisión importante confirmada por código: `NeuralPropulsion.audit()` declara **músculos nativos en shadow, sin propiedad sobre el torque físico**, y apoyo/frenado mediante la prótesis heredada. `advance()` entrega `requested=[forward,0.]`. Por tanto, **MN shadow distinta con fuerzas idénticas no demuestra fallo mecánico: esas MN no son el actuador de este ensayo**. Tampoco un futuro desplazamiento mediante esa prótesis demostraría marcha nativa de seis patas. 

Los cambios ORN/PN/DNb05 respaldan una respuesta al estímulo en esos observables; no identifican por sí solos una cadena de mediación exclusiva ORN→PN legacy→DNb05→locomoción.

## 2. Discriminador primero: auditoría causal de DNg100, sin nueva vida

**Una extracción acotada sobre el preparado y los estados guardados disponibles**, no otra revisión de todo el repositorio.

La copia pública exige los identificadores `[10045,10056,10118,10065]`; calcula avance con la media de las dos primeras deltas y comprueba que el lector consume la publicación del intervalo anterior. Esas guardas protegen orden y latencia, pero no demuestran independientemente que el índice usado corresponda a la coordenada causal correcta. 

Seguiría exactamente:

**ID anatómico → índice del operador → estado dinámico → `release()` → `last_dn` → basal fijo → avance crudo → comando entregado.**

Para las dos filas DNg100, extraería entrada excitadora/inhibidora efectiva, preactivación, objetivo y tasa —o sus equivalentes en la ley realmente ejecutada—, máscaras y reemplazos aplicables. **No usaría como entrada efectiva un CSR leído después de que se restauraron los pesos temporales PN/APL.** La reconstrucción debe respetar esos propietarios y hacerse en almacenamiento independiente, sin alterar el organismo.

La comparación entre preparado y estado estimulado tiene este poder discriminante:

| Hallazgo | Interpretación permitida |
|---|---|
| Cambia el estado causal, pero no la señal que lee el motor | Problema de lectura/publicación que debe localizarse y corregirse. |
| Cambia la entrada efectiva, pero la ley mantiene el objetivo o salida invariable | Limitación localizada de la respuesta del modelo en esa condición; no autorización para ajustar excitabilidad. |
| No cambia la entrada efectiva de DNg100 | La propagación observada en otras neuronas no alcanza funcionalmente esa entrada; no culpar todavía a su umbral. |
| Lectura y operador son coherentes, y DNg100 sigue sin modularse | Conservar el negativo de esa ruta; no prolongar el mismo estímulo esperando que arranque. |

**Límite:** el operador preparado sólo informa del preparado. Si no están guardados los operandos suficientes durante el estímulo, la causa temporal queda no identificada; no debe reconstruirse ficticiamente desde las medias de la traza.

No repetiría la activación directa de septiembre. Su antecedente sirve para revisar qué salida y qué propietario físico se probaron entonces, no como demostración de que la mecánica actual responde a una orden efectiva.

## 3. Segundo discriminador: completar el contraste direccional 43/44

Lo elegiría **antes que otra pareja larga de olor uniforme**, siempre que la auditoría anterior no descubra una discrepancia de implementación que invalide el montaje.

El plan público conserva una intervención bien definida. Para cada muestra:

\[
\delta_m=\operatorname{media}(L_Y,R_Y)
-\operatorname{media}(L_0,R_0)
\]

Se comparan **virtual** \((L_Y,R_Y)\) y **control** \((L_0+\delta_m,R_0+\delta_m)\). **No reemplazaría el control por \((m,m)\): eso también eliminaría la asimetría basal y cambiaría la pregunta registrada.** El plan exige dominio válido, media emparejada y contraste lateral verificable. 

Dos condiciones de código son decisivas:

**Latencia:** `TapeBoundary.sample()` prepara la entrada del siguiente intervalo mediante `next_input=dt//1_000_000+1`. La comparación debe hacerse sobre la cinta **consumida**, no sólo sobre la concentración recién publicada.

**Movimiento controlado:** `FrozenMotor` entrega los comandos del donante, no los nuevos comandos neuronales. Incluso conserva un cálculo histórico de avance con offset `0.2`. Es un control válido para aislar sensibilidad neural, **pero no debe trasladarse al ensayo 45 y presentarse como propulsión espontánea**. 

### Presupuesto y decisión

Propondría una autorización nueva, separada del bloqueo anterior: **dos ramas de 120 ms**, con prefijo común de 20 ms y contraste de 100 ms; máximo **900 s de pared por rama, 1.800 s agregados y 6.000 s CPU agregados**, sin reintentos. Son topes propuestos, no una previsión de coste. El STOP44 por exceso de CPU permanece intacto. 

Reutilizaría el sham44 únicamente con el mismo motor, fuentes, preparado y contexto que lo calificaron. Si eso no puede sostenerse, **no lanzaría la pareja fingiendo que el control se transfiere**.

Conservaría los criterios ya registrados: diferencias por eslabón, integral absoluta del cambio de yaw crudo y media de los últimos 50 ms. El plan44 fijó materialidad en **0,001° de integral o 0,02°/s de media**, sin imponer un signo obligatorio. No los reinterpretaría después como prueba de giro correcto. 

**Detenerse** ante fallo de contexto, cinta, presupuesto o ausencia de efecto material descendente. **Seguir** sólo si aparece sensibilidad lateral material, con su signo y latencia documentados. El resultado favorable justificaría estudiar control cerrado; no cerraría ninguna etapa por sí mismo.

## 4. Qué exige cada etapa

| Objetivo | Evidencia necesaria |
|---|---|
| **Arranque** | Cambio sensorial seguido de señal propulsiva efectiva y respuesta física atribuible a ella, sin avance impuesto. Campaña45 es negativa. |
| **Etapa4: navegación** | Movimiento generado por el organismo y decisiones direccionales útiles bajo sus propias entradas sensoriales. Distancia y posición de fuente pueden usarse para evaluar, nunca para mandar. Un giro o una respuesta bilateral no bastan. |
| **Etapa5: corrección** | Tras una perturbación física definida, reducción del error atribuible al feedback activo frente a una condición que no actualice la corrección. Debe distinguirse recuperación activa de frenado o retorno mecánico pasivo. |

El yaw virtual de 2° modifica una **entrada sensorial contrafactual**; no perturba mecánicamente el cuerpo. Con movimiento reproducido desde una cinta no puede demostrar corrección en lazo cerrado. Asimismo, yaw aplicado cero en45 excluye evaluar su capacidad de steering por esa salida.  

## Vídeos fieles

Renderizaría **las poses registradas**, sincronizadas por reloj, con cámara, escala y velocidad iguales entre brazos. Mostraría estímulo virtual, DNg100, MN shadow y comando realmente aplicado en paneles separados. Nada de animar patas desde MN shadow ni generar movimiento “explicativo”.

Si qpos es idéntico, **la animación corporal debe ser idéntica**: la diferencia está en los paneles neuronales. Eso comunica el resultado negativo sin ocultarlo.

**Decisión final:** conservar45 como evidencia de propagación sin iniciación; auditar la ruta efectiva de DNg100; y, si el montaje queda coherente, completar únicamente el contraste lateral pendiente. No cambiaría hoy el lector por una neurona que “responda más”, ni añadiría locomoción impuesta para declarar superadas las etapas4/5.
