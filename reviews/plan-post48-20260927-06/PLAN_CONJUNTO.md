# Plan conjunto después de la campaña 48

27 de septiembre de 2026. **Cambiar la prioridad inmediata: de repartir más olor a identificar la transformación que impide una orden útil.** La campaña 48 queda negativa; las etapas 4/5 siguen abiertas. El lector complementario del chat Motor C++/CUDA coincide con el análisis canónico sobre los mismos registros: no es otra adquisición ni demuestra que toda la implementación sea correcta o que el modelo biológico sea suficiente.

Se contrastaron propuestas de Matrix Astra, el chat existente «Motor C++/CUDA», ChatGPT ASTRA_V2, ChatGPT Motor_V2 y una clasificación acotada de Jev. No se crearon subagentes Codex. Las propuestas previas y las respuestas se conservan; no fueron ciegas respecto del historial compartido. El acuerdo entre asesores no sustituye medidas. PRO no se verificó. Jev priorizó opciones explícitas; no navegó ni revisó código.

## Qué cambió con esta revisión

1. **Ya ejecutamos el primer diagnóstico barato.** En ocho bloques guardados de 48 —901–1000 y 2901–3000 ms de los cuatro brazos— reconstruimos exactamente el margen, el objetivo/tasa finalmente consumidos y la derivada de DNg100. No apareció una contradicción en esas operaciones. El objetivo es cero; el estado conserva residuos subnormales, que una media puede redondear a cero. [Cálculo y límites](DIAGNOSTICO.md).
2. **Bajar solamente el umbral no explica el rescate que se esperaba.** Con las entradas registradas, `net + drive` permanece negativo incluso al evaluar algebraicamente theta=0. Cualquier umbral no negativo sigue dando objetivo cero con la ley vigente. Es un contrafactual sobre entradas fijas, no una nueva trayectoria ni una prueba de inhibición biológica excesiva.
3. **No tenemos todos los aportes presinápticos por etapa.** Hay saldos positivos/negativos y un vector publicado cada 100 ms; eso no permite reconstruir retroactivamente cada contribución recurrente. El siguiente instrumento debe declarar esa frontera. Un producto de caminos `W^k·ΔPN` sirve para priorizar anatomía, no para demostrar transferencia causal.
4. **Los rescates más obvios ya tienen negativos.** Activación directa DNg100/DNg97 reclutó MN sin marcha; P9 y retirada AOTU019→DNa02 fallaron en otra referencia LIF; normalización global y preparación prolongada tampoco resolvieron orientación. Son antecedentes de otros modelos/preparaciones, no controles actuales de 48. [Revisión del motor](aporte_motor/APORTE.md).

## Tres explicaciones rivales vigentes

| Hipótesis | Operación que se examina e información legal | Discriminador y falsador |
|---|---|---|
| **A. Ruta y balance de entradas** | Puertos realmente liberados, pesos efectivos, identidad/NT y estado de los aferentes de DNg100 y DNb05; DNa02 como contraste adicional. No usar posición de la fuente ni dirección correcta como señal neural. | Separar contribuciones excitatorias/inhibitorias y poblaciones en estados completos disponibles. Una ruta anatómica débil sólo prioriza; la explicación causal requiere una perturbación localizada con respuesta firmada aguas abajo. Se rechaza si el nodo intervenido responde como se predijo y el siguiente observable no lo hace. |
| **B. Ley neuronal o estado heredado** | Identificar la transformación exacta que rectifica/suprime la entrada, sus parámetros y procedencia; preservar filtros, recursos y recurrencia. Cambios sólo restringidos por evidencia independiente. | Una ley/estado alternativo debe anticipar una respuesta bajo una condición que no se usó para escogerlo, con control de igual intervención/coste. Aumentar actividad o mejorar también el sham no demuestra la explicación. Sin correspondencia entre datos y entrada del modelo, el parámetro queda no identificable. |
| **C. Implementación o interfaz** | Reconstrucción del operador, unidades, IDs, escritor final y consumidor, usando estados y fuentes congeladas. | Exigir una discrepancia reproducible y un contraejemplo mínimo. La contabilidad ejecutada descarta fallos concretos del margen/objetivo en las ventanas observadas; no certifica todo el circuito. Si aparece un bug, reparar y repetir sólo los brazos afectados, conservando el negativo original. |

Estas hipótesis se distinguen de las rutas estratégicas: **orientación con el cuerpo actual primero; interfaz CNS→VNC→MN acotada después; músculos y seis patas más adelante**. Las etapas son instrumentos del objetivo de inteligencia causal, memoria y adaptación; no su definición completa.

## Qué reutilizar del laboratorio y del predictor

| Instrumento existente | Uso ahora | Límite que conserva |
|---|---|---|
| Laboratorio / `matrix_workbench` y observadores | Una consulta explícita de transferencia y estado. Trabajar en ASTRA con fuentes identificadas y resultados propios. | `laboratorio.py` de ASTRA llama al laboratorio histórico: no ejecutar su promoción ni escribir en FLYWIRE. Un estado COMPLETE no admite una ley biológica. |
| Operador efectivo y radar de discrepancias | Reutilizar la descomposición aditiva, conservar contexto restante y mostrar información ausente. | Ordena discrepancias observadas; no identifica por sí solo causas ni estima error sin referencia. La contabilidad de esta ronda usa datos existentes, no un predictor nuevo. |
| `LiveForecast125` | Donante del método: contexto completo, misma historia, comparación con referencia. Evaluar portabilidad sólo si una consulta concreta necesita pronóstico. | Exige otro padre exacto y perfil de 15,625 µs. Su aceleración histórica 5,286× en 120 ms no se transfiere al motor 13 ni al modelo biológico. No instalarlo sobre 48 por el nombre. |
| Predictor fisiológico ORN mecanismo+MLP | Conservar máscaras de calidad y los fallos como lecciones de adquisición. | Transferencia/amplitud rechazadas; ganancias ORN no calibran DNg100. Repararlo no es un requisito global para estudiar orientación. |
| Predictor de lectura DNa02 | Reutilizar observador y datos originales. | El FIR de un animal mejoró RMSE 6,816%, bajo el 10% fijado: sigue rechazado. La reserva ya está expuesta. No trasplantar su kernel a DNb05. |

No construir ahora otro simulador Shiu/Pugliese ni un generador molecular general. Esos trabajos sirven como contrastes de método y de dominio. Mejorar el predictor sólo cuando se haya localizado una limitación que cambie la elección del experimento.

## Datos que cambian decisiones

Se revisaron catálogo histórico, biblioteca, índice complementario y originales pertinentes. Los dos MAT DNa02 locales se verificaron por tamaño, SHA256 y estructura: el bilateral contiene voltajes/conducta de 575 s; el otro contiene 1.900 s y un canal de pulsos sin etiquetas de antena por ensayo. [Comprobación actual](LOCAL_MAT_CHECK.json). No se reabrieron reservas ni se reajustó un predictor.

- **Rayshubskiy, eLife 102230:** contraste de DNa02 ante entrada lateral, incluso durante inmovilidad. Los datos locales sirven para observación; la falta de lado por pulso impide un contraste direccional etiquetado en el segundo MAT. No asignar izquierda/derecha según orden ni inferir una ley sináptica de un registro de voltaje/conducta.
- **Yang, Cell 2024:** respaldo funcional de DNb05 y giro durante marcha. El código 2P ya está local; los registros crudos no están en ese ZIP y el artículo los ofrece por solicitud. Esto corrige la primera propuesta de ChatGPT de tratar DNb05 como sólo anatómica. No identifica `q→ΔF/F`, ganancia ni filtro de nuestra prótesis.
- **Sapkal, Nature 2024:** activación/silenciamiento y estado locomotor de BDN2/DNg100. Usar un subconjunto compatible de sus datos para restricciones concretas; no convertir “hambre” en un multiplicador ni exigir DNg100 activo para toda orientación.
- **Schlegel, eLife 66018:** organización de rutas olfativas hacia descendentes. Sirve para seleccionar identidades y relés a comprobar en MaleCNS; conectividad femenina no aporta automáticamente pesos masculinos ni efecto dinámico.
- **Shiu 2024 y Pugliese, preprint revisado en 2026:** ejemplos de predicción de intervenciones con validación experimental y límites de basal/neuromodulación o VNC. Ya hay referencias locales: reutilizar sus negativos antes de repetirlas. Ritmo motor no equivale a apoyo, orientación ni aprendizaje.
- **Matheson 2022 y Kathman 2026:** distinguen olor, viento, dirección objetivo y persistencia neural. Restringen qué observar en recuperación; no justifican introducir un controlador externo ni copiar un circuito sin dependencia demostrada. El depósito Kathman ofrece 1,3 GB: en esta ronda sólo se consultó su ficha, sin descargarlo completo.

Enlaces primarios, datasets y decisiones concretas en [FUENTES_PRIMARIAS.md](FUENTES_PRIMARIAS.md). La corrección de identidad/sexo/preparación es necesaria antes de transferir una conclusión, no una exigencia de reproducir toda la fisiología.

## Secuencia ejecutable siguiente

**1. Completar una atribución acotada desde estados existentes.** Dueño: Matrix Astra. Una sesión máxima de 60 minutos, 300 s CPU, 4 GiB RAM, cero integración neuronal/corporal y cero descargas masivas. Elegir previamente los checkpoints completos disponibles de 48 y recuperar sólo los operandos necesarios del operador efectivo. Examinar DNg100/DNb05 y relés identificados; no elegir otra salida porque gira mejor. Distinguir aportes presinápticos actuales, transformación receptora y lector. Si faltan estados de frontera, registrarlos como ausentes y detener la reconstrucción exacta; los snapshots de 100 ms no rellenan lo que falta. Salida: una operación localizada que merezca intervención, o una carencia concreta que indique qué registrar. No otra auditoría global.

**2. Como máximo dos prototipos de instrumento, sólo si hacen falta.** P1: contabilidad por fuentes y observación de puertos, reutilizando el operador/radar. P2: un pronóstico dinámico desechable de la consulta elegida, con frontera e historia conservadas. Se mantienen como alternativas A/operator exacto, B/pronóstico mecanístico con contexto y C/predictor fisiológico restringido por datos; este último queda pospuesto por falta de validación compatible. No integrarlos por votación ni convertirlos en el cerebro.

Para P2, fijar antes IDs, escalas, relojes, estados y observables. Validar en una condición nueva frente al motor de referencia, preservando también sham y un control de intervención. No puntuar como predicción la PN usada de entrada. Reutilizar tolerancias científicas vigentes; medir coste total de preparación+ejecución. Si no aporta precisión suficiente y al menos 2× de reducción de coste en esa consulta, no conservarlo como acelerador. El 2× es un criterio nuevo prospectivo de utilidad de ingeniería, no una rehabilitación de fallos anteriores.

**3. Una sola intervención mecanística, no otra vida exploratoria.** Elegir A o B sólo tras el punto 1; C sería reparación. Misma prehistoria/checkpoint completo, sin reinicializar filtros, recursos, baselines ni memorias. Cambiar una operación local cuyo efecto inmediato y descendente se haya predicho con signo, ventana y mínimo material. Rechazarla si el efecto local aparece y el siguiente efecto predicho no, o si el control produce el mismo cambio. Un acierto neural y un acierto conductual son resultados separados.

Presupuesto orientativo máximo para ese futuro contraste: cuatro continuaciones de hasta 100 ms, una sola GPU, 1.200 s de pared, 1.500 s CPU, 4 GiB RAM y 12 GiB GPU, hasta dos variantes nuevas sin barrido. Se deriva del coste de 48 —aproximadamente 1.502 s de pared por segundo simulado agregado— dejando margen para carga/guardado. **No se lanzó aquí.** Congelar contrato y comprobar que el checkpoint permite reanudación real; un volcado diagnóstico no basta. Si la pregunta exige latencias mayores o más memoria, debe justificarse otro presupuesto antes de ejecutar, no ampliarlo tras un fallo. Un negativo a 100 ms no demuestra ausencia de una respuesta lenta.

**4. Volver a la orientación funcional cuando exista una predicción distinta del negativo 46.** Mantener DNb05 como lector actual; observar DNa02 es un contraste, no sustituirlo automáticamente. Comparar izquierda/derecha, control común de entrada total emparejada y sham, con avance asistido idéntico y declarado. Fijar signo, magnitud útil, ventana y reader antes de correr. No basta otra diferencia pequeña de signo correcto: debe llegar al mando aplicado. Esta prueba no exige iniciar avance por DNg100, ni permite afirmar iniciación autónoma.

**5. Cerrar las etapas mediante conducta causal.** Para 4: fuentes/geometrías reservadas y ventaja de orientación/aproximación del feedback vivo frente a replay emparejado, conservando asistencia y controles. Para 5: viento físicamente alcanzable, direcciones/tiempos no empleados para elegir la solución, comparación con controles y reducción sostenida del error posterior. Conservar el negativo de 40: el error aumentó de 19,41° a 27,56°. Una prueba lateral breve no cierra 4 y una DN activa no cierra 5. La iniciación autónoma y la marcha de seis patas mantienen resultados separados.

## Decisión, responsables y parada

La viabilidad de investigar y refutar estas alternativas está respaldada por infraestructura y datos existentes. **La suficiencia de este preparado para superar 4/5 sigue sin demostrarse; no hay fundamento para prometerlo ni para declarar imposible el proyecto.** La ausencia de calibración biológica completa no impide una prueba funcional correctamente etiquetada; tampoco permite llamar biológica a una mejora de ingeniería.

Matrix Astra conserva integración del plan y próxima consulta de transferencia. Motor C++/CUDA aporta instrumentos/portabilidad cuando exista esa consulta, sin otra corrida GPU duplicada. ChatGPT revisa predicciones y fuentes; Jev clasifica opciones acotadas. No se mandaron correos a autores ni se delegó control del organismo.

Esta ronda termina con plan conjunto, un diagnóstico ejecutado y evidencia portátil. No se eligió todavía una ley neuronal nueva. Clasificación del plan: **PROMETEDOR_NO_CONFIRMADO**; identidades algebraicas comprobadas localmente sólo en su alcance. Campaña48 mantiene **DESCARTADO** para su criba. La publicación completa de 48 sigue pendiente del límite Git de 2 GiB; esta entrega pequeña no la sustituye.

Reproducir el diagnóstico, con Python 3 y NumPy, desde una extracción nueva:

```bash
python3 check_recorded_law.py --verify
python3 -O check_recorded_law.py --verify
```

La cápsula contiene el subconjunto crudo necesario para este cálculo, protocolo, fuentes del analizador, procedencia y revisiones. No contiene el cerebro ni todos los datasets históricos; no ofrece reejecutar 48. `VERIFICACION_CAPSULA.json` documenta reproducción y corrupciones deliberadas. Las limitaciones de acceso y coste figuran en `CIERRE.json`.
