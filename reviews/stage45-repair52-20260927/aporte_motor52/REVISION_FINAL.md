# Resultado independiente de la campaña52

**Las etapas4/5 siguen abiertas. La reparación de unidades quedó comprobada; G no produjo una mejora olfativa útil frente al control I. El aire conserva una respuesta neuronal que justifica un control de intensidad, todavía sin demostrar orientación corporal.**

Matrix completó diez brazos90ms y16ms de cualificación:916msCNS,2909,568sCPU y2601,541s de pared de cola. Esta revisión sólo lee arrays y código en CPU; no ejecuta otra simulación. No modifica51,52 ni los históricos fuera de `aporte_motor52`.

## Qué comprobé directamente

- Los siete controles iniciales y los diez brazos completos: reloj, identidad del panel, ORN, última salida y conservación del prefijo dentro de cada ley.
- La entrada de aire usa **una sola conversión×10** de la velocidad corporal anterior. Reconstruí el receptor desde las ecuaciones y cinemática guardadas, sin importar el receptor del ejecutor. La posición también conserva la conversión nativa a mm.
- Los tres pares con/sin observador conservan los arrays comunes, la secuencia de auditoría de eventos y los11/12 hashes de propietarios exigidos. La cualificación pasó además desde copia limpia con lecturas del histórico/árbol original e importaciones CNS/GPU prohibidas.
- La proyección51 incluida en52 coincide con los arrays originales51 y sus hashes. Recalculé las diferencias52−51, contrastes G/I y conteos temporales por grupo; coinciden con `PILOT_RESULTS.json` y `POPULATIONS.json` en los campos contrastados.

No reconstruí los estados completos que sólo se representaron mediante hashes. No ejecuté el verificador adicional de pasos/eventos aceptados ni la publicación completa. La prueba de neutralidad se refiere a tres pares2ms con observadorDNg común, no a toda duración posible ni al observadorDNg aislado.

## Resultado funcional y neuronal

| Contraste fijado antes de correr | Resultado52 | Interpretación |
|---|---:|---|
| Aire izquierdo/derecho, sin olor: semidiferencia del mando registrado | −0,104472°/s | Magnitud superior al mínimo0,02; lectura sin aplicar al cuerpo. |
| Aire izquierdo/derecho, con olor | −0,115090°/s | Misma conclusión; ambos contrastes coinciden exactamente con51 en los lectores originales. |
| Ventaja específica G frente a I ante olor | 0,0000591091°/s | Sólo0,2955% del mínimo0,02: criterio negativo. |
| Mismo contraste G−I en DNb05 | 5,38765×10⁻⁸ unidades de salida | También inferior al mínimo1,6×10⁻⁵. |

Los dos campos de aire reducen el mando medio respecto al control de campo cero; no aparece la inversión bilateral esperada como criba de orientación. La semidiferencia temporal tiene21 muestras positivas y19 negativas en51–90ms, con un cambio de signo. Una media distinta de cero no certifica un controlador direccional correcto.

La suma de entradaJO derecha es14,7436% mayor que la izquierda, tanto en la frontera aditivaFP64 registrada como en su conversiónFP32 reconstruida. El receptor conserva la asimetría anatómica y la misma ley por célula; aún no separa configuración espacial de cantidad total. No corresponde renombrar este contraste como selectividad direccional a intensidad emparejada.

G e I conservan avance basal también sin olor: máximos de mando0,064791 y0,151480mm/s, respectivamente. El pequeño movimiento registrado no acredita iniciación selectiva por olor. **El mando angular aplicado es cero en los diez brazos.** No hubo prueba de seguimiento ni perturbación mecánica con recuperación en52.

## Qué aportó ampliar el registro

Con el umbral descriptivo fijado de|Δq|>10⁻⁶ entre los dos campos de aire:

| Grupo | Población | Células con diferencia alguna vez | Con diferencia al final |
|---|---:|---:|---:|
| JO-C/E |335|235|235|
| AMMC/WED, sin olor |1108|581|580|
| AMMC/WED, con olor |1108|580|580|
| Todas las descendentes, con o sin olor |1314|534|532|

De las532DN finales,529 quedaban fuera del panel temporal anterior. Ahora sabemos que la respuesta no se limita al par usado para leer el giro. Esto **no** identifica qué DN debe gobernar cada acción ni autoriza escoger la que más mejore el resultado.

Los tres grupos ya muestran alguna diferencia en la muestra11ms, la primera tras el inicio del estímulo. Con muestreo cada1ms no se resuelve aquí su orden causal fino. El umbral es descriptivo, sin calibración fisiológica ni estadística; los puntos temporales pertenecen a una sola preparación.

## Efecto de reparar la velocidad corporal

| Ley | Máximo cambio en el panel antiguo de16 células frente a51 | DN finales con cambio>10⁻⁶ |
|---|---:|---:|
| G |≈0,00354|1076, con y sin olor|
| I |≈0,00825|1095, con y sin olor|

La reparación sí afecta la dinámica recurrente cuando el cuerpo se mueve. No era correcto extrapolar una entrada contrafactual pequeña a una cota de cambio cerebral. Sin embargo, el contraste olfativoG−I sigue prácticamente en el mismo negativo específico: pasó de0,0000592244°/s en51 a0,0000591091°/s en52. No se cambió el umbral para reinterpretarlo.

**Precisión de la frontera observada:** `JO_drive` y `AirOwner.seen` corresponden a la suma aditivaFP64 anterior a la conversión del operador; no son una captura directa de cada entradaJO del kernel. Inspeccioné la fuente ejecutada `engine/fp32_operator.py`, SHA256 `af1edfcf7a5b08bbf5679e8d3437233f31014f40bf58637f3e3d422d779c5624`: reserva `drive` enFP32 y copia allí el argumento antes de llamar aCUDA.

El suplementoCPU `FRONTERA_FP32.json` reconstruye esa conversión desde los registros. En los cuatro brazos con aireL/R cambian valoresFP64 frente a51, pero **cero valores cambian después del castFP32**; esto explica la invariancia observada del lector sin negar la corrección×10. En campo cero yG/I sí cambian valores convertidos. Es una reconstrucción bajo el contrato de entradaJO nativa cero del preparado, no una nueva observación interna porRHS. No se cambia ninguna fuente científica ni se ejecuta CNS. El siguiente emparejamiento debe comprobarse después de la conversión, en el punto que consume el kernel.

## Siguiente decisión que recomiendo

**A, interfaz sensorial:** una comparación finita con cantidadJO consumida emparejada entre campos, conservando soporte y declarando la transformación artificial. Informar también normaL2/distribución: igualar sólo la suma no iguala todos los aspectos de la entrada. Si el efecto sobrevive, fijar de antemano lector, signo y efector para probar orientación corporal online frente a controles/replay. No entrenar ni elegir otro lector usando los ganadores del panel52.

**B, transmisión/estado:** conservar como rival la intervenciónPN acotada, después de su no-op vivo y estado de eventos cualificados. El fracasoG/I reduce la prioridad de repetir esas conductancias como rescate selectivo, no descarta cualquier ley celular posible.

**C, contexto propioceptivo:** conservar el donante temporal declarado y su control de marginales cuando corresponda. No mezclarlo ahora con otro cambio de ley ni integrar patas para ocultar la falta de mando sensorial.

Una permutación que intercambia células con idéntica entrada dentro del mismo grupo puede ser un no-op; debe comprobarse que realmente cambia la entrada antes de gastar otra vidaCNS. El control de intensidad es el próximo discriminador, no una certificación de etapa. Etapa4 todavía exige conducta dependiente de información online; etapa5 exige después recuperación ante perturbación física con la misma arquitectura.

## Evidencia y coste de esta revisión

`SCIENCE_INDEPENDIENTE.json`: recomputación de los diez brazos,0,4533sCPU,148,7MB de pico. `DELTA_Y_POBLACIONES.json`: contraste con originales51, conteos por milisegundo y primera muestra por célula,0,6460sCPU,150,6MB de pico. Se ejecutaron con comprobaciones activas bajo `python -O`; ceroCNS/GPU. Los costes son de estos cálculos, no todo el trabajo de lectura. Permanecen dentro del presupuesto120sCPU/1GiB/50MiB de la revisión.

`verify52.py science` ya está ejecutado y reproducible desde datos52 más su mapeo canónico incluido. `audit_delta52.py` utiliza también los originales51 para verificar la procedencia de su proyección; no se presenta como un verificador portable sólo con52. La reproducción desde copia limpia realizada por esta revisión corresponde a cualificación; no se atribuye una descarga/publicación que realiza Matrix.
