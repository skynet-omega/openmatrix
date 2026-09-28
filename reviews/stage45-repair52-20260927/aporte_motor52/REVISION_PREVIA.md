# Revisión de Motor C++/CUDA — reparación y observación52

**Actualización tras los siete controles:** verificación independiente de los16ms guardados **PASS**, repetida desde copia limpia con lectura del histórico y del árbol original prohibida. Los tres pares conservan arrays comunes exactos, secuencia `event_audit` y hashes de los propietarios exigidos. La conversión×10 y su frontera temporal se reconstruyeron desde posiciones/velocidades guardadas. Etapas4/5 siguen abiertas; los diez brazos90ms aún no están revisados.

**Dictamen prospectivo inicial:** el diseño permite una repetición acotada de51; aún no demostraba neutralidad viva. Se revisaron fuentes antes de congelar y se comunicaron las observaciones a Matrix Astra. La primera nota se cerró durante la cualificación, antes de los brazos90ms; no es una revisión ciega del historial51. El anexo final registra la comprobación posterior, sin cambiar la propuesta ni sus criterios.

## Qué se conserva y qué se repara

Diez brazos de90ms y16ms de cualificación, desde el mismo preparado48 restaurado. La única reparación física es `10 * qvel[:3]`: el cuerpo usa centímetros y el receptor exige milímetros por segundo. Se conserva el reloj: en cada intervalo el aire recibe la velocidad y orientación del cuerpo al cierre anterior. Campo cero sigue incluyendo movimiento relativo; durante la identidad4ms se desactiva la entrada JO completa mediante `prefix_ms=4`.

Las leyes G/I, amplitudes del aire, ventana51–90ms y criterios51 se conservan. El control natural izquierda/derecha mantiene dosis totales diferentes. Este diseño reproduce el efecto de la reparación; no resuelve por sí mismo la confusión entre dirección e intensidad ni aplica giro al cuerpo.

## Instrumentación revisada

`Panel.record` recibe el mismo vector de salida comprometida que ya se leía para16 células. Copia2757 posiciones elegidas por anatomía:335JO-C/E,1314DN y1108AMMC/WED. Añade694ORN en arrays separados. No llama nuevamente al solver, `release`, coeficientes ni GPU. Son salidas del modelo; no se convierten automáticamente en potenciales o frecuencia de disparo.

La prueba CPU con un vector real guardado51 pasó bajo `python -O`: entrada marcada como sólo lectura conservada; buffers sin alias; cambiar el registro no cambia la entrada y cambiar la entrada después no cambia registros previos. Detectó tres corrupciones deliberadas del verificador. Coste del cálculo medido0,0174sCPU, pico123,3MB, ceroCNS/GPU. `PRUEBA_PANEL_CPU.json` identifica fuentes y alcance. Esta prueba sólo comprueba la copia CPU.

Es proporcionado **no** hashear todo el organismo alrededor de cada copia de2757 valores. Para la neutralidad viva se conservan tres pares2ms ON/OFF con intervención desde el primer intervalo: padre con aire/olor, G con olor e I con olor. Comparan todas las trazas comunes, registro DNg heredado y secuencia `event_audit`, más hashes separados del estado final. El observador DNg existe en ambos brazos: no se prueba aquí su ausencia ni su neutralidad.

## Cobertura del estado y cambios solicitados

El testigo final cubre los ocho propietarios cualificados49 —sesión, prótesis, publicación, operador almacenado, operador efectivo, entrada, intervalo y motor— más frontera, `event_audit` y aire. G/I añaden su estado propio, incluidos q0, S, gE0 y rangos con máscaras explícitas para infinitos iniciales. El estado corporal completo pertenece al árbol de sesión; su contrato histórico usa el vector de integración de MuJoCo, no sólo qpos/qvel. `used_light` no se omite.

Se pidieron y comprobaron dos correcciones del verificador antes de congelar:

1. Exigir las11 claves de propietario,12 enG/I, y formato SHA256; comparar solamente dos diccionarios iguales permitiría aceptar la misma omisión en ambos.
2. Comprobar finitud, identidad, filas y última muestraORN contra la salida final; comprobar dimensiones sin identidad ni contenido no bastaba.

`event_audit` sí incluye identidad, productor y tiempo por evento. El historial incluye evaluaciones del integrador: una secuencia idéntica de auditoría no se debe presentar como conteo depurado de eventos físicos aceptados. Tampoco los hashes finales prueban igualdad de cada estado intermedio no registrado.

## Comprobación independiente preparada

`verify52.py` sólo utiliza Python/NumPy y archivos guardados. No importa CNS, CuPy ni MuJoCo ni modifica fuentes del ejecutor. Tiene tres modos:

- `panel-test`: prueba de copias descrita, ya ejecutada.
- `qualification`: compara las trazas, arrays comunes, secuencias de auditoría y hashes con inventario exigido; reconstruye la entrada aérea y verifica factor×10 una sola vez y su desfase de un intervalo. Pendiente de los siete resultados.
- `science`: después de cualificación, recalcula contrastes DNb05/mando, dosisJO, movimiento y cobertura temporal del panel en los diez brazos. No declara navegación ni recuperación.

La reconstrucción del receptor usa las ecuaciones directamente, sin importar el receptor de52. El cruce velocidad52 exige identidad con `10 * velocidad_nativa_previa`, no una tolerancia ajustada a su salida. La conversión independiente de cuaternión tiene una tolerancia de redondeo declarada. Un control adicional sobre los datos51 rechazó la antigua velocidad sin×10 (`CONTROL_ERROR51.json`,0,0222sCPU). El primer intento de ese control no pudo usar la primera muestra porque51 no guardaba `initial_native_velocity`; se verificaron las fronteras observables2..90 sin inventar el estado inicial.

Los hashes se comparan como testigos guardados; este análisis no reconstruye estados completos que no se archivaron. Tampoco vuelve a demostrar ecuaciones CUDA: el verificador de Matrix conserva ese chequeo de DNg. Se incorporó una copia canónica de `node_ids.npy` en `aporte_motor52/donors`,1.333.728bytes, SHA256 `ab90597b7b0ce07cbc22cb39a65b70bea2ac73cc7fd225951bd9d73f2fc8dd3f`, contrastada con la procedencia09. El verificador la usa por defecto y admite `--node-ids` exigiendo ese mismo hash; ya no necesita leer el histórico para cualificación/ciencia. La copia limpia de cualificación quedó verificada después, como registra el anexo; `panel-test` conserva su donante51 y no forma parte de esa afirmación portable.

## Límites y decisión siguiente

Etapa4 exige mejor orientación corporal con información sensorial online frente a controles pertinentes. Etapa5 exige recuperación tras una perturbación física.52 conserva mando angular **observado sin aplicarlo**, así que un resultado positivo sólo puede apoyar transferencia neuronal.

Se mantienen tres alternativas distintas: A, interfaz sensorial con cantidades consumidas emparejadas; B, transmisión/estado mediante intervención PN cualificada; C, contexto propioceptivo temporal con donante declarado. No elegir neurona, ventana, ganancia o polaridad por el ganador52. La siguiente decisión usará los resultados corregidos; las patas/músculos siguen posteriores.

Presupuesto de esta revisión:120sCPU,1GiBRAM,50MiB nuevos, ceroCNS/GPU. Sólo se escribe en `aporte_motor52`; Matrix conserva la cola científica. Los tiempos anteriores son de cálculos concretos, no todo el tiempo de lectura o de sesión.

Fuentes congeladas revisadas: `observer52.py` SHA256 `f3cfffeff7c9d271d284c93d73956ffee9648cbaba6ffe37f25824389e644fad`; `pilot52_owners.py` `325ca64a22a080ccd52ec971c2ad6079138c8995c28974e80e0b396e0d72f1e1`; `run_pilot.py` `5820ecbe783a9b847e6c2b8f946535b7777c91f8f41050134b617966e3fba330`; plan `5ea82813df23d84789621f96a215852d1b546ae360fca044c53b4f5d2d7a71f2`.

## Anexo: comprobación independiente de los16ms

`QUALIFICATION_INDEPENDIENTE.json` registra PASS bajo `python -O`,0,1866sCPU y126,1MB de pico. Compara las33 trazas de identidad49 y todos los arrays comunes de cada par; exige11 propietarios en el padre y12 enG/I. El chequeo del receptor usa datos cinemáticos y ecuaciones independientes; no ejecuta el receptor del runner. La revisión sólo consume CPU; los16ms CNS fueron ejecutados por Matrix.

`PORTABLE_QUALIFICATION.json` registra la segunda ejecución desde53 archivos copiados,17.670.950bytes. Un gancho de auditoría rechaza lecturas del histórico y del árbol de campaña original fuera de esa copia, así como importar CNS/CuPy/MuJoCo. El resultado científico coincide con el primer recibo. La copia temporal se eliminó al terminar; permanecen hashes de todos sus insumos y el resultado. No se duplican checkpoints ni se afirma reanudación del CNS desde este paquete.

Comando desde una entrega que conserve la estructura52:

```bash
python -O aporte_motor52/verify52.py qualification --root . --out aporte_motor52/NUEVA_CUALIFICACION.json
```

El archivo de salida debe ser nuevo. `science` está implementado pero permanece sin ejecutar hasta que existan los diez brazos completos. La neutralidad establecida se limita a los tres pares2ms, con observador DNg común; no acredita cada trayectoria futura ni valida todas las ecuaciones del organismo.
