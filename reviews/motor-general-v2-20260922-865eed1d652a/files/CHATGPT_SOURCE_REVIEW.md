Revisión documental recibida; no ejecución externa. Mensaje e3954e54-6d54-4bf3-b1b3-4fa9fa75f9d0. Fuentes publicadas en5f04e635. PRO/máximo confirmado por usuario, selector no observado.

**Sí conservaría esta base para desarrollar el backend implícito general A. No veo motivo para desechar la separación descriptor–compilador–ejecutor. Pero encontré defectos concretos en la semántica numérica, la identidad del modelo y la confirmación de cambios que corregiría antes de considerar esos contratos cerrados.**

**Esta revisión es documental.** Intenté únicamente comprobar la disponibilidad del entorno; los comandos mínimos fallaron con `TransportTimeoutError`. **No ejecuté el núcleo, pruebas, verificador ni paquete extraído.** Los contraejemplos siguientes se deducen del código; no los presento como reproducciones locales.

## 1. Bloqueante numérico: CPU y GPU no comparten todavía la semántica de todos los literales

En `Expression`, CPU compila el AST original de Python. En cambio, `cuda_expr()` convierte cada constante mediante `repr(float(node.value))`. Por tanto, una expresión constante puede calcularse con enteros exactos en CPU y con números FP64 ya redondeados en GPU. La reparación de `scale` y masa no corrige esta diferencia. 

**Contraejemplo admitido por la gramática**, con estado adimensional y `tau` en segundos:

```python
rhs = "(9007199254740993 - 9007199254740992)/tau"
```

En CPU, la resta de enteros es **1**. La traducción CUDA convierte ambos literales al mismo valor representable, `9007199254740992.0`; la resta es **0**.

No es una pequeña discrepancia del integrador: **los dos evaluadores entregan derivadas distintas antes de integrar**.

**Corrección:** definir una única semántica numérica en el IR y aplicarla antes de emitir ambos programas. Por ejemplo, constantes y operaciones FP64 explícitas también en el evaluador CPU. Otra política podría rechazar expresiones que no puede representar coherentemente, pero rechazar solo literales grandes no basta: los resultados enteros grandes también pueden construirse mediante operaciones.

**Falsador:** esta expresión debe producir la misma derivada en ambos backends conforme a la política declarada, o ser rechazada explícitamente. No debe aceptarse y devolver dos ecuaciones efectivas.

No he identificado que este caso aparezca en las cargas publicadas; **no atribuyo a este defecto sus errores o tiempos registrados**.

## 2. Bloqueante de identidad: un mismo hash puede describir dinámicas diferentes

`digest()` ordena las claves mediante `sort_keys=True`, pero `Model` asigna posiciones a los estados recorriendo su orden de inserción. A su vez, la masa COO utiliza índices de ese almacenamiento. 

Considérese una población con estados `x`, `y`, ambos con ecuación `-rate*estado`, y masa diagonal:

\[
M=\operatorname{diag}(1,2).
\]

| Orden de los estados | Dinámica por identidad |
|---|---|
| `x`, `y` | \(\dot x=-rate\,x;\quad \dot y=-rate\,y/2\) |
| `y`, `x`, dejando el mismo COO | \(\dot x=-rate\,x/2;\quad \dot y=-rate\,y\) |

Los diccionarios difieren únicamente en el orden de sus claves: **`digest(spec)` coincide, pero las ecuaciones asignadas a las identidades no**.

Esto afecta tanto a la identificación experimental como a los recibos de `replace()`: podrían registrar iguales `before` y `after` aunque haya cambiado la dinámica.

**Corrección mínima:** incorporar a la identidad ejecutable el mapa ordenado que da significado a los índices de masa y puertos. Alternativamente, construir un IR canónico y remapear conjuntamente estados, masa y conexiones. **Ordenar solamente las claves al serializar no resuelve el problema.**

### Consecuencia para snapshots

Los checkpoints guardan descriptor, estado, perfil nominal, algoritmo, tiempo y próximo paso, pero no identifican la versión del ejecutor ni los valores efectivos de las tolerancias. `restore()` reconstruye con el código instalado en ese momento. Eso permite cargar silenciosamente un snapshot bajo otra implementación o bajo un `PROFILES` modificado. 

La reanudación exacta **en el entorno probado** sigue siendo una afirmación delimitada. Para extenderla, añadiría esquema ordenado, dtype e identidad numérica del backend/compilador y del perfil. Un cambio compatible puede permitirse mediante una migración explícita; no debe confundirse con reproducción del ejecutor original.

## 3. Bloqueante transaccional: se confirma un estado migrado sin evaluar las nuevas ecuaciones

`replace()` construye el motor nuevo, migra estados por identidad y comprueba que esos valores sean finitos. **No evalúa el nuevo RHS en el estado migrado y en el tiempo actual antes de sustituir el objeto vivo.** 

Contraejemplo:

- El estado vivo conservado es `x=0`.
- El nuevo descriptor declara `initial=1` y `rhs="log(x)/tau"`.
- La construcción sobre `initial=1` es válida.
- La migración conserva `x=0`, que es un número finito.
- La transacción se confirma, aunque el nuevo RHS contiene `log(0)`.

El siguiente avance fallará por derivada no finita, pero **el modelo anterior ya fue reemplazado**. Las pruebas actuales de edición inválida comprueban errores de descriptor, como retardos no admitidos; no cubren esta incompatibilidad entre una descripción válida y el estado heredado. 

**Corrección:** antes de confirmar, evaluar puertos, variables derivadas y RHS del candidato en `self.t` y en el estado efectivamente migrado, sin avanzar el estado vivo. Cuando existan restricciones explícitas de dominio, comprobarlas también.

**Falsador:** el ejemplo anterior debe rechazarse conservando modelo, estado, reloj, controlador y recibos originales. Esto verifica validez en la frontera de la transacción; no promete que una trayectoria nunca encuentre posteriormente una singularidad.

## 4. Validación de entradas: hay conversiones silenciosas que deben cerrarse

`cell_ids` y los índices COO de conexiones se convierten a `int64` **antes de comprobar su validez original**. Un índice fraccionario puede truncarse y pasar después el control de rango: por ejemplo, un índice negativo entre −1 y 0 puede convertirse en 0. También puede cambiarse silenciosamente una identidad celular. 

Exigiría integralidad y rango **antes** de convertir, con la misma regla para offsets y coordenadas discretas. Esto es una reparación de admisión, no un nuevo mecanismo.

Respecto de unidades, el código comprueba **dimensiones simbólicas**, no conversiones de escala. Por ejemplo, `1000*mV` y `mV` producen el mismo diccionario dimensional, sin conversión numérica. El comentario declara que no hay conversión implícita `mV→V`; conviene que la gramática rechace factores de escala no implementados, en vez de aceptar una notación que parezca aplicarlos. **No encontré por ello una conversión incorrecta demostrada en los modelos publicados.** 

## 5. Lo que sí conserva correctamente la ruta GPU examinada

En el código leído, las direcciones capturadas permanecen asociadas a buffers propiedad de `GPU`. La aceptación copia `fine` sobre `x`, en lugar de intercambiar arrays y dejar al grafo apuntando al almacenamiento anterior. Inicialización, captura, lanzamiento, escritura y lectura utilizan el mismo stream. `flag` se reinicia una vez por intento y acumula los problemas de sus subetapas antes de permitir la confirmación. **No encontré en esta ruta el defecto de reinicio intermedio del indicador señalado para el prototipo PN.** 

La secuencia RK4 completo frente a dos medios pasos utiliza tiempos de etapa coherentes; el divisor 15 corresponde a esa estimación de duplicación para RK4. La norma implementada es un **máximo ponderado**, no una RMS que diluya una variable entre muchas. No sustituye una comprobación del error global. 

El clamp constante no diagonal modifica la ecuación correspondiente antes de resolver la masa; coincide con el falsador publicado `[0, 0.5]`. **He contrastado la implementación y leído ese resultado; no lo he reproducido.**  

## 6. El obstáculo específico para A implícita: falta el Jacobiano del acoplamiento completo

La diferenciación actual genera una **derivada diagonal local**, considerando constantes los puertos de entrada. No sigue la cadena salida → CSR → entrada. Eso es coherente con el alcance limitado del candidato C; **no puede reutilizarse como Jacobiano completo de A**.  

Un falsador mínimo es:

\[
f(x,u)=\frac{-x+u}{\tau},\qquad u=2x.
\]

La derivada total es \(+1/\tau\). La derivada local que mantiene \(u\) fijo es \(-1/\tau\).

Para la composición general:

\[
f(x)=F\bigl(x,b+W\,H(x)\bigr),
\]

el producto necesario es:

\[
J_fv=F_xv+F_u\,W(H_xv).
\]

Añadir esta propagación al compilador mantiene la arquitectura genérica; no requiere excepciones anatómicas. **Es una capacidad pendiente, no evidencia de que el RHS explícito actual omita esas conexiones:** ese RHS sí recalcula salidas y acoplamiento en cada evaluación.

## 7. Publicación y verificador: dos correcciones acotadas

**Cobertura de extensiones.** `verify.py` exige 16 registros, pero no comprueba que sean exactamente las 16 combinaciones distintas de cuatro semillas, dos algoritmos y dos perfiles. Una lista con duplicados podría pasar esa comprobación. Además, `validate_record()` contrasta `eligible_trajectory`, mientras las extensiones almacenan `eligible`; esa bandera concreta no queda contrastada. Los errores se recalculan, pero la cobertura y esa coherencia requieren pruebas adicionales. **No afirmo que haya duplicados en los resultados entregados.**  

**Tiempo completo.** En `benchmark.py`, `started` se inicia después de las importaciones y de construir `fixture`. Por eso `total_worker_s` no representa literalmente todo el proceso. Conviene llamarlo tiempo de la ventana medida o añadir una medición externa. **Esto no invalida el tiempo de avance con escáner, cuya frontera está definida por separado.** 

La reparación FP64 está identificada y la pérdida de condición ciega se declara. El negativo de C en HH y la cobertura parcial de exactitud grande también están delimitados en el informe. No encuentro motivo documental para convertirlos en resultados distintos de los publicados.  

## Decisión para continuar

**Conservar descriptor, generación de RHS, almacenamiento residente, puertos y separación de perfiles. Corregir primero semántica FP64, identidad/estado, admisión de índices y prevalidación de transacciones. Después desarrollar el JVP global y conectar el backend implícito.**

La ausencia declarada de GPU para masa general, DAE, ruido, retardos, eventos de umbral y cuerpo **no es un defecto oculto de esta entrega**. Tampoco la velocidad sintética permite promover esas capacidades. 

No exigiría repetir inmediatamente la campaña grande para comprobar los contraejemplos de esta revisión: son pruebas pequeñas de contrato. Si una reparación cambia la evaluación numérica utilizada por aquella campaña, entonces corresponde identificar y repetir los brazos afectados.

**Archivos leídos:** índice `README.md`; `PUBLIC_REVIEW_REQUEST.md`, `RESULTADOS.md`, `PLAN.json`, `CORE_REPAIR.json`; `model.py`, `runtime.py`, `autodiff.py`, `cli.py`; `validate.py`, `extensions.py`, `extension_test.py`, `review_falsifiers.py`, `external_falsifiers_02/RESULT.json`; `benchmark.py`, `verify.py` y `verify_package.py`, todos del commit indicado.

**El dictamen es conservar y reparar, no reiniciar la arquitectura ni aprobarla todavía como motor general validado.**
