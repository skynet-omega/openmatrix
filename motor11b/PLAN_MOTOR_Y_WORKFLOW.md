# Motor y reparto del trabajo — 22 de septiembre de2026

## Decisión que cambia la siguiente ronda

El objetivo de rendimiento pedido es un segundo simulado en60 segundos reales. BLAS1 medido necesita5051,66 segundos reales por segundo simulado: falta aproximadamente84,19×. Aún no sabemos si esa meta cabe en este hardware con estas ecuaciones y precisión. Hay memoria libre, pero eso no demuestra que sobren ancho de banda, capacidad FP64 o paralelismo útil.

El código actual ya contiene kernels CUDA, Numba paralelo, buffers reutilizados y eliminación algebraica del sistema PN. Reescribirlo sólo por cambiar de lenguaje no garantiza el salto. Sí hay una pista concreta:178838 incógnitas PN,25,58millones de entradas del conectoma y un circuito de ejecución con muchas conversiones, evaluaciones pequeñas y sincronizaciones. El perfil04 cuenta13680 llamadasget deCuPy y350914asarray en10ms; no todas representan transferencias, y su duración host no mide exclusivamente PCIe.

La sonda A mide33,87s inclusivos en PN y15,69s en KC dentro de87,51s. Incluso asignando coste cero a esas dos regiones, el techo ilustrativo sería2,306× si el resto permanece igual. El perfil B después de limitar BLAS sigue pendiente. No usar ese techo como predicción de una reescritura que cambia todo el esquema de ejecución.

## Dos alternativas rivales, no una arquitectura elegida

| Alternativa | Operación exacta que se investiga | Ventaja posible | Falsador |
|---|---|---|---|
| Tramo acoplado compilado y residente en CPU | Mantener estados/estructuras contiguos en RAM y ejecutar varias operaciones del mismo subpaso desde un conductor compilado, con sumas y realimentación preservadas | Reducir llamadas Python, conversiones y crucesCPU/GPU del tramo; control sencillo del orden FP64 | El tiempo completo incluyendo fronteras no mejora o el coste restante ya excede el presupuesto de tiempo |
| Tramo acoplado residente en GPU | Mantener estados y coeficientes enVRAM, agrupar kernels dependientes y transferir sólo fronteras necesarias en sus tiempos originales | Aprovechar paralelismo por neurona/arista y evitar viajes de estados intermedios | Sincronizaciones, fases seriales, coste FP64 o memoria limitan el tramo; falla el residuo o la continuidad del estado |

La opción híbrida sólo es útil si sus fronteras son baratas. Conservar un solve42×42 enCPU puede ser rápido de forma aislada y costoso si obliga a sincronizar cada etapa. Los subpasos temporales dependen del anterior: no se pueden ejecutar simultáneamente como si fueran neuronas independientes. Con64 subpasos por milisegundo, la meta permite aproximadamente0,94ms real por subpaso completo. Esto es un presupuesto de ingeniería, no una nueva tolerancia numérica.

NVIDIA recomienda medir transferencias junto al cómputo y mantener datos enGPU entre kernels cuando resulte conveniente. [GuíaCUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device). Existen simuladores neuronales comoArbor; su existencia no implica compatibilidad directa con nuestra dinámica PN, acoplamientos y cuerpo. Conviene consultar sus técnicas, sin asumir que sustituyen al motor actual. [Arbor](https://docs.arbor-sim.org/en/stable/).

## Captura mínima y presupuesto prospectivo

La primera decisión requiere capturar un tramo que cruce las fronteras costosas, no sólo la matriz42×42. Guardar una vez topología/plan estático; después estados, entradas y salidas por etapa, coeficientes de dos subpasos consecutivos y estados ocultos necesarios para continuar. Cubrir preparación y olor. Calcular el tamaño antes de simular; los31,43MB observados por cápsula no cabían en20MB.

Preflight: codificaciónUTF8 explícita, validación de que todos los campos requeridos se guardarán, reproducción CPU del formato, e importador deNsight que pueda convertir la versión de la traza. Alternativamente usar temporizaciónCPU y eventosCUDA con alcance documentado; la instalación del importador no debe convertirse en puerta para todo prototipo. Una captura sin estados completos sólo admite un benchmark de función.

Presupuesto propuesto para la próxima ronda, aún no consumido: una captura integrada de hasta1200s, dos prototipos como máximo, hasta600s de benchmark por prototipo,8GiB RAM y12GiBVRAM, sin reintentos de simulación ni cambios de criterios. Medir calentamiento aparte y luego referencia/candidata con iguales entradas; incluir conversiones y transferencias. Detener una alternativa si falla en el residuo original o si su aceleración no reduce materialmente el coste total. Antes de promover habrá que registrar criterios numéricos prospectivos y probar continuidad integrada en un intervalo nuevo.

## Autocrítica concreta

Mi revisión previa comprobó34 pruebas y hashes, pero no detectó que el tamaño de las cápsulas excedería el límite ni que faltaba el importador deNsight. Tampoco cubrió la lectura bajo el localeASCII que produjo el fallo. Esa es una omisión de ingeniería: un preflight breve habría evitado parte del coste perdido. Se recuperó la comparación deSONDA desde los archivos existentes, sin repetir el organismo, y se conserva el fallo original.

Por otra parte, el fallo de igualdad exacta enMOTOR11 no refuta toda optimización; es necesario estudiar cuánto cambia y qué consecuencias tiene. Pasar de ahí a declarar equivalencia sólo por errores pequeños sería otro exceso. Etapa3 permanece abierta. Optimizar el motor compra capacidad experimental, no valida por sí solo la hipótesis biológica.

## Flujo aplicado

Jev: clasificación acotada de tareas y, cuando haya una necesidad concreta, filtrado de pasajes. Codex: diseño, programación, ejecución y verificadores. ChatGPT: revisión externa de hipótesis, errores y alternativas. El usuario autorizó coordinación directa por la app; ya se envió ENCARGO_ENVIADO_A_CHATGPT.md. Los archivos continúan disponibles enF:\Downloads. No se han usado subagentes locales.

La conexiónJev está probada y los resultados se incluyen enjev_workflow/. Su recomendación no opera dentro del organismo ni controla sus decisiones. No se promete ahorro global hasta medirlo. La API no recibe pesos, conectoma ni la clave dentro del contexto. El paquete de retorno contiene datos y código, no credenciales.

## Revisión externa recibida y respuesta local

ChatGPT respondió directamente y su texto se conserva en REVISION_CHATGPT.md. Coincide en comparar un tramoCPU compilado conGPU residente y en medir primero bajoBLAS1. Su propuesta de adoptarBLAS1 se interpreta sólo como referencia experimental del desarrollo; no sustituye todavía la política operativa. El umbral1,25e-6pA citado es el de los reportes lineales observados, no una tolerancia universal para todos los estados y pasos del organismo. Cada verificador conservará su criterio vigente; cualquier presupuesto de error nuevo será prospectivo.

La crítica adicional es pertinente: una aceleración84,19× exige que el coste realmente inalterado no exceda aproximadamente1,19% del tiempo actual. No conocemos aún esa cobertura. La coincidencia entre revisores no sustituye medición. El revisor no tuvo el ZIP11B y declaró esa limitación.

El usuario autorizó también subir directamente los ZIP. El intento mediante Computer Use falló antes de abrir el navegador: app-server no se pudo iniciar por ruta inexistente, os error3, incluso tras reiniciar el kernel. No hubo subida. La mensajería directa sí está verificada y el paquete se conserva enF:\Downloads; la reparación del accesoUI queda como bloqueo técnico concreto, no como una solicitud de permisos.
