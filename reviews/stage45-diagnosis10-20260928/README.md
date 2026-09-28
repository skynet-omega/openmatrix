# Diagnóstico de fondo — etapas 4/5, 28 de septiembre de 2026

**Conocemos el bloqueo funcional, pero no una causa única demostrada.** La preparación responde a entradas sensoriales; todavía no hemos demostrado que esas respuestas produzcan una orden útil, aplicada al cuerpo, que mejore orientación y recuperación. La carencia central es identificar y validar esa cadena. Los pesos, las leyes celulares, el estado, las entradas y el lector son hipótesis competidoras; la evidencia no permite culpar exclusivamente a uno.

Esta revisión no ejecutó CNS/GPU ni cambió el organismo. Conserva los negativos de 40–52. En 52 hubo respuesta aérea en muchas DN y no hubo rescate olfativo material con G/I; el giro calculado no se aplicó. Por tanto, esa campaña diagnostica transferencia y no demuestra ni refuta por sí sola toda navegación posible. No exigir signo neural constante: una respuesta temporal puede producir una corrección corporal útil.

## Respuestas a las preguntas del usuario

| Pregunta | Respuesta sustentada | Qué sigue sin saberse |
|---|---|---|
| ¿Son los pesos de cada neurona? | Hay pesos por conexión y parámetros por célula, además de estados, puertos y lectores. El constructor genérico combina contactos y signo por neurotransmisor con parámetros previos. El operador efectivo incorpora especializaciones y normalización morfológica ya aplicada. | Qué parámetros son responsables del fallo y cuáles son identificables con los datos disponibles. No está demostrado que corregir pesos baste. |
| ¿No tenemos leyes neuronales claras? | Sí hay ecuaciones explícitas y un motor que las integra. La preparación combina leyes genéricas de tasa y modelos especializados. | Su adecuación fisiológica conjunta para esta tarea no está establecida. Reproducir una ecuación no identifica la ley biológica. |
| ¿Predictor y escáner bastan? | Son útiles para localizar actividad, comparar y acelerar integración. No predicen automáticamente el efecto de una intervención nueva ni convierten correlación en causa. | Falta demostrar predicción reservada de signo, magnitud y tiempo para la intervención relevante y un estado distinto. |
| ¿Faltan datos? | Hay anatomía, fisiología, comportamiento, artículos y negativos locales reutilizables. Parte está descargada pero sin extracción/mapeo suficiente; otra parte sólo es código o metadatos. | Correspondencia entre tipo celular, preparación, estímulo, unidades, observable y estado del modelo. No es simplemente falta de gigabytes. |
| ¿Proyecto desordenado e índice atrasado? | La evidencia está conservada y mayormente enlazada, pero los puntos de entrada acumulan planes antiguos. El catálogo histórico y su complemento son cortes fechados, no índices de las últimas campañas. | La revisión fue focal, no una certificación de todos los archivos. Se corrigieron tres enlaces actuales y se creó un mapa de acceso. |
| ¿Laboratorio/herramientas obsoletos? | La entrada de ASTRA delega al laboratorio histórico de MATRIX; su configuración apunta a otra preparación. Nueve versiones instaladas coinciden con el registro local de herramientas. | No comprobamos aquí GPU ni si todas las versiones son las más recientes. Actualizar paquetes no tiene una relación demostrada con el fallo conductual. |
| ¿Ajustar todo manualmente? | No es una estrategia defendible para decenas de millones de entradas del operador. Ajustar hasta obtener movimiento mezclaría identificación con fabricación de conducta. | Podemos estudiar parámetros compartidos por tipos/circuitos y sólo excepciones justificadas, usando mediciones y condiciones reservadas. No garantiza una solución única. |

## Modelo efectivo: lo observado y lo supuesto

El constructor histórico `anatomical_rate_brain.py` usa una dinámica de relajación hacia una función rectificada de entrada y recurrencia: tau, ganancia, umbral, tasa máxima y pesos. Su peso inicial se deriva del número de contactos y un signo asignado por neurotransmisor. Esa asignación no sustituye información de receptores ni una medición de eficacia sináptica. Sus unidades de entrada no son automáticamente corriente o voltaje medidos.

**No confundimos ese constructor con la preparación actual.** Motor leyó seis arrays del operador de 48/sham, padre de 52, comprobó hashes e igualdad de coeficientes CPU/CUDA guardados. La transformación morfológica `s=volumen/mediana`, `gain←gain/s`, `theta←theta*s` ya existe; no hay que añadirla otra vez. Tampoco su existencia demuestra que el volumen identifique capacitancia y eficacia. Los umbrales DNg100 guardados son aproximadamente 663,26 y 559,19 en unidades internas, no mV. La igualdad de arrays guardados no promete igualdad de trayectorias bajo toda precisión numérica.

El manifiesto del operador contiene 166.700 filas celulares y 25.582.938 entradas de pesos. Estas últimas no son 25 millones de mediciones fisiológicas independientes ni todos los contactos sinápticos individuales. La solución escalable a estudiar es una parametrización compartida con incertidumbre y heterogeneidad justificada, no editar esas entradas para obtener PASS.

## Qué aportan realmente nuestras herramientas

- **Escáner visual de 12 s:** su recibo declara 20.342 de 166.700 células, 121 muestras cada 100 ms y 887 enlaces mostrados, seleccionados por contactos anatómicos. Son enlaces estáticos entre somas, no corrientes medidas ni rutas activas demostradas. El observador de 52 es otro recurso: cubre temporalmente las 1314 DN. Ampliar cobertura corrigió una limitación real, sin identificar por ello un circuito de giro.
- **`live_forecast`:** acelera la integración del mismo modelo bajo sus condiciones de cualificación; no descubre parámetros biológicos faltantes.
- **Predictor mecanístico ORN:** tiene un alcance de preparación y protocolo reducido; su prueba independiente no aprobó los criterios fijados. Una calibración retrospectiva de ganancia no reparó la transferencia entre protocolos. Una lista vacía de espigas anotadas tampoco demuestra silencio biológico.
- **Dictionary Learning en herramientas09:** la configuración pequeña comparada no superó consistentemente Ridge. Eso no descarta toda la familia, pero no justifica instalar un sistema mayor como prerrequisito de orientación. Predecir un registro condicionado por otros registros no equivale a predecir una intervención.
- **Jaxley/SBI:** están disponibles y tienen pruebas instrumentales sintéticas; no hemos convertido eso en identificación fisiológica del cerebro. Pueden servir para una pieza acotada cuando exista un observable que distinga las alternativas.

Una mejora útil ahora sería añadir al análisis existente predicciones congeladas sobre intervenciones excluidas del ajuste, con unidades, estado, incertidumbre y un comparador simple. No hace falta construir un predictor universal antes de la próxima prueba.

## Datos disponibles y barreras de utilización

El catálogo histórico declara 36.977 archivos, 110 PDF y 2.195 páginas indexadas. Son cifras de ese corte, no un recuento nuevo ni cantidades de experimentos independientes. Revisé disponibilidad y fuentes explícitas; para archivos grandes comprobé presencia/tamaño y, cuando correspondía, el directorio del ZIP, sin afirmar validación completa de su contenido.

| Recurso local | Disponibilidad revisada | Decisión pendiente |
|---|---|---|
| Suver, DOI 10.5061/dryad.k06kh8f | Contenedor descargado; miembro 7z de unos 9,48 GB identificado. | Extraer sólo lo pertinente y resolver correspondencia APN/WPN con el modelo. No descargarlo otra vez. |
| Kadakia, DOI 10.5061/dryad.1ns1rn8xd | Datos usados por el predictor; voltaje/PID/ORN y recursos conductuales identificados. | Protocolos, etiquetas de espigas y preparaciones deben mantenerse separados; no son una medición simultánea de todo el circuito. |
| Gugel, DOI 10.5061/dryad.v15dv420q | Curvas procesadas de corriente/frecuencia. | Resolver identidad y condición celular; columnas no equivalen a animales. |
| Prisco, DOI 10.5061/dryad.bk3j9kdd1 | Contenedor PN/KC/APL identificado. | Calcio y perturbaciones deben conectarse mediante un observador explícito con las variables del modelo. |
| Yang/DNb05 | Código de análisis local verificado. | Este recurso no aporta por sí solo una calibración q→fluorescencia→giro ni grabaciones verificadas. |
| Kathman2026 | README/metadatos locales. | No etiquetarlo como grabaciones descargadas. |

Motor consultó cinco SQLite en modo lectura: cuatro tienen 334 registros y 34.872 observaciones cada uno; uno está vacío. Cubren una preparación biológica `a2_s_03` y ocho condiciones simuladas históricas. No sumar las copias como experimentos independientes ni confundir registros con animales. El laboratorio es reutilizable, pero esa cobertura es selectiva y no sustituye el catálogo amplio.

## Organización: reparación concreta y límite

Antes de corregir, README, ESTADO_ACTUAL y PLAN_ABC medían aproximadamente 34,4, 182,6 y 68,1 kB. El chequeo sintáctico encontró cero enlaces locales faltantes entre 80 y 264 enlaces revisados de los dos primeros, y tres rutas mal relativas entre 77 de PLAN_ABC. Se repararon esas tres. Otros tres destinos del catálogo histórico contienen prosa mal formada; no acreditan pérdida de datasets y el histórico quedó intacto.

La puerta `laboratorio.py` de ASTRA ejecuta MATRIX y su configuración conserva un checkpoint de septiembre14 y una prioridad antigua de etapa2. Las campañas actuales usan otra preparación. Esto puede desviar una consulta; no demuestra que el motor esté roto. No cambiamos la configuración histórica ni promovimos un checkpoint experimental.

Se conservó copia con hashes de los tres documentos antes de editar. El README se redujo a un punto de entrada y se añadió `MAPA_VIGENTE.md`, distinguiendo campañas, preparación, herramientas y recursos históricos. ESTADO_ACTUAL conserva la cronología, con este diagnóstico arriba. No se reconstruyó la biblioteca ni se movieron datos científicos.

## Qué dicen los asesores y qué concluyo yo

Todos formularon tres rutas antes de esta comparación, con exposición al historial declarada. **ChatGPT ASTRA_V2** prioriza entrada/configuración, dinámica identificable y vía causal hacia acción. **ChatGPT_Motor_V2** destaca leyes efectivas, contexto corporal y procedencia de variables. **Motor C++/CUDA** propone entrada, transmisión y lector/efector; además verificó arrays y la cobertura SQLite. Sus informes completos y rectificaciones se conservan aquí.

Mi conclusión propia es que debemos dejar de tratar «más activación» como sustituto de «información capaz de controlar». A la vez, tampoco podemos inferir incapacidad del cerebro a partir de un lector pequeño o de una campaña que no aplica el giro. Hay que contrastar por separado entrada, dinámica y uso corporal de la señal.

Corregí tres excesos de interpretación durante el intercambio y ambos ChatGPT aceptaron los matices: no exigir signo neuronal constante; una pérdida de efecto tras igualar L1 no identifica cantidad como causa única; FP32/FP64 son representación numérica, no unidades físicas. El 14,7436 % compara dos condiciones de campo L/R que estimulan ambas antenas; no dos antenas aisladas. ChatGPT revisó documentos y conceptos, no ejecutó los arrays; no se verificó PRO. El acuerdo de los asesores no constituye réplica biológica independiente. Jev no participó en esta ronda.

## Tres rutas vigentes y criterios para decidir

| Ruta | Pregunta e información usada | Discriminador y falsador |
|---|---|---|
| A. Representación sensorial | ¿El contraste aéreo contiene algo adicional a la diferencia de cantidad? Misma preparación y ley; entrada JO registrada después de FP32. | Cuatro brazos L/R × sin/con olor, cantidad L1 emparejada, distribución y L2 declaradas. Si persiste, la suma por sí sola no basta. Si desaparece, el efecto no es robusto a esa transformación; no identifica una causa única. |
| B. Transmisión y estado | ¿Una ley/estado compatible con datos cambia la transferencia selectiva? Variables observables y tipos identificados, sin blanco conductual privilegiado. | Intervención finita con control no-op, o pocos parámetros compartidos contrastados con intervenciones reservadas. Si exige reajustar cada condición, no se valida; si alternativas son indistinguibles, declarar no identificable. No repetir G/I por defecto. |
| C. Lectura y acoplamiento corporal | ¿La información disponible puede gobernar el cuerpo en contexto? Lector/signo/escala fijados prospectivamente y control físico de la interfaz. | Comparar feedback online con replay/control pertinente. Si la interfaz no aplica el efecto previsto o la señal online no ofrece ventaja reservada, falla esa hipótesis. No seleccionar retrospectivamente DN por la trayectoria más favorable. |

Motor C++/CUDA recibió autorización del usuario para preparar A en una campaña nueva, propuesta como53. Esta revisión no la duplica y no atribuye todavía un resultado. B/C siguen alternativas, con máximo dos prototipos completos por ronda y presupuesto previo. Contexto propioceptivo pertenece a C; no justifica integrar ya seis patas. Etapa4 requiere conducta dependiente de información online; etapa5 recuperación ante perturbación reservada durante segundos con la misma arquitectura. Los pilotos de 90 ms no sustituyen esas pruebas.

## ¿Es factible predecir sin conocer cada neurona a mano?

Sí hay precedentes limitados. Shiu y colaboradores produjeron predicciones sobre circuitos de alimentación/acicalamiento usando conectividad y supuestos celulares, contrastadas con intervenciones. Eso apoya intentar predicciones concretas; no valida nuestro preparado ni toda navegación. [Shiu et al., 2024](https://www.nature.com/articles/s41586-024-07763-9).

En visión de mosca, Lappalainen y colaboradores compartieron parámetros por tipos: 734 parámetros libres para un modelo de 45.669 células. Es un ejemplo de reducción del problema, no una receta validada para todo MaleCNS. Su ajuste usó una tarea, decoder y BPTT; ese entrenamiento no queda autorizado para nuestro organismo. [Lappalainen et al., 2024](https://www.nature.com/articles/s41586-024-07939-3).

El trabajo del «effectome» propone combinar anatomía e intervenciones para estimar efectos causales bajo supuestos concretos y los estudia en simulación. No es un dataset completo de eficacias reales ya disponible para cargar. [Pospisil et al., 2024](https://www.nature.com/articles/s41586-024-07982-0).

Suver distingue ensayos conductuales y fisiológicos: no transferir directamente velocidades, duraciones o hambre entre ambos ni llamar a eso calibración de JO. Motor conserva las condiciones concretas en su informe. [Suver et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6533146/).

**Juicio de factibilidad:** hay evidencia y medios para decidir el siguiente experimento sin ajustar manualmente todo el cerebro. No hay evidencia suficiente para prometer superar4/5 con la preparación actual, ni un resultado que demuestre imposibilidad general. Mejor predicción significa acertar una intervención nueva dentro de un alcance declarado, no reconstruir todo lo desconocido desde el conectoma.

## Alcance, entrega y parada

Fuentes locales y metadatos en `EVIDENCIA_LOCAL.json`, `HERRAMIENTAS_METADATA.json` y `aporte_motor/OPERADOR_EFECTIVO.json`; propuestas iniciales y respuestas completas preservadas. Las cifras instrumentales se reconstruyen desde esos recibos en `MEDICIONES.md`. Cero entrenamiento, cero simulación CNS/GPU, cero instalaciones o descargas masivas. Las comprobaciones cronometradas no representan todo el tiempo de lectura/redacción; no se inventa un coste CPU total de la sesión.

Presupuesto prospectivo: 180 s CPU propios +120 s del asesor, 4 GiB RAM, 30 MiB nuevos, 25 min de revisión activa; la entrega documental/red se contabiliza aparte si termina después. Clasificación: **PROMETEDOR_NO_CONFIRMADO** para las rutas de diagnóstico, ninguna admisión de etapa. Parada por diagnóstico solicitado y mapa completados. El paquete pequeño conserva este expediente, no contiene ni reproduce los gigabytes del organismo o los datasets; las cápsulas científicas anteriores mantienen su alcance.
