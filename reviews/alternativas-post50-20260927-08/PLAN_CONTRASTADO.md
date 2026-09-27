# Plan contrastado después de 50

**Hay cuatro ternas propias y un plan de selección; no hay solución demostrada ni nuevas simulaciones del CNS. Etapas 4/5 abiertas.** La consulta anterior había pedido una sola acción a cada ChatGPT y dejó la discusión demasiado estrecha. Esta revisión pidió tres operaciones distintas a cada asesor y registró la terna propia antes de leer sus respuestas nuevas. Todos conocemos los negativos previos; no se declara revisión ciega.

## Lo que propuso cada uno

| Autor | Primera alternativa | Segunda alternativa | Tercera alternativa |
|---|---|---|---|
| ChatGPT ASTRA_V2 | Lector de cinco pares DN, con signos publicados y mediana de cambios respecto al prefijo propio | Predictor de recurrencia mediante Jacobiano/JVP del modelo real | Propiocepción de zancada estructurada frente a la misma cinemática desfasada |
| ChatGPT_Motor_V2 | Intercambiar recíprocamente estados PN realmente observados entre contextos guardados | Instrumentar cinco pares DN preseleccionados por Yang, sin entrenar decoder | Proyección anatómica de DN hacia canales premotores izquierda/derecha |
| Motor C++/CUDA | Integración por conductancias para células genéricas | Lector de dos acciones DNa02/DNg13 | Entrada mecanosensorial antenal de dirección del aire |
| Matrix Astra, autor | C1: conductancias con reversión y voltaje explícito | C2: regulación intrínseca local comparada con un sesgo/control emparejado | C3: lector oponente DNb05/DNb06 preseleccionado por función |

Son doce propuestas con coincidencias, no doce mecanismos independientes. Los textos originales y sus revisiones están preservados. C1 y M1 convergen por separado; eso no constituye prueba de utilidad.

## Qué cambió al contrastarlas con datos

Se inspeccionaron las cinco parejas de Yang en los cuatro estados finales de 48, fijando las identidades antes de leer sus valores. Coste del cálculo: 9,452 s CPU, cero CNS/GPU. El q nativo coincide exactamente con las cuatro DN disponibles en la última traza de cada brazo. La extracción corrigió dos errores auxiliares antes de producir resultados: ubicación de trazas por bloques y diferencia entre tasas publicadas y q nativo; se conservan recibos.

DNb06 y DNg13 sólo presentan residuos subnormales idénticos entre brazos; DNa02 está prácticamente silenciosa. DNa01 presenta actividad principalmente derecha y DNb05 está activo. Esto reduce la prioridad del rescate inmediato por añadir DNb06 o DNg13 al lector. No excluye transitorios ni otra preparación. No se escoge DNa01 como reemplazo por ser la que mostró mayor diferencia.

La mediana propuesta por ASTRA produce cero en el contraste exploratorio de esos endpoints contra sham. Ese cálculo no reproduce su protocolo temporal con baseline preestímulo, pero revela una objeción concreta: tres canales silenciosos suprimen los canales activos. ASTRA retiró la promoción inmediata de su mediana. El lector poblacional general sigue siendo una hipótesis; esta variante pierde prioridad.

El JVP necesita el RHS híbrido completo y estados correspondientes. No están guardados todos los estados intermedios de 50; no podemos reconstruir gratis su propagador ni tratar la respuesta a perturbaciones finitas como lineal. Queda como herramienta prospectiva con validación propia, sin conceder capacidad predictiva actual. Un transplante PN tampoco separa por sí solo ley y estado: ambos determinan la respuesta.

La hipótesis propioceptiva tiene una fuente concreta: el depósito de Fujiwara contiene fases y posiciones 2D de articulaciones de tres patas izquierdas sincronizadas con actividad y movimiento. Verificamos documentación y código pequeños, no los 1,11 GB crudos. Falta la correspondencia a los ángulos y receptores del cuerpo actual. Una cinta articular impuesta sería una intervención diagnóstica declarada, no marcha desarrollada por el organismo.

La homeostasis C2 no recibe prioridad biológica por analogía. La fuente adulta contrastada describe compensación en mushroom body durante días, no un setpoint universal de las DN en segundos. Como ingeniería, necesitaría controlar también la contribución inicial y la dependencia de estado, además de la actividad media; un simple sesgo medio no basta.

## Mi análisis

El motor puede reproducir correctamente una abstracción que todavía sea insuficiente para navegar. La presencia de conexiones y las sumas exactas permiten dejar de culpar a una desconexión simple en las filas verificadas; no convierten los parámetros provisionales en fisiología. Seguir cambiando sólo patrones olfativos dejaría intactos los supuestos del receptor neuronal, las modalidades sensoriales y el lector.

También había riesgo de exigir más de lo que mide 50. Para un controlador separable `u=k(L−R)`, el contraste factorial de interacción es cero aunque conserve información direccional. Por tanto, J positivo no es una necesidad universal de orientación. Esto no rescata retrospectivamente 40/46 ni afirma que el CNS actual contenga ese controlador; impide transformar el criterio de una hipótesis temporal en una puerta para cualquier arquitectura.

Falta calibración específica, pero eso no obliga a esperar todos los datos antes de probar ingeniería. Hay que fijar una operación y sus supuestos, controlar el aumento indiscriminado de actividad y medir utilidad causal. Una mejora funcional puede conservarse como provisional sin equivalencia biológica.

## Tres rutas sustanciales que mantengo abiertas

### A. Percepción de la dirección del aire

**Cambio:** un propietario sensorial transforma aire relativo al cuerpo en deflexión antenal/entrada de subtipos JO-C/E identificados. El entorno aporta aire y olor; el circuito recibe sensores, nunca rumbo correcto ni una regla externa olor×viento→giro. El torque histórico no se reinterpreta como aire medido.

**Fundamento:** [Suver 2019](https://datadryad.org/dataset/doi:10.5061/dryad.k06kh8f) documenta comparación antenal para dirección del viento; [Álvarez-Salvado](https://datadryad.org/dataset/doi:10.5061/dryad.g27mq71) separa orientación durante olor de búsqueda posterior. La comprobación local independiente encuentra 335 JO-C/E con lateralidad de entrada por `rootSide` (203 L, 132 R), todas por nervio AN. No son 335 receptores funcionalmente equivalentes: falta fijar subtipos y correspondencia con relés.

**Primera criba:** mapear esa frontera anatómica y una transducción explícita, con cero CNS; viento nulo y reflejo físico de antenas deben dar las propiedades correspondientes. No exigir simetría artificial de poblaciones anatómicamente desiguales.

**Experimento discriminante:** factorial olor sí/no × aire sí/no, más control de aferencia desacoplada o replay si pasa la criba. Un giro explicado por aire solo acredita anemotaxis, no navegación olfativa. Debe conservarse la distinción entre una modalidad sensorial nueva y la perturbación física reservada de etapa 5.

**Falsador:** entrada consumida y relés afectados sin efecto descendente direccional material, o beneficio igualmente explicado sin olor/por replay. No aumentar dosis tras ese fallo. Si añadir aire cambia las observaciones permitidas por el contrato original, registrar el ensayo como multisensorial nuevo; nunca reetiquetar un PASS histórico.

### B. Integración neuronal por conductancias

**Cambio:** las filas genéricas mantienen voltaje y reciben corrientes `gE(EE−V)` y `gI(EI−V)` más fuga; la conductancia total cambia también la constante temporal. Conservar conectoma, receptores especializados, señales legales y lector en el primer contraste. La escala y reversión son supuestos explícitos, no valores medidos de DNg100.

**Fundamento:** el código ya usa una distinción parecida para visión, mientras las filas genéricas siguen una suma firmada y relajación fija. La fisiología inhibitoria PN local sirve como restricción parcial, no calibración DN. Con entradas DNg negativas congeladas, cambiar sólo ganancia/tau no resuelve el objetivo cero; esta candidata cambia la operación recurrente.

**Primera criba:** equilibrios, sensibilidad y constantes temporales con E/I registrados, junto a un control de corrientes de actividad y respuesta excitatoria emparejadas en un punto ajeno a navegación. Usar V explícito, no reconstruir conductancias desde un saldo agregado como si fuese voltaje. Un contrafactual local no demuestra causalidad de la red.

**Experimento discriminante:** padre y candidata ante dos entradas laterales espejo, con prefijo común por modelo y control de deriva al adoptar el nuevo estado. El experimento de adopción no reclama una historia interna idéntica imposible entre variables distintas. Si se estudia respuesta uniforme en lugar de lateralidad, se etiqueta transferencia, no orientación.

**Falsador:** sólo amplificación constante, cambio de basal, saturación global o ausencia de transferencia sensorial selectiva material. No seleccionar otro potencial, otra conversión de unidades o nuevas filas después de mirar el resultado.

### C. Contexto propioceptivo de locomoción

**Cambio:** una entrada de movimiento articular medida, independiente del objetivo, se transforma por los receptores existentes; se compara coordinación temporal original con una cinta que conserva marginales y altera relaciones temporales. No se integra todavía marcha autónoma de seis patas.

**Fundamento:** [Fujiwara 2022 y su depósito](https://zenodo.org/records/6365304) permiten contrastar señales de zancada y estado visual/motor. La evidencia corresponde a otro circuito; es una hipótesis de transferencia, no una explicación establecida de DNb05.

**Primera criba:** confirmar un segmento pertinente, unidades, articulaciones observadas y qué grados de libertad quedan no identificados. Si las posiciones 2D no permiten nuestro transductor, se conserva ese límite sin fabricar ángulos. Comprobar marginales en la salida del transductor, no sólo en el movimiento original: sus filtros tienen memoria y borde temporal.

**Falsador:** con intensidad y condiciones emparejadas, la coordinación no añade respuesta selectiva/feedback o el efecto se explica por una asimetría de la cinta. Una actividad mayor por cualquier movimiento no acredita este mecanismo.

## Orden de trabajo y presupuesto propuesto

1. **Hecho en esta revisión:** recoger las cuatro ternas, contrastar fuentes, inspeccionar cinco pares en cuatro estados, comprobar el efecto de la mediana y verificar la población JO. Se reprodujo la proyección guardada bajo `-O` y se rechazaron tres corrupciones de identidad/valor/no finito. Esto cambia la prioridad sin gastar una vida neuronal.
2. **Siguiente acto concreto:** criba A de correspondencia JO→relés y criba B de equilibrios/sensibilidad desde los operandos guardados; C conserva su comprobación de mapeo cinemático. Para esta preparación posterior: hasta 120 s CPU, cero CNS/GPU y descargas pertinentes hasta 50 MB. Sin nueva arqueología general.
3. **Dos prototipos como máximo:** A y B por separado si sus cribas permiten especificarlos. C queda como tercera ruta; C2/homeostasis y los lectores alternativos quedan documentados con sus falsadores, sin fusionarlos automáticamente. No hay una candidata operativa lanzada por este informe.
4. **Tope propuesto para el siguiente piloto:** hasta nueve brazos de 100 ms (cinco de A y cuatro de B) más hasta 16 ms de cualificación de software: 916 ms CNS, 5000 s CPU, 4200 s pared, 24 GiB RAM, 14 GiB VRAM y 8 GiB nuevos; cero reintentos automáticos. Amplía explícitamente los 800 ms inicialmente sugeridos por el motor para no omitir el sham y el control de aferencia. Antes del lanzamiento deben quedar congelados escala, inicialización, subtipos, ventanas, umbrales materiales y errores numéricos pertinentes. Son topes prospectivos de diseño, no recursos ya consumidos ni permiso pendiente del usuario.
5. **Etapa 4:** el superviviente debe mejorar orientación con interacción sensorial online frente a replay emparejado y controles de lateralidad/retirada, con geometrías reservadas y misma arquitectura. Una señal DN, una marcha impuesta o una mejora en replay físico no bastan. Iniciación autónoma y orientación asistida se informan por separado.
6. **Etapa 5:** congelar el superviviente y probar perturbación mecánica reservada con recuperación de orientación. Si se cambia el modelo tras verla, deja de ser confirmación reservada. Mantener criterios conductuales y comparadores; no sumar PASS de organismos distintos.

**Clasificación:** plan PROMETEDOR_NO_CONFIRMADO. La viabilidad de estos discriminadores es concreta; la suficiencia del organismo para superar 4/5 sigue sin demostrar. Motivo de cierre de esta consulta: comparación solicitada completa y primeras cribas de datos ejecutadas, con decisiones que pueden fallar y sin promesa de éxito.
