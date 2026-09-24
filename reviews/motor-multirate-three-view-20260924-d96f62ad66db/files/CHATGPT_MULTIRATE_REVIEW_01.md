# Revisión externa ChatGPT: multirritmo y dos carriles

Respuesta textual recibida el 24-09-2026 en la conversación autorizada «Investiga el conectoma de mosca»; propuesta externa, no validación independiente de la cápsula ni instrucción del usuario. El propio revisor declara qué archivos leyó y qué no ejecutó. Se conserva el texto original, incluidos marcadores internos de citas del chat.

## Dictamen

**No usaría «0,5–5 ms rápido / 100–1.000 ms lento» como fundamento del planificador.** Las fuentes primarias respaldan múltiples escalas y mecanismos temporales, pero no esa partición binaria del cerebro ni la cadena completa **ORN→PN→LH→DNb05→músculo del cuello**.

**El multirritmo sigue siendo una hipótesis numérica razonable; los resultados actuales no respaldan todavía 40–60 barridos/ms.** La caché temporal falla tanto su puerta de reducción de productos como el coste CPU, y el fixture acoplado falla su defecto de corriente. No convertiría esos negativos en ajustes de tolerancia.

**Alcance:** leí fuentes y recibos; no descargué la cápsula ni ejecuté scripts, CUDA o MuJoCo. La aritmética indicada abajo se calculó sobre los valores publicados.

## 1. Biología: escalas distintas no equivalen a dos carriles independientes

| Evidencia primaria | Qué respalda | Qué no respalda |
|---|---|---|
| **Kazama y Wilson, 2008:** la entrada excitatoria lateral a PN comienza aproximadamente **1,5 ms después** de la directa; ambas componentes aparecen en la misma preparación. | Diferencias temporales entre vías sinápticas convergentes. | Que toda corriente sea «lenta», o que cada región pertenezca a un único carril. :chatgpt-content-reference{index="0"} |
| **Gaudry et al., 2013:** diferencias ipsi/contralaterales de primera espiga de **2,47 ms en DM6** y **1,01 ms en DM1** —esta última no alcanzó significación convencional—. El giro optogenético comenzó unos **70 ms después de la respuesta ORN**. | Diferencias de pocos milisegundos pueden asociarse con una conducta posterior. | Que 1–2 ms sea la latencia total del circuito, ni que 70 ms sea una constante neuronal. El experimento conductual tenía la cabeza fijada al cuerpo. :chatgpt-content-reference{index="1"} |
| **Nagel y Wilson, 2011:** transducción y generación de espigas aportan dinámicas distintas; una realimentación adaptativa modifica la transducción. | Procesos rápidos y adaptación pueden coexistir en una misma ORN. | Una división anatómica rígida entre neuronas rápidas y lentas. :chatgpt-content-reference{index="2"} |
| **Yang et al., 2024:** DNb05 presenta correlación bilateral con giro durante marcha. Las intervenciones detalladas se centran en DNa02/DNg13; sus cambios de actividad preceden variables locomotoras unos **150 ms**, parcialmente afectados por la mecánica de la esfera. | Relación de ciertas descendentes con componentes del giro. | Una ruta funcional completa desde LH hasta músculos cervicales a través de DNb05. Las conexiones cervicales discutidas en ese trabajo corresponden a DNa02, no demuestran esa afirmación sobre DNb05. :chatgpt-content-reference{index="3"} |

**Una descendente que atraviesa el conectivo del cuello no es, por eso, una neurona motora del cuello.** Tampoco debe transferirse automáticamente un resultado de estimulación optogenética a la latencia de transducción química del olor. En particular, la lateralización de Gaudry se produjo con la cabeza inmovilizada: no necesitó una rotación visible de la cabeza para generar el giro medido. :chatgpt-content-reference{index="4"}

La conclusión utilizable es más estrecha: **seleccionar pasos por defecto, sensibilidad y acoplamiento, no por etiquetas «reflejo», «memoria», PN o KC**. Una variable de adaptación lenta puede modular rápidamente otra; una transmisión sin espigas puede variar continuamente.

## 2. Qué dicen realmente las nuevas pruebas

### La caché temporal queda negativa en su contrato

Los 26 productos más la pasada inicial representan **64/27 = 2,37× menos recorridos**, pero la candidata consumió **2,10× el tiempo CPU** del producto directo: 14,95 frente a 7,12 s. Además, la validación externa gastó otros 35,06 s, correctamente informados aparte. Eso no es velocidad de un motor CUDA ni de una trayectoria producida por la candidata. 

Hay dos límites importantes en la captura:

- **`operator_epoch` se rellena con ceros y se utiliza W inicial fijo.** Es válido para estudiar ese mapa lineal offline, pero no certifica los pesos efectivos de cada evaluación ni sus sustituciones PN/APL. 
- **`query_s` es un reloj local.** En `REAL_RESULT.json`, la consulta 35 está a 62,3877 µs y la 36 vuelve a cero. No todos los retrocesos deben interpretarse como una trayectoria física: hay subetapas especulativas y reinicios del reloj de época. La caché puede seguir siendo válida para W fijo porque verifica el residuo de la entrada actual; una reconstrucción temporal o una cola de eventos necesita también **época, fase y propietario**. 

La mediana del **1,53 % de aristas** al aplicar `|Δs|>10⁻⁶` es ahora información entre consultas reales, más pertinente que un endpoint de 1 ms. **Sigue sin ser una fracción de trabajo que pueda omitirse con fidelidad certificada.** 

### El fixture acoplado separa dos errores distintos

En `coupled_temporal_fixture.py`, la iteración converge cuando cambian poco los valores de corriente en **inicio, mitad y final**. Después se comprueba:

\[
R_I(t)=W\widetilde s(t)-P(t)
\]

en otros puntos. Puede converger el problema de tres nodos y seguir siendo inadecuada la interpolación entre ellos. **Más iteraciones sobre los mismos nodos no garantizan eliminar ese defecto.** El máximo publicado, 0,00193943, sigue excediendo 0,001.  

Tampoco interpretaría \(4004/89\) como una aceleración validada: la referencia es un RK4 fino elegido para el fixture, y la candidata no satisface todas sus condiciones.

### La neutralidad observada tiene un alcance concreto

`trace_real_rhs_1ms.py` compara el hash de **`h.state`**, contra el array CNS esperado, además de contadores. No realiza en esa función una comparación integral de todos los estados corporales y de propietarios especializados. El recibo respalda neutralidad de ese vector y controles, **no por sí solo de todo el organismo serializado**. 

## 3. ¿Pueden bastar 40–60 barridos/ms?

**No con el coste aislado actual y manteniendo intacto todo lo demás.** La envolvente publicada permite esta cuenta condicional:

| Productos completos/ms | A 1,396 ms/producto | Margen hasta 60 ms reales |
|---:|---:|---:|
| 40 | 55,840 ms | 4,160 ms |
| 42 | 58,632 ms | 1,368 ms |
| 60 | 83,760 ms | Ya excedido |

No son tiempos medidos dentro del organismo. Pero muestran que **«40–60» no constituye por sí solo un presupuesto suficiente**. Pasar de 1.146 a 42 requiere **27,29× menos productos**, y el perfil todavía deja aproximadamente **1,033 s/ms fuera de `NativeGraph.advance`**. 

También deben contarse los **ocho predictores descartados y ocho avances aceptados**, no solo ocho bloques físicos. Una fórmula de coste basada en ocho bloques no se traslada directamente al acoplamiento actual. Suprimir el trabajo predictor necesita una equivalencia propia, no una resta contable.  

**Los cortes físicos deben conservarse, pero un corte no tiene por qué requerir un CSR completo** si existe un forzamiento temporal correcto y una cota de su error. Ese es el problema matemático pendiente. Evaluar poco una variable lenta no es suficiente para resolverlo.

## 4. Tres algoritmos distintos

Todos requieren preservar masa, estados, entradas comprometidas, SET/ADD y orden de eventos. Para una aproximación \(\widetilde y\), con masa constante invertible en el tramo:

\[
\mathcal R=M\dot{\widetilde y}-F(t,\widetilde y).
\]

Una cota de trayectoria necesita propagar \(M^{-1}\mathcal R\), no solamente comprobar que una corriente o una iteración cambie poco. Los eventos requieren además verificar su transición y tiempo. Si no puede acotarse un cruce rasante, corresponde refinar o volver al padre, no declarar el evento irrelevante.

### A. MRI-GARK: distintos ritmos por operador, no por anatomía

Separar exactamente:

\[
M\dot y=F_f(t,y)+F_s(t,y),
\]

donde \(F_f\) contiene operaciones locales económicas y \(F_s\) la corrección recurrente cara. Si se usa una corriente predicha dentro de \(F_f\), **su diferencia respecto de la corriente real pertenece a \(F_s\)**.

Entre etapas lentas, resolver un problema rápido forzado por combinaciones temporales de evaluaciones lentas:

\[
M\dot v=F_f(t,v)+\sum_j\gamma_j(t)\,F_s(t_j,Y_j).
\]

Los coeficientes \(\gamma_j\) deben proceder de un esquema publicado con condiciones de orden, no de escoger dos tiempos biológicos. Existen métodos MRI-GARK acoplados precisamente porque una separación aparentemente razonable puede ser inestable cuando las particiones interactúan fuertemente. :chatgpt-content-reference{index="15"}

**Error y fallback.** Estimador del esquema más defecto del forzamiento entre etapas; cortes exactos en eventos. Si la envolvente o el dominio falla, dividir el macrointervalo o repetirlo con el padre. Cambio de W, máscara, capacidades o puertos invalida las evaluaciones lentas incompatibles.

**Experimento real:** un bloque completo de 125 µs, desde estado capturado, con sus eventos y operadores efectivos. Comparar propuestas, eventos y defecto contra el padre refinado. Contar todas las evaluaciones lentas, reintentos y predictor. **Falsador:** necesita tantos nodos/correcciones que no reduce trabajo, o falla antes el error de acoplamiento.

**Ventaja potencial:** reduce evaluaciones completas aun cuando todas las fuentes evolucionen. **Riesgo:** el operador caro quizá no sea lento en el sentido requerido.

### B. QSS2 con trayectorias y contabilidad de influencia

Aproximar cada transmisión mediante una trayectoria lineal mantenida:

\[
\widehat s_j(t)=s_j(t_j)+\dot s_j(t_j)(t-t_j),
\qquad |s_j-\widehat s_j|\le\eta_j.
\]

Actualizar sus destinos cuando la envolvente alcanza \(\eta_j\), **aunque no haya ninguna espiga**. Para W fijo, los coeficientes del polinomio de entrada se actualizan por columnas CSC.

\[
|I_i-\widehat I_i|
\le\sum_j|W_{ij}c_j|\eta_j+\epsilon_{\mathrm{aritmética},i}.
\]

En visión deben mantenerse conductancias positivas y negativas separadas. Los puertos impulsivos usan sus transiciones originales; no se identifican con las transmisiones graduadas.

QSS2 tiene antecedentes formales, pero sus resultados para sistemas lineales y sus hipótesis no constituyen una garantía automática para este sistema no lineal híbrido. :chatgpt-content-reference{index="16"}

**Error y fallback.** Propagar el defecto de entrada por un sistema de comparación de estados; presupuestos por unidad y receptor. Si demasiadas columnas se invalidan, volver al CSR completo. El caché de una propuesta rechazada no se publica como historia física.

**Experimento real:** una época con \(s,\dot s\), eventos y versiones del operador realmente consumidos. Medir actualizaciones ponderadas por grado, coste CSC y cota recurrente. No elegir \(\eta_j=10^{-6}\) porque dio una fracción atractiva en el diagnóstico.

**Falsador:** las cotas obligan a actualizar casi todo el grafo, o el scatter y su contabilidad cuestan más que el padre. **Ventaja potencial:** explotación temporal local; no exige dos frecuencias globales.

### C. Colocación implícita de la trayectoria de corriente

En un bloque, representar la corriente por coeficientes \(G\). El solucionador local completo —incluida su masa— produce una trayectoria de transmisión \(\mathcal S(G)\). Resolver conjuntamente:

\[
R(G)=G-\{W\mathcal S(G)(t_j)\}_j=0.
\]

Una corrección Newton–Krylov usa:

\[
J_Rv=v-\{W\,D\mathcal S(G)[v](t_j)\}_j.
\]

Esto es distinto de fijar una corriente e iterar cuatro veces: resuelve el acoplamiento y permite diagnosticar por separado **residuo nodal**, **error de representación temporal** y **error del solver local**.

**Error y fallback.** Recalcular el residuo original y acotarlo entre nodos. Si la interpolación es insuficiente, subdividir según una regla predeclarada; si las iteraciones fallan, volver al padre. Cambios de eventos o de W invalidan linealizaciones y precondicionadores afectados.

**Experimento real:** el mismo bloque de 125 µs de A, con máximo de nodos e iteraciones fijado antes. Contar también cada producto W dentro de JVP y auditorías. **Falsador:** converge en nodos pero falla entre ellos —el negativo actual—, o las resoluciones consumen demasiados productos.

Waveform relaxation tiene antecedentes para combinar transmisión continua y espigas, pero su ganancia intercambia comunicación por reintegraciones; no garantiza ventaja en esta GPU. :chatgpt-content-reference{index="17"}

## 5. Qué haría primero

**Priorizaría A como siguiente prototipo y mantendría B como rival; no implementaría los tres esta ronda.** La evidencia ya muestra que reducir bytes por producto y la caché afín global no bastaron. A prueba directamente si el acoplamiento caro puede evaluarse menos, sin presuponer que la mayoría de las neuronas está inactiva.

Secuencia finita propuesta:

1. **Una captura real de 1 ms, máximo 240 s**, solo si los datos existentes no contienen los campos necesarios. Añadir época, fase predictor/confirmado, `h`, operadores efectivos y estado inicial completo de los propietarios del bloque.
2. **Un bloque de 125 µs por candidato**, máximo dos candidatos y 120 s de replay cada uno, sin cuerpo ni nueva campaña conductual. El cuerpo permanece una dependencia pendiente, no un coste eliminado.
3. Contabilizar productos completos y equivalentes por aristas, auditorías, rechazos, invalidaciones, memoria y pared. Si el bloque ya requiere más de cinco productos completos, se ha consumido casi todo el presupuesto de 40/ms contando solo ocho bloques, **antes de los predictores**. Ese resultado puede descartar la ruta para la meta, aunque sea numéricamente correcta.

No seleccionaría el algoritmo por el número menor de productos solamente. Debe superar el error de trayectoria y eventos; posteriormente, ganar en el organismo completo. El antiguo FAIL, la sonda de caché y el residual del fixture quedan sin reinterpretar.

## Alcance de lectura y ejecución

**Leídos:** README; `MULTIRATE_REVIEW_REQUEST_01.md`; `TRACE_PLAN_01.json`; `trace/RESULT.json`, `TRACE_DELTA_01.json` y el capturador; código de caché y `REAL_RESULT.json` por tramos; `coupled/RESULT.json`, su código y nota matemática; `FEASIBILITY_ENVELOPE.md`; resumen del negativo multiescala; MANIFEST y ARCHIVE. Consulté las fuentes primarias citadas.

Hay una inconsistencia documental menor: el pedido dice que la cápsula no está incluida, pero **el manifiesto sí enumera** `trace/base_csr_rhs_first64.npz`, 154.282.391 bytes, y el archivo dividido tiene 154.342.526 bytes. La frase del pedido quedó desactualizada. **No pude descargar las partes ni verificar sus hashes.**   

**No ejecuté scripts, cápsula, CUDA, MuJoCo ni organismo.** Solo recalculé los cocientes y presupuestos mostrados.

**Conclusión:** las escalas biológicas motivan investigar multirritmo, pero no certifican una partición. El requisito decisivo es demostrar que el acoplamiento recurrente puede reconstruirse con menos evaluaciones y error controlado. Hoy no está demostrado; la meta de 1 s/60 s exige además reducir sustancialmente el resto del motor.
