Turno da66c032-272a-4a3a-8bc4-5dab0b2cea2a. Revisión documental; no ejecutó arrays ni CUDA.

**Priorizaría B: separar la evolución local de los mecanismos del cálculo global de sus entradas, conservando la realimentación mediante iteración y control del error. Lo contrastaría con C: un backend de orden alto sobre el mismo operador completo. A —esquema actual y propuesta nominal— quedaría como control, no como apuesta principal para un salto grande.**

La guardia ha mejorado la precisión observada, pero no ha eliminado el patrón costoso: **cada intento CNS sigue evaluando seis veces una cadena global de coeficientes**. Cambiar solamente el controlador o reducir sincronizaciones no ataca necesariamente ese trabajo. La referencia que falla durante el milisegundo 15 tampoco permite calificar la trayectoria de 20 ms.  

**Esta revisión es documental.** Leí las fuentes concretas que enumero al final; no ejecuté CUDA, pruebas ni comparaciones desde arrays y no comprobé independientemente el ZIP.

## 1. Qué permiten realmente las dependencias de `coefficients_gpu`

Seguí `GpuCoefficientBufferBrain.coefficients_gpu()` hasta `gpu_coefficient_layout.assemble`. Este último conserva la cadena de reemplazos del operador, no representa una única multiplicación CSR seguida de una dinámica homogénea. 

| Dependencia examinada | Qué se puede reutilizar | Qué no se puede congelar silenciosamente |
|---|---|---|
| **Puertos físicos \(q,s\)** | Su respuesta analítica para una historia provisional o confirmada dada. | La historia emitida por el organismo ni sus efectos sobre los receptores. |
| **Salidas especializadas PN y entradas externas** | El adaptador mantiene sus buffers durante cada intercambio declarado. | No son constantes para toda la simulación; deben renovarse en la frontera correspondiente. |
| **CSR base** | Es lineal respecto de las transmisiones antes de aplicar la respuesta postsináptica. | La respuesta posterior: `tanh`, conductancias visuales y sus dependencias de estado. |
| **Recursos ORN y receptores PN→KC** | Geometría, mapas y parámetros constantes. | Recursos y gates: sus coeficientes se recalculan desde el estado de la etapa. |
| **APL, axón regional, adaptación y retina** | Algunas coordenadas están mantenidas por otro propietario; algunos términos admiten estructura local. | Otras coordenadas siguen evolucionando. No puede tratarse toda la región como una entrada fija. |

Estas distinciones aparecen en el adaptador y en la cadena generada. En particular, los recursos ORN utilizan `state[rows]*caps` en cada etapa; el receptor PN→KC tiene activación y relajación dependientes de la actividad presináptica. **Que exista un buffer `held_rate` no significa que siga siendo la fuente activa de todas las rutas olfativas.**     

### La separación algebraica aprovechable está antes de la no linealidad

Para cada suma de entrada compatible, se puede buscar una representación:

\[
I(t)=A_{\mathrm{din}}\,s_{\mathrm{din}}(t)
+A_{\mathrm{puerto}}\,p(t)
+I_{\mathrm{mantenido}}.
\]

Es una separación de **contribuciones**, no de neuronas independientes. Las matrices deben incorporar las rutas efectivas y sus fracciones, no solamente la `W` original.

El kernel base muestra por qué la precisión del puerto no basta. En las células no visuales, la suma entra en una `tanh` rectificada. En las visuales, la ecuación base equivale a:

\[
\dot v=\frac{0,25+g_e-(1+g_e+g_i)v}{\tau}.
\]

Aunque \(g_e(t)\) y \(g_i(t)\) se conozcan exactamente, aparece su producto con el voltaje evolutivo. **No se puede sumar una “respuesta al evento” independiente del estado sin justificar esa aproximación.** 

Además, un salto en \(q\) no implica necesariamente un salto en todas las derivadas receptoras: el filtro \(s\) es continuo. Conviene distinguir consumidores directos de \(q\), consumidores de \(s\) y consumidores de estados regionales. Esa distinción puede permitir menos cortes globales, pero tiene que obtenerse del operador final, después de sus reemplazos.

### Una restricción de implementación importante para B

La cadena actual modifica temporalmente pesos PN y utiliza vistas con transmisiones sustituidas por uno; las rutas APL también dependen de un buffer efectivo de pesos. **No ejecutaría llamadas parciales concurrentes a esa cadena suponiendo que es una función pura.**

Antes de paralelizar sus evaluaciones, B necesita representar esas contribuciones como datos inmutables de la época o como almacenamiento privado. De lo contrario, dos evaluaciones podrían interferir mediante el mismo buffer, aunque sus estados de entrada fueran distintos. Esto es una dependencia del diseño leído, no un fallo concurrente que haya reproducido.  

## 2. Las tres alternativas y qué pregunta responde cada una

### A. Esquema actual con propuesta nominal separada del corte

**Pregunta:** ¿cuánto trabajo se pierde al reconstruir lentamente el paso después de cada frontera?

La política nominal puede evitar que un evento reduzca artificialmente las propuestas posteriores. Sin embargo, conserva los cortes y las seis evaluaciones por intento aceptado. El resultado de 10→9 pasos pertenece al fixture; no mide el organismo.  

La conservaría como **control de bajo coste de ingeniería**. No dedicaría otra ronda a variantes sucesivas del factor de crecimiento. Su falsador sigue siendo un evento tardío seguido de una respuesta rápida: recuperar la propuesta nominal nunca debe saltarse la frontera ni aceptar un error que el control rechazaría.

### B. Formas de onda de entradas + evolución local con corrección recurrente

**Ésta es mi apuesta principal, todavía como hipótesis.**

Durante el intercambio ya declarado, B representaría las entradas como funciones temporales. Los puertos admitidos aportarían sus respuestas analíticas; las contribuciones recurrentes desconocidas se aproximarían a partir de una trayectoria provisional. Los mecanismos locales evolucionarían con esas entradas y después se **recalcularían las contribuciones recurrentes y se corregiría la trayectoria**.

La aceptación debe depender de la consistencia del acoplamiento y del error temporal, no solo de que cada célula complete su integración. No se publican eventos o estados de una iteración descartada.

El cambio de trabajo buscado es:

> Pasar de recorrer globalmente las conexiones por cada corte local a recorrerlas en un conjunto menor de evaluaciones de la trayectoria, conservando el efecto temporal de los eventos en sus consumidores.

Esto no exige que el grafo tenga componentes independientes. Sí exige que la corrección converja con pocas iteraciones; si requiere tantos recorridos globales como el método actual, no hay mejora útil.

La relajación de formas de onda tiene precedentes en simulación neuronal con acoplamiento eléctrico. Pero esos estudios muestran también que reducir comunicaciones puede aumentar las iteraciones y que la ventaja depende de la plataforma; **no proporcionan una aceleración transferible a una sola GPU de MATRIX**. :chatgpt-content-reference{index="13"}

**Falsador de B:** un circuito recurrente con dos entradas temporales de igual integral y distinto orden. La implementación debe preservar la diferencia de respuesta y el efecto de retorno. Si solo pasa al mantener fija la entrada recurrente, ha eliminado realimentación, no acelerado el mismo sistema.

### C. Backend general de orden alto, sin cambiar el operador

Como segundo prototipo reutilizaría **ERK8**, ya explorado en vuestro núcleo genérico, aplicado al RHS real:

\[
F(t,y)=r(t,\widetilde y)\odot
\bigl(a(t,\widetilde y)-\widetilde y\bigr),
\qquad
\widetilde y=P(t,y),
\]

donde \(P\) proyecta los puertos analíticos. Las coordenadas propiedad de esos puertos conservan su tratamiento explícito; no se integran una segunda vez.

**Mantendría inicialmente las mismas fronteras obligatorias de eventos.** Así C pregunta si un método embebido distinto reduce el trabajo necesario entre fronteras, mientras B pregunta si puede reducir el número de evaluaciones globales impuestas por ellas.

ERKStep admite masa identidad; ese alcance corresponde al sistema normalizado presentado al ejecutor CNS, **no a la PN con masa no diagonal**. PN, membranas espaciales y sus propietarios permanecerían intactos. Una extensión posterior a masas generales necesita un backend que las admita, no una conversión silenciosa. :chatgpt-content-reference{index="14"}

No asumiría que el orden alto gana: emplea más etapas y pierde la integración exacta del decaimiento congelado del esquema actual. Los pocos rechazos actuales tampoco prueban que ERK no vaya a quedar limitado por estabilidad. Las tablas oficiales permiten esta comparación; no garantizan su resultado. :chatgpt-content-reference{index="15"}

**Falsador de C:** mismo estado, operador e historias físicas, pero aparecen valores fuera de dominio, se incumple \(10^{-4}\) o se necesitan más evaluaciones y tiempo que A. No se rescata mediante `clip`, reducción de estados ni cambio de constantes.

## 3. El experimento discriminante que haría

**Un único contraste en dos niveles: primero aislar el trabajo CNS; después probar al sobreviviente con el organismo cerrado.** No construiría simultáneamente otra variante Rosenbrock, un nuevo solver de masa y un modelo reducido.

### Nivel 1: misma entrada física, recurrencia CNS libre

Elegiría, mediante una regla previa, **dos intercambios reales**: uno con mayor número de eventos y otro con pocos, dentro del tramo cuyo estado y procedencia se puedan reconstruir. Separaría predictor descartado e intercambio confirmado.

En A, B y C se mantienen idénticos:

- Estado inicial completo y operador efectivo.
- Historia de los puertos físicos y entradas mantenidas de esa época.
- Rutas especializadas, máscaras y parámetros.
- Instantes y campos del comparador.

**Lo que no se reproduce desde la referencia es la respuesta recurrente del CNS.** Cada candidato debe calcularla. Introducir los `target/rate` futuros de la referencia resolvería indirectamente el problema y anularía el contraste.

El resultado debe separar **número de recorridos dispersos, evaluaciones locales, correcciones de acoplamiento, rechazos y tiempo total**, incluyendo preparación de las representaciones temporales. No basta que un kernel aislado sea rápido.

Esta fase distinguiría:

| Resultado | Decisión |
|---|---|
| B reduce mucho los recorridos y conserva error/causalidad | Continuar B integrado. |
| C reduce trabajo sin necesidad de particionar | Preferir C: menor complejidad. |
| B necesita muchas correcciones y C queda limitado por estabilidad o fronteras | No hay salto demostrado; el dato orienta un backend implícito posterior, no autoriza anunciarlo ya. |
| Solo A nominal mejora modestamente | Conservarla si compensa, sin convertirla en solución de la meta. |

### Nivel 2: comparación acoplada y horizonte de segundos

Solo el candidato que supere la fase anterior pasa al organismo, con la guardia y propietarios comunes, sin reproducir entradas futuras del cerebro o cuerpo.

Primero lo contrastaría sobre el tramo refinado utilizable y sus transitorios; después realizaría **una ejecución continua con objetivo de un segundo y presupuesto fijo**, no otra colección indefinida de extremos cortos. El resultado puede ser una trayectoria parcial por coste. Eso informa rendimiento, no éxito de etapa 3.

**Presupuesto propuesto para una ronda nueva:** dos prototipos, ocho replays CNS como máximo y hasta cuatro cargas del organismo, con techo agregado de 5.400 s. Incluye fallos; no se amplía para buscar PASS. A nominal ya existe y funciona como control, no como tercer desarrollo.

Para llamar a esto un salto de arquitectura, fijaría prospectivamente una mejora de al menos **2× en avance completo** bajo el mismo contrato, no solo ahorro de llamadas. Ese umbral es una decisión de coste/beneficio de la ronda; **no equivale a alcanzar 1 s/60 s**.

## 4. Qué información falta para implementar B sin inventar equivalencia

Las fuentes bastan para identificar las clases de dependencia. **No bastan para certificar qué rutas están activas, sus intersecciones y el volumen de trabajo separable en la instancia ejecutada.**

No pediría otro dataset anatómico completo. Falta concretar un **contrato del operador efectivo por época**, con referencias a los arrays locales existentes:

**Mapa ordenado de coordenadas y propietarios; rutas PN/APL efectivas; `mode/fraction/slot` de reemplazos regionales; máscaras y mecanismos habilitados; estado inicial; puertos y buffers mantenidos; y método/clase de `coefficients_gpu` realmente instalados.**

También importa qué operaciones sobrescriben una salida anterior y cuáles la utilizan para modificarla. La cadena generada calcula el operador base y después reemplaza determinados resultados. Eliminar trabajo muerto podría ser válido; eliminar un resultado que una capa posterior necesita no lo sería.  

**No afirmo que esos datos no existan en el ZIP:** no he reconstruido ese contrato activo desde sus arrays. Antes de programar la partición, lo produciría localmente desde el propietario real, no a partir de etiquetas de neuronas.

Para extender el motor a otros cerebros, la selección debe basarse en primitivas matemáticas —suma dispersa, cascada afín, reacción local, conductancia, masa y puerto temporal—. La aplicación actual puede aportar adaptadores anatómicos; **el scheduler no debería seleccionar algoritmos por los nombres PN o KC**.

## 5. El fallo de dominio debe investigarse en paralelo, no confundirse con el rediseño

La captura añadida conserva índice, valor y reloj del endpoint intentado antes del rollback. Es útil y no cambia por sí misma las aceptaciones exitosas. **No recupera el valor perdido de la corrida que falló durante el milisegundo 15.** 

Hay un discriminador matemático sencillo. Para una actualización:

\[
z_{\mathrm{nuevo}}
=z+\bigl(1-e^{-hr}\bigr)(a-z),
\]

si \(h\ge0\), \(r\ge0\) y \(z,a\in[0,1]\), el resultado pertenece a \([0,1]\) en aritmética exacta. Por tanto, para el índice ofensivo interesa conservar **el valor anterior, target, rate y valores antes/después de proyectar el puerto**.

Eso distingue un target/rate inválido, una proyección fuera de contrato y un posible efecto aritmético. **Sin esos valores no concluiría cuál ocurrió.** No cambiaría el límite ni insertaría un clamp.

El diagnóstico puede cerrarse sobre una llamada CNS capturada si existe una frontera reproducible anterior. Eso no rehabilita una sesión corporal fallida. Mientras tanto, B/C pueden investigarse en estados válidos; simplemente no se utiliza una referencia incompleta como certificado de 20 ms.

## Dictamen

**La dirección con más potencial es reducir cuántas veces se reconstruye la entrada global, no seguir abaratando únicamente el bucle que lo solicita.** B ataca esa estructura; C comprueba si un integrador general más eficaz consigue suficiente ahorro sin asumir el coste de una partición iterativa. A nominal sirve para medir cuánto del problema era solo recuperación del paso.

Los datos actuales justifican conservar la guardia por su mejora numérica observada, **no aprobar el motor ni extrapolar velocidad de tiempo real**. La reforma solo merece permanecer si conserva la respuesta temporal y mejora el coste del organismo, incluidos los componentes que el nuevo backend no acelera.

**Archivos leídos:** los nueve prioritarios solicitados; completos `coefficient_buffer_brain.py`, `gpu_coefficient_layout.py`, `orn_peripheral_terminal_brain.py` y `olfactory_endogenous_brain.py`; el kernel y los métodos de coeficientes de `gpu_visual_brain.py`; y las secciones pertinentes de `kc_apl_dynamic_brain.py`, `pnkc_receptor_brain.py` y `kcgamma_regional_brain.py`. Consulté además documentación primaria de ARKODE y el trabajo de relajación de formas de onda citado. **No revisé exhaustivamente las 700 piezas ni ejecuté estas propuestas.**
