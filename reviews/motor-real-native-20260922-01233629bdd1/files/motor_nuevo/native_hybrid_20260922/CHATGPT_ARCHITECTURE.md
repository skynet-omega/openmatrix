**Priorizaría dos implementaciones: A, compilar la redundancia geométrica conservando el método actual; y B, un Rosenbrock-W acoplado que reutilice una factorización por paso.** Mantendría C —operador pasivo más actualizaciones de rango bajo— como rival condicionado a demostrar un rango realmente pequeño de las actualizaciones. **El rango 15 del conjunto de bases no demuestra esa condición.**

Releí `kc_fused_warp.py`, `kc_adaptive.py` y `kc_projected_batch.py` del commit `6187361…`. **No examiné las fuentes nuevas ni ejecuté sus prototipos.** El perfil nuevo, la mejora de 1,16× y la proporcionalidad/rango de las bases son datos locales que comunicas; las propuestas siguientes combinan esa evidencia con análisis matemático y fuentes externas.

## 1. Qué permite concluir el perfil, sin prometer una aceleración

Con tus tiempos, membranas/eventos espaciales representan aproximadamente el **67,3 %** del paso. Si solamente aceleramos ese bloque, conservando el resto:

\[
T_{\mathrm{nuevo}}
=0,923+\frac{1,897}{S_{\mathrm{membranas}}}
\quad \text{segundos reales por milisegundo simulado}.
\]

Incluso hacerlo gratuito daría un máximo aritmético de **3,06× total**, bajo ese régimen y sin cambios en el resto. No es una predicción del rendimiento sostenido: muestra por qué ninguna optimización exclusivamente KC puede justificar por sí sola la meta global.

El código publicado sí proporciona una oportunidad concreta: **cada ensayo adaptativo calcula un paso completo y dos medios pasos; cada uno ensambla y resuelve dos sistemas eléctricos**. Son seis ensamblados/factorizaciones pequeñas por intento, además de compuertas, residuos y eventos. El perfil agregado todavía no dice qué proporción corresponde a cada operación.  

| Alternativa | Trabajo que intenta eliminar | Decisión |
|---|---|---|
| **A. Compilación estructural de operadores** | Repetición de geometría durante ensamblado y aplicación. | **Implementar primero.** |
| **B. Rosenbrock-W acoplado con eliminación algebraica de compuertas** | Factorizaciones repetidas y duplicación de paso como único estimador. | **Segundo prototipo, separado de A.** |
| **C. Operador pasivo reutilizable + actualizaciones de rango bajo** | Refactorización completa cuando cambia una perturbación pequeña. | Solo si el rango efectivo y la reutilización justifican su coste. |

## 2. A: compilar geometría repetida, no comprimir la fisiología

### La representación que propondría

El ensamblado original distingue una contribución al operador y otra al término independiente:

\[
K(g)=G_0+\sum_{p,c} f_{pc}(g)\,G_{pc},
\]

\[
b(g)=b_{\mathrm{externo}}
+\sum_{p,c} f_{pc}(g)\,E_{pc}\,b_{pc}.
\]

En el código, los factores de canal son \(m^3h\), \(p\) y \(n^4\); sus potenciales de inversión no son todos iguales. Esa distinción debe sobrevivir a cualquier agrupación.  

Si verificas:

\[
G_{pc}=a_{pc}\widehat G_p,\qquad
b_{pc}=a_{pc}\widehat b_p,
\]

puedes calcular **dos coeficientes**, no uno:

\[
\alpha_p=\sum_c a_{pc}f_{pc},
\qquad
\beta_p=\sum_c a_{pc}f_{pc}E_{pc}.
\]

Entonces:

\[
K=G_0+\sum_p\alpha_p\widehat G_p,
\qquad
b=b_{\mathrm{externo}}+\sum_p\beta_p\widehat b_p.
\]

**Todas las compuertas permanecen.** Solo se evita repetir una misma geometría. No sustituiría \(\beta_p\) por un potencial de inversión fijo multiplicado por \(\alpha_p\), ni dividiría por \(\alpha_p\) cuando puede ser cero.

La reducción posterior a una base independiente debe comprobar por separado el operador y el término independiente. **Un rango calculado únicamente sobre las matrices \(G_{pc}\) no certifica que el par completo, con sus inversiones y fuentes, tenga el mismo rango.**

### «Proporcional al redondeo» no es identidad algebraica exacta

Hay que distinguir:

**Factorización derivada de la construcción geométrica:** reorganiza una expresión matemática conocida.

**Factorización obtenida truncando valores singulares pequeños:** aproxima los operadores almacenados.

La segunda puede ser aceptable dentro del presupuesto numérico, pero no debe anunciarse como exacta. Para una solución candidata \(\widetilde v\), el residuo respecto del sistema original satisface:

\[
b-A\widetilde v
=
(\widetilde b-\widetilde A\widetilde v)
+\Delta b-\Delta A\,\widetilde v.
\]

Esto permite conservar el criterio original mediante una evaluación directa o una cota verificable de los términos de compresión. **Evaluar solamente el residuo del sistema comprimido sería insuficiente.** Si certificar la cota cuesta más que el ahorro, la pieza no se conserva.

### Cómo convertirlo en infraestructura general

El IR debería describir **operadores estáticos, coeficientes dinámicos, restricciones e identidad de sus soportes**. El compilador decide si conviene ensamblar, aplicar de forma factorizada o especializar dimensiones. No necesita una rama llamada «KC».

La separación entre restricciones, bases, física puntual y aplicación del operador ya tiene un precedente útil en **libCEED**. Su documentación distingue preparación, ensamblado parcial y aplicación sin formar toda la matriz. Eso respalda el diseño, **no prueba que instalar libCEED acelere estos sistemas 17×17**. :chatgpt-content-reference{index="4"}

**Falsador de A:** una combinación admisible de compuertas, conductancias y polaridades pasa el residuo comprimido, pero incumple el residuo original o cambia eventos fuera del contrato. Incluiría conductancias nulas, extremos de compuertas y cancelaciones del término independiente, no solo estados de reposo.

**Estimación, no medición:** pasar de 51 a 15 términos reduce aproximadamente **3,4× el número de contribuciones de ese ensamblado**. No reduce 3,4× los solves, las compuertas, los eventos ni el organismo.

## 3. B: cambiar el trabajo por paso mediante Rosenbrock-W, conservando estados

Ésta es la alternativa con una posibilidad distinta de salto: **no ensamblar y factorizar repetidamente por cada evaluación del estimador**, sino reutilizar el operador lineal entre etapas de un método apropiado.

Un candidato concreto es **Rosenbrock-W RA34PW2**, de cuatro etapas, orden tres y estimador embebido de orden dos. PETSc documenta su estabilidad y el uso de un Jacobiano aproximado recalculado normalmente una vez por paso. No basta congelar arbitrariamente el Jacobiano dentro del método actual: deben utilizarse sus ecuaciones de etapa y coeficientes correspondientes. :chatgpt-content-reference{index="5"}

### La oportunidad específica: 85 estados no requieren necesariamente un solve denso de 85×85

Con voltajes \(v\) y compuertas \(g\):

\[
M_v\dot v=F(v,g),\qquad \dot g=Q(v,g).
\]

Una etapa linealizada contiene:

\[
\begin{pmatrix}
M_v-h\gamma J_{vv} & -h\gamma J_{vg}\\
-h\gamma J_{gv} & I-h\gamma J_{gg}
\end{pmatrix}
\begin{pmatrix}\delta v\\\delta g\end{pmatrix}
=
\begin{pmatrix}r_v\\r_g\end{pmatrix}.
\]

Las compuertas del modelo publicado evolucionan independientemente entre sí **condicionadas al voltaje**, de modo que \(J_{gg}\) es diagonal en esa representación. Esto se desprende de las funciones de tasas y de la actualización de cada compuerta. 

Definiendo \(D=I-h\gamma J_{gg}\), se pueden eliminar **las correcciones lineales** de las compuertas:

\[
S=M_v-h\gamma J_{vv}
-(h\gamma)^2J_{vg}D^{-1}J_{gv},
\]

\[
S\delta v=r_v+h\gamma J_{vg}D^{-1}r_g,
\]

\[
\delta g=D^{-1}(r_g+h\gamma J_{gv}\delta v).
\]

**No se eliminan las compuertas del estado ni se las pone en equilibrio instantáneo.** Se resuelve exactamente el sistema lineal de etapa mediante su complemento de Schur. Si otro mecanismo tiene transiciones entre compuertas o química acoplada, \(D\) será un bloque distinto: el compilador debe descubrir esa estructura, no imponer diagonalidad.

Con un tableau de diagonal común, la factorización de \(S\) puede reutilizarse entre sus etapas. Frente a las seis factorizaciones actuales, la candidata puede utilizar **una factorización y varias sustituciones**, pagando a cambio el Jacobiano, los términos cruzados y las evaluaciones de las etapas.

Eso es un cambio potencialmente importante de trabajo, **no una aceleración medida ni una garantía de aceptar pasos mayores**.

### Condiciones que no omitiría

El método debe integrar las mismas dependencias entre voltaje y compuertas. Omitir \(J_{vg}\) o \(J_{gv}\) sin usar una aproximación admitida por el método cambia su comportamiento numérico.

También debe tratar correctamente la dependencia temporal de los puertos. La implementación `TSROSW` de PETSc declara actualmente soporte para sistemas autónomos; no la conectaría a entradas temporales ignorando los términos correspondientes. Puede utilizarse una formulación adecuada o una transformación explícita, pero debe quedar en el contrato. :chatgpt-content-reference{index="7"}

**Falsador de B:** reducir solves y cumplir el error de voltaje, pero cambiar compuertas, emisión de eventos o historia retardada por encima del criterio. Otro negativo material sería que el coste del Jacobiano y la resolución acoplada eliminen el ahorro de factorizaciones.

## 4. C: operador pasivo más actualizaciones de rango bajo

La estructura buscada sería:

\[
A=A_{\mathrm{pasivo}}+UV^{T},
\]

con \(r=\text{número de columnas de }U\) **mucho menor que la dimensión del sistema**. Woodbury permite resolver usando la factorización pasiva y un sistema pequeño de dimensión \(r\), sin formar inversas explícitas. :chatgpt-content-reference{index="8"}

**Pero dimensión del espacio de bases y rango matricial son conceptos distintos.** El conjunto formado únicamente por \(I_{17}\) tiene dimensión uno; esa matriz sigue teniendo rango 17. Por tanto:

> **“Las 51 bases tienen rango 15” no demuestra que la actualización del sistema 17×17 sea una perturbación barata de rango bajo.**

Hay que medir el rango de las actualizaciones y comprobar si existe un subespacio común reutilizable. Si hacen falta 15 direcciones para un sistema de 17, los productos, almacenamiento y sistema reducido pueden costar más que factorizar directamente.

La reutilización también depende de qué permanece constante: paso temporal, masa, geometría y parte pasiva. **No se comparte una factorización entre células que solo se parecen anatómicamente ni después de cambiar sus coeficientes.**

**Falsador de C:** el rango necesario para cumplir el residuo original deja de ser pequeño, la corrección pequeña se vuelve mal condicionada o el coste completo supera A. En ese caso conservaría el solve directo y cerraría esta ruta para ese régimen.

### Dónde dejaría la precisión mixta

No la escogería como otro prototipo completo de esta ronda. Podría evaluarse posteriormente **solo dentro de la resolución lineal**: factores de menor precisión, residuo calculado contra \(A,b\) originales en FP64 y refinamiento con recuperación FP64.

Es una estrategia establecida, pero su utilidad depende de condicionamiento, convergencia y costes; no del cociente nominal de TFLOPS. En 17×17, conversiones y refinamientos pueden anular la ganancia. **Gates, estados, eventos y criterio final continuarían en FP64.** :chatgpt-content-reference{index="9"}

## 5. Eventos: el contrato común que decide si la aceleración sirve

Aquí las dos implementaciones tienen distinto riesgo.

**A conserva el método temporal.** Debe mantener instantes de evaluación y confirmación; una diferencia de redondeo puede aun alterar una decisión cercana al criterio de evento, y eso se mide.

**B cambia el método temporal.** No puede sustituir silenciosamente el detector causal de máximos por un umbral, ni emitir eventos desde etapas provisionales. El código heredado utiliza voltaje previo, pendiente previa, valle, conteos y clipping al confirmar los medios pasos. Esos campos forman parte del estado, no son telemetría prescindible. 

Exigiría que B proporcione la información requerida en los instantes del propietario de eventos y contraste la historia resultante. Si necesita cambiar la localización temporal del evento, eso debe evaluarse como modificación numérica explícita. **No basta igualdad de conteos finales ni un residuo eléctrico pequeño.**

El evento tardío que estás reparando debe permanecer como prueba negativa común: un estimador que compara dos aproximaciones que omiten el mismo salto no puede autorizar un paso por tener diferencia cero. Las reparaciones de rollback, propietarios y presupuestos también deben entrar en la referencia común, antes de comparar velocidad.

## Recomendación para esta ronda

**Implementaría A y B por separado; no una mezcla simultánea de compresión, precisión reducida y cambio temporal.** A pregunta cuánto sobra en el ensamblado. B pregunta cuánto sobra en la organización matemática del paso. C queda pendiente de su prueba de rango y reutilización, sin gastar todavía otra implementación completa.

Las pruebas pequeñas sirven para falsar cada transformación. **La decisión de conservarla se toma sobre segundos continuos del organismo conservado**, incluyendo compilación/preparación por separado, intentos rechazados, residuos, eventos, retardos y coste completo. Si el presupuesto corta la trayectoria, se informa como tal.

La generalización útil no es declarar que todas las neuronas se resuelven igual. Es que el motor reconozca **estructura afín, bloques de dependencia, masa y actualizaciones** desde las ecuaciones, y utilice una especialización solo cuando sus condiciones están demostradas. Así el trabajo sobre estas membranas puede producir primitivas reutilizables, en lugar de otro solver seleccionado por el nombre «KC».
