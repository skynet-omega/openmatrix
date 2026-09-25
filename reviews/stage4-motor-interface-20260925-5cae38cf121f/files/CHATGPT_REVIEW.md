# Respuesta externa recibida — campaña36 y próxima decisión

Mensaje cf9fe063-7fb5-43d4-8596-c896a06b6e4a, conversación Agente ChatGPT. Propuesta externa; el revisor declara lectura de fuentes y tests sintéticos, sin ejecutar nuestros NPZ. Los enlaces sandbox de su entorno no son archivos locales.

## Decisión: mantengo la prioridad del lector, sin declararlo culpable

**Mantengo A —identificación del lector con datos independientes compatibles— como prioridad de preparación.** El resultado no demuestra que el lector sea la causa dominante ni que una ganancia mayor resuelva el problema.

Cambiaría únicamente la siguiente operación: **antes de invertir en otra simulación, calcularía una cota algebraica para toda una familia de lectores estáticos sobre las señales ya registradas**, sin seleccionar parámetros. Conservaría B y C como alternativas sustantivas, no como controles subordinados a una explicación ya elegida.

La clasificación sigue siendo **`PROMETEDOR_NO_CONFIRMADO`**: el efecto descrito favorece conservar el contraste original en esta implementación discretizada, pero ambos brazos incumplen la mejora histórica y falta la referencia refinada de ambos brazos a 1 s. Eso está explícito en los archivos consultados.  

## 1. Qué permite concluir el resultado, y qué corregiría

La pareja es una **intervención sobre las entradas**, no simplemente una correlación entre actividad y conducta. Bajo la identidad inicial documentada, permite interpretar un efecto total de igualar las antenas **dentro del modelo ejecutado**. No permite repartir ese efecto entre transformación sensorial, historia neural, lector y dinámica corporal. Tampoco identifica utilidad del feedback olfativo espacial, porque ambos brazos consumen entradas grabadas. 

Hay dos precisiones que afectan el siguiente diagnóstico.

### Cambiar la media neural no demuestra no linealidad

La advertencia relevante se confirmó: conservar la concentración media **no conserva necesariamente la actividad neural común**. Pero sería excesivo presentar el cambio observado como una demostración específica de no linealidad o adaptación.

Un ejemplo puramente matemático basta. Definiendo \(C=(L+R)/2\) y \(D=(L-R)/2\), un circuito lineal asimétrico podría producir:

\[
q_L=a_L(C+D),\qquad q_R=a_R(C-D).
\]

Entonces:

\[
\frac{q_L+q_R}{2}
=
\frac{a_L+a_R}{2}C
+
\frac{a_L-a_R}{2}D.
\]

Cuando \(a_L\neq a_R\), eliminar \(D\) cambia la media neural **sin ninguna no linealidad**. Por tanto, tus diferencias de medias son compatibles con asimetría lineal, transformación no lineal, historia o combinaciones de ellas. Además, **contraste antenal cero no equivale a contraste DN cero**.

### La media DNb05 no entra directamente en el lector archivado

El verificador utiliza:

\[
x(t)=[q_L(t)-b_L(t)]-[q_R(t)-b_R(t)],
\]

\[
u(t)=5\,\tanh[250x(t)]
\quad\text{en grados/s},
\]

con los canales DNb05 en las columnas NumPy **2 y 3 de `DN_q_usada`**. 

Una misma cantidad añadida a ambos canales se cancela exactamente en esa ecuación. Esto **no vuelve irrelevante la actividad común dentro de la red**: puede modificar posteriormente el contraste mediante las dinámicas neurales. Sí impide atribuir al cambio de media un efecto *directo* sobre este lector.

También conviene mantener separados los observables. Por simple resta de `MEASURES.json`, la diferencia entre mandos netos integrados es **+0,092261530° de mando**, mientras que la diferencia entre errores finales es **+0,128338101°**. No son dos estimaciones de una misma cantidad; su diferencia no identifica un torque, una pérdida de transmisión ni una causa corporal. 

## 2. Las tres alternativas que conservaría

### A. Identificación independiente del lector — prioridad principal

La pregunta es: **¿qué relación actividad→mando puede justificarse fuera de esta vida expuesta?**

Compararía primero el lector estático actual con **un único candidato dinámico parsimonioso**, solamente cuando haya datos capaces de distinguirlos. La dinámica adicional sería una hipótesis, no una mejora asumida. La comparación debe retener ensayos o sesiones completos, mantener la misma información disponible para ambos candidatos y evaluar predicciones antes de introducirlos en el organismo.

El requisito decisivo no es obtener cualquier registro neural: es disponer de observables compatibles y una correspondencia de unidades identificable. Una función desconocida de observación y una ganancia desconocida del lector pueden compensarse entre sí; un buen ajuste no separaría automáticamente ambas cosas.

La ficha de Zenodo consultada publica **código de adquisición y análisis**, no constituye por sí sola el conjunto de grabaciones necesario para esa identificación. :chatgpt-content-reference{index="5"} Mientras falten datos compatibles, A puede producir un protocolo y restricciones de identificabilidad, **no coeficientes fisiológicos**. No sustituiría automáticamente DNb05 por datos de DNa02 ni calibraría sobre el yaw de campaña 36.

**Discriminador:** predicción retenida bajo unidades y observación declaradas.  
**Resultado insuficiente:** rescatar solamente la trayectoria expuesta o añadir dinámica que no mejora la predicción independiente.

### B. Transformación e historia PN–DN — rival mecanístico real

Aquí separaría dos preguntas: si una aproximación afín predice la respuesta al contraste y si hace falta historia para predecirla. No son equivalentes.

Una condición futura económica de formular es el **contraste intermedio**:

\[
(C,D/2),
\]

manteniendo la historia de \(C\), el tercer canal y la preparación. Una hipótesis restringida, afín respecto de la amplitud del contraste, predice:

\[
\widehat y_{D/2}(t)
=
\frac{y_D(t)+y_0(t)}{2}.
\]

Esto proporciona una predicción temporal completa **sin ajustar parámetros nuevos**. Puede evaluarse en PN y DN antes del lector.

Es importante precisar su alcance: **un sistema lineal con memoria también satisface esa igualdad**. Un residuo material rechazaría esa hipótesis afín; no identificaría por sí solo adaptación, una sinapsis o un comparador. Un residuo pequeño tampoco demostraría ausencia de memoria.

Para localizar el rechazo en PN–DN habría que controlar las otras entradas y realimentaciones relevantes. De lo contrario, se estaría contrastando la respuesta del sistema completo. Tampoco trataría la media de un par PN como representación suficiente de todas las entradas de las DN.

**No ejecutaría ahora otro segundo completo para “demostrar no linealidad”.** Primero dejaría registrada la predicción y exigiría que el ensayo propuesto pueda cambiar una decisión concreta. Un eventual prefijo corto tendría su propio horizonte prospectivo; no certificaría el resto del segundo.

### C. Utilidad del feedback espacial — rival funcional

Esta alternativa pregunta algo distinto: **¿la entrada actualizada por la situación espacial ayuda a corregir una perturbación?**

El diseño útil sería online frente a replay con la **misma perturbación corporal**, preparación y resto de condiciones. Antes de ejecutarlo debe justificarse que la perturbación produce una separación observable entre las entradas olfativas que realmente consumen ambos brazos. Una comparación nominal con entradas prácticamente iguales sería poco informativa.

El resultado se evaluaría mediante recuperación del error y mantenimiento del soporte corporal durante un horizonte fijado previamente. La posición de la fuente y el error de rumbo permanecerían en el evaluador, no como información adicional del lector.

**Discriminador:** ventaja correctiva online después de una perturbación común, con separación sensorial comprobada.  
**Resultado negativo:** ausencia de ventaja en ese diseño y horizonte; no demostraría inutilidad universal del feedback.

Estas tres alternativas pueden coexistir. Una contribución sensorial beneficiosa puede ser insuficiente por una combinación de transformación neural, lector y realimentación.

## 3. El siguiente discriminador que haría ahora: una cota sin elegir ganancia

La exploración archivada ya muestra cancelación y evalúa varias ganancias. También advierte correctamente que el límite de ganancia infinita **no es una cota para todas las ganancias intermedias**. 

Ese punto puede cerrarse mejor, sin nuevas vidas.

### Pregunta restringida

> Con la señal DNb05 consumida congelada, ¿qué mando neto integrado podría producir cualquier lector estático, impar, monótono y limitado a ±5°/s?

Consideremos:

\[
u_t=5f(x_t),
\]

donde \(f\) es impar, no decreciente y \(|f|\leq1\). Esta familia incluye \(\tanh(gx)\) para **toda** \(g>0\), pero es más amplia.

Para cada amplitud observada \(a>0\), definimos:

\[
W(a)=\sum_t \Delta t_t\,
\operatorname{sign}(x_t)\,
\mathbf 1\{|x_t|\geq a\}.
\]

Entonces:

\[
5\min\!\left(0,\min_a W(a)\right)
\leq
\sum_t\Delta t_tu_t
\leq
5\max\!\left(0,\max_a W(a)\right).
\]

La razón es algebraica: sobre las amplitudes ordenadas, cualquier función monótona acotada puede expresarse mediante incrementos no negativos. Su integral queda como una combinación de las sumas de cola anteriores y cero. **No hace falta buscar una ganancia ni recomendar un umbral.**

Para comparar los brazos con **el mismo lector**, se utiliza \(W_{\mathrm{sinD}}(a)-W_D(a)\). No sería válido optimizar un lector diferente para cada brazo y presentar después esa diferencia como comparación emparejada.

### Qué discrimina y qué no

Si un requisito de **mando**, establecido independientemente, queda fuera de ese intervalo, esa familia de lectores no puede alcanzarlo **sobre esas señales congeladas**. Si queda dentro, únicamente permanece sin excluir.

No convierte la cota en una cota de yaw corporal. Tampoco excluye lo que ocurriría al cambiar el lector y regenerarse la actividad neural durante una nueva vida.

Incluso los conteos publicados permiten una cota exterior sencilla. Con 546 muestras negativas y 454 positivas, de 1 ms, cualquier salida que conserve el signo de \(x\) y respete ±5°/s cumple:

\[
-2{,}730^\circ
\leq
\int u\,dt
\leq
+2{,}270^\circ.
\]

Es una cota amplia de **mando**, derivada de esos conteos; no un resultado corporal ni una ganancia propuesta. La cota monótona calculada con las amplitudes reales puede ser más restrictiva. Los conteos proceden de `DECISION.md`; **no los recalculé desde los NPZ**. 

**Criterio de parada:** una pasada CPU. Si la cota no excluye ninguna explicación relevante, se archiva como no discriminante y se continúa con A. No abriría una campaña de búsquedas de ganancia sobre esta vida.

## 4. Código NumPy entregado

Preparé el diagnóstico usando el esquema y la ecuación del verificador archivado. **No lo ejecuté sobre tus trazas reales**, porque no conseguí descargar los binarios en este entorno.

El programa calcula las cotas individuales y la cota emparejada con un mismo lector. Además, genera la predicción de contraste intermedio para B, marcada explícitamente como **hipótesis, no traza observada**. Para predecir su mando aplica la `tanh` congelada a la actividad DN predicha: **no promedia los mandos**, operación que no sería equivalente.

:chatgpt-content-reference{index="10"}[Descargar código, pruebas y explicación — ZIP, 12 kB](sandbox:/mnt/data/diagnostico36_numpy.zip)

:chatgpt-content-reference{index="11"}[Descargar solamente `discriminador36.py`](sandbox:/mnt/data/campana36_diagnostico/discriminador36.py)

Colocando el script en la raíz de tu extracción:

```bash
python3 discriminador36.py \
  --control campanas/etapa4_contrast_ablation_20260925_36/identity_01/traces.npz \
  --sinD campanas/etapa4_contrast_ablation_20260925_36/no_contrast_01/traces.npz \
  --out diagnostico36_sin_ajuste
```

Produce `DIAGNOSTICO.json` y `PREDICCION_INTERMEDIA.npz` en un directorio nuevo. No modifica las entradas ni ejecuta el organismo. Registra hashes y señala si coinciden con los publicados para estos dos brazos. La predicción intermedia **no modifica el runner archivado ni habilita automáticamente una condición nueva**.

Ejecuté **11 pruebas sintéticas en modo normal y las mismas 11 con `-O`**, todas satisfactorias. Incluyen empates de amplitud, cotas frente a enumeración monótona, ganancias sintéticas, lector compartido, memoria lineal, no linealidades y rechazo de entradas incompatibles. La salida del diagnóstico sobre el caso sintético fue idéntica en ambos modos. Los registros están incluidos en el ZIP. Esto prueba propiedades del código nuevo, **no resultados del organismo**.

## 5. Qué leí y qué no ejecuté

**Leí completos**, en el commit indicado: `analysis_01/REPORT.md`, `analysis_01/MEASURES.json`, `DECISION.md`, `REPRODUCCION_CAPSULA.md`, `summarize_pair.py`, `chatgpt_verificar_cintas_cd.py`, `READOUT_DIAGNOSIS.json` y `ARCHIVE.json`. Consulté parcialmente el README del índice y las primeras 180 líneas de `run_replay.py`.

**No conseguí visualizar `COMPARISON.png` ni descargar el ZIP original o los tres NPZ.** No verifiqué sus hashes mediante descarga, no recalculé sus métricas crudas, no ejecuté tus verificadores, no repetí la igualdad de los 588 arrays y no ejecuté CUDA, MuJoCo, nuevas vidas ni referencias refinadas. La cápsula también declara que los checkpoints completos necesarios para repetir la igualdad de preparados no están incluidos. 

Externamente, consulté la ficha de Zenodo; el acceso al artículo PMC falló, por lo que no revalidé allí la disponibilidad de grabaciones a petición.

**Conclusión:** mantengo A como preparación principal; haría ahora la criba algebraica sin ajuste, conservaría B con una predicción explícita y mantendría C para una perturbación que realmente separe online y replay. Nada de ello justifica todavía cambiar ganancias, invertir signos, alargar la vida ni promover precisión numérica, equivalencia fisiológica o navegación.
