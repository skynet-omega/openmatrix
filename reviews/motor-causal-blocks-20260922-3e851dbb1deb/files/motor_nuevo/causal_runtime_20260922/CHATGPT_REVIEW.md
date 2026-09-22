**Conservaría A como optimización provisional y mantendría descartada la B ejecutada. Para la siguiente ronda priorizaría momentos de puertos con control del error frente a un Jacobiano realmente acoplado, usando una sesión residente común.** Esa sesión es infraestructura de ambas alternativas, no un tercer método numérico.

Encontré una **dependencia concreta en la recuperación que revisaría antes de atribuir la divergencia a CUDA**: las modificaciones efectivas de `tau/theta` no están representadas en el estado neural que se compara.

**Revisión documental:** leí las fuentes y recibos indicados abajo. No ejecuté el organismo, kernels, verificador ni arrays nuevos; tampoco comprobé independientemente el manifiesto de 593 archivos.

## 1. Recuperación: puede restaurarse el estado y perderse parte del operador

La cadena leída muestra lo siguiente:

**Preparación:** `preparar_candidata()` modifica `h.tau`, `h.rate_theta` y sus copias CUDA, pero no modifica los correspondientes `h.brain.tau_s` y `h.brain.theta`. 

**Serialización:** `HybridVisualBrain.state_dict()` guarda estado, parámetros generales, reloj y controlador; no guarda esos dos arrays efectivos. **Reconstrucción:** `_build()` vuelve a obtenerlos desde `brain.tau_s` y `brain.theta`; `GpuVisualBrain._build()` crea después sus copias CUDA.  

**Rollback:** `_restore_joint()` reconstruye el padre mediante `from_state()`, borra el diccionario del objeto y lo reemplaza por el reconstruido. La reinstalación del ensayo de recuperación instala ejecutores, pero no vuelve a aplicar la preparación registrada.  

No es una posibilidad puramente abstracta: el recibo de `recovery_04` contiene cambios efectivos. Para las PN, por ejemplo, las constantes originales son aproximadamente **0,0190513 y 0,0152035 s**, reemplazadas por **0,0171274 s**; los umbrales también cambian. 

**Conclusión:** la igualdad del estado serializado no prueba igualdad de la función de evolución utilizada después. Este mecanismo puede explicar una continuación distinta; **no he demostrado que explique toda la discrepancia de `recovery_04`**.

### Comprobación inmediata

Antes de avanzar otra vez, comparar literalmente, antes del fallo y después de reconstruir:

`h.tau`, `h.rate_theta`, `h.cuda['tau']`, `h.cuda['theta']`.

Extendería la comprobación a los pesos efectivos modificados fuera del propietario estático. La solución no es reajustarlos: es **preservar las intervenciones ya declaradas como parte del operador efectivo de la época**, con valores e identidad propios. No recalcular medias ni volver a multiplicar pesos que puedan estar intervenidos.

También separaría dos contrastes: **reconstrucción fría sin fallo** frente al control continuo, y **fallo–rollback–reconstrucción** frente a esa reconstrucción fría. Así se distingue pérdida al cargar de contaminación causada por el intento rechazado.

Mantendría bloqueada la sesión corporal fallida. El diagnóstico publicado acredita diferencias posteriores, no recuperación segura. 

## 2. Cota de bases: estructura correcta, certificación aritmética todavía incompleta

**No encontré que la agrupación mezcle incorrectamente los potenciales de inversión.** `compile_basis()` agrupa conjuntamente matriz y vector; `compile_warp()` calcula por separado el coeficiente matricial y el coeficiente del término independiente ponderado por cada inversión. También incluye en la segunda etapa el efecto de la diferencia del operador sobre el voltaje anterior. 

La desigualdad que sustenta el enfoque es correcta. Si

\[
\|\Delta A\|_\infty\le d_A,\qquad
\|\Delta b\|_\infty\le d_b,
\]

entonces, para la solución calculada \(\widetilde v\),

\[
\|b-A\widetilde v\|_\infty
\le
\|\widetilde b-\widetilde A\widetilde v\|_\infty
+d_b+d_A\|\widetilde v\|_\infty.
\]

El denominador reducido utilizado por el código también sigue la dirección conservadora correspondiente, **si las cotas suministradas son válidas**.

**La carencia está en esa última condición:** el margen `256*eps` se justifica mediante el número de términos, pero no viene acompañado de una derivación completa para todos los redondeos del ensamblado y la evaluación del residuo. En particular, inicializar márgenes con el valor absoluto de una suma ya formada no sustituye, bajo cancelación, una cota basada en las magnitudes de sus sumandos. No afirmo haber encontrado una violación en la geometría real. 

`check_basis.py` compara operadores y soluciones con tolerancias, pero **no comprueba independientemente que el residuo original quede por debajo de la envolvente calculada**. Los pequeños errores publicados respaldan la equivalencia observada; no completan esa prueba.  

**Acción acotada:** reconstruir ambas etapas originales en precisión superior o con redondeo dirigido, y contrastar la desigualdad anterior usando entradas reales más extremos admisibles y cancelaciones. Mantendría la expresión “cota implementada” hasta cerrar eso, no “certificación general”.

## 3. Primer rival: momentos ponderados por la respuesta del receptor

La alternativa prometedora no es promediar eventos, sino **integrar su contribución temporal sin obligar a todas las variables a detenerse en cada marca**.

Para una dependencia lineal aditiva congelada:

\[
\dot z=Lz+Bu(t),
\]

la contribución correcta del puerto es

\[
\int_0^h e^{L(h-s)}B\,u(s)\,ds.
\]

Los filtros exponenciales y sus cascadas permiten calcular esa contribución mediante expresiones analíticas o estados auxiliares pequeños. Los métodos de integración de eventos publicados explotan precisamente estructuras de decaimiento exponencial, con condiciones específicas; no validan automáticamente receptores no lineales arbitrarios. :chatgpt-content-reference{index="11"}

**La cota debe cubrir la respuesta, no solo la diferencia entre dos promedios.** Por ejemplo, si se aproxima el kernel receptor \(K(s)\) mediante un polinomio \(P_m(s)\),

\[
\left\|\int_0^h[K(s)-P_m(s)]u(s)\,ds\right\|
\le
\sup_s\|K(s)-P_m(s)\|
\int_0^h|u(s)|\,ds.
\]

Los momentos de \(u\) pueden calcularse conservando todas las marcas y amplitudes. Esa desigualdad cubre **el error de aproximar el kernel lineal**; aún faltaría presupuestar el error de congelar coeficientes y el del acoplamiento recurrente.

### Falsador que conservaría

Dos pulsos de conductancia, con inversiones \(E_1\) y \(E_2\), aplicados en orden opuesto. Si cada pulso produce la transformación

\[
v\mapsto \alpha v+(1-\alpha)E,
\]

las respuestas finales difieren en

\[
(1-\alpha)^2(E_2-E_1).
\]

Las integrales de conductancia por receptor son iguales, pero el resultado no. **Un esquema que conserva esas integrales y pierde el orden no pasa**, aunque corrija el ejemplo de evento tardío.

Además, los momentos deben poder evaluarse para cualquier prefijo solicitado. Las historias generadas por un predictor siguen siendo provisionales: si cambia la respuesta recurrente o se rechaza el bloque, se descartan sus eventos y momentos. No se puede tratar como entrada física confirmada una historia calculada con un acoplamiento aún no aceptado.

**La alineación actual queda como recuperación numérica y control**, no como paso que pueda eliminarse sin sustituto verificable. Su mayor coste está medido en esta ronda.  

## 4. Segundo rival: Jacobiano voltaje–compuertas realmente acoplado

`ros_step.cu` declara y ejecuta bloques cruzados nulos entre voltajes y compuertas. Por ello, el negativo actual descarta esa implementación, **no la variante acoplada propuesta**. 

Para las mismas ecuaciones, una etapa linealizada tendría:

\[
W=
\begin{pmatrix}
M_v-\gamma hF_v &-\gamma hF_g\\
-\gamma hQ_v&I-\gamma hQ_g
\end{pmatrix}.
\]

Con \(D=I-\gamma hQ_g\), la eliminación algebraica de las correcciones de compuertas da:

\[
S=M_v-\gamma hF_v-(\gamma h)^2F_gD^{-1}Q_v.
\]

**Se conservan todas las compuertas como estados.** Solo se elimina temporalmente una parte del sistema lineal de etapa. Si las cinéticas declaradas hacen diagonal a \(D\), eso es aprovechable; no debe imponerse a mecanismos futuros.

El falsador previo a una corrida costosa es comparar el producto y la solución del sistema completo con su Schur, incluyendo ambos bloques cruzados y los términos de inversión. Después, medir si disminuyen los pasos rechazados o las factorizaciones suficientemente para compensar el Jacobiano.

**Un Jacobiano mejor no corrige por sí solo el evento tardío.** Esta alternativa debe conservar la alineación o un tratamiento de puertos ya validado. La documentación de RA34PW2 admite Jacobianos aproximados, pero eso no garantiza eficiencia ni fidelidad de cualquier aproximación sobre este sistema híbrido. :chatgpt-content-reference{index="15"}

## Decisión para la siguiente ronda

**La sesión residente única debe ser común a los dos rivales y conservar operador efectivo, estado, eventos provisionales, historias y confirmación transaccional.** Primero resolvería la pérdida potencial de `tau/theta`; mantener propietarios residentes no basta si el contrato sigue omitiendo parámetros que cambian las ecuaciones.

Después compararía, sin combinarlos inicialmente:

**Momentos de puertos + método temporal conservado**, frente a **Jacobiano acoplado + tratamiento de eventos correcto**. La agrupación de bases puede mantenerse en ambos si supera su comprobación específica.

No reutilizaría como referencia de corrección el método rápido que comparte el defecto temporal. Tampoco extrapolaría los 5 ms a un segundo: las pruebas pequeñas deciden si la transformación es admisible; la decisión de rendimiento exige finalmente una trayectoria continua del organismo, con el mismo contrato de error.

**Lectura realizada:** `REVIEW_REQUEST.md`, `RESULTADOS.md`; compilador, comprobaciones y recibo de bases; `native_cell.py`, `cell_control.cpp`, `physical_events.cu`, `graph_control_v2.cpp`, `ros_step.cu`; guardia, prueba y diagnóstico de recuperación; y las capas pertinentes de serialización, reconstrucción y preparación en `legacy_sources`. Consulté las secciones de ejecuciones y comparaciones de `VERIFIED.json`, no lo reconstruí desde arrays. **No revisé exhaustivamente las 182 fuentes legadas ni ejecuté estas propuestas.**

**El hallazgo prioritario es cerrar la identidad del operador restaurado. La apuesta numérica prioritaria es integrar el efecto de los eventos con una cota, no simplemente ejecutar más rápido el mismo muestreo que puede omitirlos.**
