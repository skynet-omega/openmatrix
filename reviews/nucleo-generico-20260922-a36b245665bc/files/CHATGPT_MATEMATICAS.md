Revisión documental recibida el 22-09-2026, mensaje 46cfa5cc-476c-4013-b542-0fa9cef29db9. ChatGPT leyó ARQUITECTURA.md, no el núcleo nuevo. No ejecutó pruebas. PRO/máximo confirmado por usuario, selector no observable.

**Mi decisión es implementar A como siguiente backend general: integración implícita adaptativa con masa dispersa constante, ecuaciones/Jacobianos generados y estados residentes en GPU.** Mantendría el RK explícito actual para comprobar extensibilidad y problemas no rígidos; no lo convertiría en la única implementación del «modo rápido».

**B quedaría condicionado a una separación temporal y de acoplamiento demostrable. C, a que exista un operador lineal rígido reutilizable cuyo coste justifique las funciones de matrices.** No desarrollaría los tres simultáneamente.

Leí íntegramente `ARQUITECTURA.md` del commit indicado y contrasté documentación técnica y licencias. **No ejecuté código ni examiné el núcleo nuevo.** El documento anterior propone masa constante definida positiva y eventos explícitos; tu mensaje actual delimita una primera versión más estrecha, sin detección de eventos de umbral. Tomo esa delimitación actual como contrato, no anuncio capacidades pendientes. 

## 1. A concreta: ARKStep implícito, no otro solver anatómico

Para esta ampliación, fijaría inicialmente:

\[
M_\nu\dot x=f_\nu(t,x),
\]

con \(M_\nu\) **constante, dispersa y definida positiva durante cada intervalo entre mutaciones**. El índice \(\nu\) identifica la versión del modelo. Masa singular o dependiente del estado seguiría rechazada explícitamente.

La implementación que propongo como referencia es **SUNDIALS ARKStep**, completamente implícito, con **ESDIRK3(2)5L[2]SA** —identificador `ARKODE_ESDIRK325L2SA_5_2_3`—. Es una pareja de órdenes 3/2 con método y estimador embebido A/L-estables. Su elección es una primera candidata fundamentada para rigidez, **no una afirmación de superioridad empírica sobre vuestro motor**. :chatgpt-content-reference{index="1"}

### La ventaja matemática de conservar la masa en el residuo

En una etapa implícita:

\[
R_i(Y)=MY-b_i-h\gamma f(t_i,Y).
\]

Newton resuelve:

\[
\left(M-h\gamma J_f\right)\Delta=-R_i.
\]

Así se evita introducir una resolución con \(M\) dentro de **cada evaluación de la función no lineal**. ARKStep documenta precisamente esta formulación para masa constante no identidad. Eso no elimina todas las resoluciones de masa: siguen existiendo operaciones asociadas a actualización, estimación del error e interpolación. :chatgpt-content-reference{index="2"}

El compilador debería producir, desde la misma descripción:

**Evaluación de \(f\), producto \(J_fv\) y, cuando se necesite, valores y estructura del Jacobiano.** El patrón debe incluir las dependencias de los puertos y conexiones, no solamente los bloques locales de cada mecanismo.

Usaría **Newton–GMRES precondicionado** como camino general. Que \(M\) sea definida positiva **no hace que \(M-h\gamma J_f\) sea simétrica o definida positiva**: las reacciones y conexiones pueden romper ambas propiedades. CG solo sería una especialización habilitada por condiciones verificadas.

El precondicionador puede aprovechar bloques locales, sparsidad y agrupamientos del grafo. Pero las conexiones omitidas del precondicionador **permanecen en el operador y en el residuo completo**. Esa es la diferencia entre acelerar una resolución y alterar la dinámica.

**No empezaría con un reparto IMEX automático por etiquetas como «visual», «químico» o «neuronal».** Primero todo implícito; después se justifica qué términos pueden tratarse explícitamente sin imponer otra restricción dominante de estabilidad.

## 2. Cuándo merece continuar B o C

| Ruta | Implementación que tendría sentido | Obstáculo decisivo |
|---|---|---|
| **B: multirrate / formas de onda** | Particiones matemáticas que intercambian trayectorias y corrigen el acoplamiento hasta cumplir un criterio común. | La masa y la realimentación pueden acoplar derivadas entre particiones. Pasos locales pequeños no certifican precisión global. |
| **C: exponencial / Krylov racional** | Acción de funciones de un operador lineal constante por versión del modelo, sin formar matrices exponenciales densas. | Las resoluciones desplazadas, la base Krylov y las mutaciones pueden consumir el ahorro. |

### B: no esconder el acoplamiento de masa

Para una partición \(p\):

\[
M_{pp}\dot x_p
=f_p-\sum_{q\ne p}M_{pq}\dot x_q.
\]

Si \(M_{pq}\ne0\), intercambiar solo valores \(x_q\) y omitir sus contribuciones derivativas **cambia las ecuaciones**.

Transformar a \(\dot x=M^{-1}f\) es matemáticamente posible bajo el contrato propuesto, pero puede convertir un acoplamiento disperso en una dependencia efectiva global. No resuelve gratuitamente el problema de particionar.

**MRIStep sigue exigiendo masa identidad.** No lo conectaría directamente al nuevo contrato ni asumiría que una transformación por \(M^{-1}\) conserva su ventaja computacional. Los métodos MRI-GARK ofrecen condiciones de orden y estabilidad para el acoplamiento; no autorizan simplemente asignar un paso distinto a cada grupo. :chatgpt-content-reference{index="3"}

**Falsador de B:** cambiar la partición, manteniendo exactamente el modelo y la precisión exigida, produce errores fuera del presupuesto global aunque todos los integradores locales declaren éxito.

### C: reutilizar operadores, no respuestas

Para un reparto declarado:

\[
M\dot x=-Kx+g(t,x),
\]

la acción exponencial corresponde a \(L=-M^{-1}K\). Krylov racional puede aprovechar resoluciones desplazadas relacionadas con \(\sigma M+K\), sin construir \(M^{-1}\) explícitamente.

Hay métodos adaptativos publicados para las acciones necesarias en integradores exponenciales. **Controlar esa acción no controla automáticamente la aproximación de \(g\), el acoplamiento completo o una mutación.** :chatgpt-content-reference{index="4"}

**Falsador de C:** el estimador de la función de matriz pasa, pero la trayectoria completa incumple el contrato; o la reconstrucción de operadores/bases tras mutaciones elimina su ventaja frente a A.

Mi orden sería **A primero; elegir entre B y C después de medir dónde se gastan pasos, iteraciones y comunicaciones**.

## 3. Piezas reutilizables y coste real de incorporarlas

| Pieza | Reutilización propuesta | Licencia y coste de ingeniería |
|---|---|---|
| **SUNDIALS / ARKStep** | Integración implícita, adaptatividad y coordinación de resoluciones con masa. | **BSD-3-Clause**. Requiere integrar callbacks generados, vectores/matrices y criterios de error; no suministra vuestro lenguaje de modelos. |
| **Ginkgo** | Álgebra dispersa y solvers/precondicionadores CPU/GPU detrás de la interfaz anterior. | **BSD-3-Clause**. Existe interfaz oficial con SUNDIALS; requiere controlar memoria, ejecutor, formatos y criterio de parada. |
| **PETSc TS**, alternativa a ese conjunto | Residuo implícito y ecosistema de solvers para modelos acoplados. | **BSD-2-Clause**. Es otro conjunto de dependencias e interfaces; no lo añadiría además del anterior sin una necesidad concreta. |

Las licencias permiten modificación y redistribución bajo sus condiciones de avisos y atribución; no exigen regalías por ejecución. **Las dependencias externas conservan sus propias licencias.**   :chatgpt-content-reference{index="7"}

Dos advertencias importantes:

**SUNDIALS mantiene la lógica de control en CPU**, aunque los datos permanezcan residentes en GPU. No sería correcto anunciar «adaptatividad completamente ejecutada en GPU» por utilizar sus vectores CUDA. Debe medirse la coordinación restante. :chatgpt-content-reference{index="8"}

**La interfaz SUNDIALS–Ginkgo no unifica automáticamente todas las tolerancias.** Su documentación distingue criterios de parada y recomienda uno compatible con SUNDIALS. No aceptar el criterio predeterminado de una biblioteca sin comprobar su relación con el contrato del integrador. :chatgpt-content-reference{index="9"}

Reutilizaría estas piezas **antes de escribir otro GMRES, controlador adaptativo o formato disperso propietario**. La capa propia debe concentrarse en descripción, compilación, identidad del estado, intervenciones e inspección.

## 4. Riesgos específicos del «modo rápido»

### La duplicación de paso estima error; no elimina rigidez

Comparar un paso \(h\) con dos pasos \(h/2\) es útil para estimar error en el régimen donde aplica el orden del método. Pero un RK explícito sigue teniendo una región de estabilidad limitada. Relajar tolerancias puede ahorrar pasos por precisión y, aun así, dejar el cálculo limitado por estabilidad. SciPy diferencia expresamente métodos explícitos para problemas no rígidos e implícitos para rígidos. :chatgpt-content-reference{index="10"}

Por eso **«rápido» no debería significar siempre «explícito»**. Inicialmente puede significar tolerancias menos estrictas y una política de salida más ligera, utilizando el mismo modelo y el backend apropiado.

### Tolerancias por variable no equivalen a límite individual garantizado

ARKODE utiliza una norma RMS ponderada. Si solo una variable tiene error normalizado \(e\), esa norma vale \(|e|/\sqrt N\). **Añadir muchas variables inactivas puede diluir la contribución de una variable relevante.** Esta conclusión se deriva de la norma documentada. :chatgpt-content-reference{index="11"}

Conservaría en validación el **máximo normalizado por variable y por puerto**, además de la norma interna del solver. Una prueba barata consiste en añadir estados desacoplados constantes: no deberían convertir en aceptable un error antes inaceptable del subsistema original.

### Los dos modos necesitan el mismo significado científico

No permitiría que el rápido suprima estados, congele química, cambie masa o silencie conexiones sin declararlo como **otro modelo**. Tampoco convertiría saturaciones artificiales en «protección numérica»: recortar concentraciones negativas puede ocultar un integrador inadecuado.

El informe debe separar:

**tolerancia solicitada, estimación local, error observado contra referencia y restricciones físicas comprobadas.** Sin referencia, el último error permanece desconocido.

La inspección debe leer estados confirmados o interpolaciones identificadas como tales, nunca una mezcla de buffers de distintas etapas. Cambiar la frecuencia del inspector no debería cambiar la trayectoria.

## 5. Mutaciones en caliente: hay una corrección matemática imprescindible para `clamp`

Con masa no diagonal, **no es correcto calcular \(M^{-1}f\) y después poner a cero la derivada de la variable fijada**.

Si \(x_c=c(t)\) es una trayectoria impuesta, las variables libres satisfacen:

\[
M_{ff}\dot x_f
=f_f(t,x_f,c)-M_{fc}\dot c.
\]

Esto permite implementar un clamp prescrito sin convertir necesariamente todo el motor en DAE: se elimina la variable conocida y se reconstruye su observación. Pero debe respetarse esa ecuación.

### Falsador de dos variables

Considérese:

\[
M=
\begin{pmatrix}
2&1\\
1&2
\end{pmatrix},
\qquad
f=
\begin{pmatrix}
0\\1
\end{pmatrix}.
\]

Sin clamp:

\[
\dot x=(-1/3,\;2/3).
\]

Al imponer \(x_1=0\), la ecuación libre exige:

\[
2\dot x_2=1
\quad\Rightarrow\quad
\dot x_2=1/2.
\]

**Un motor que devuelve \(2/3\) ha implementado mal el clamp**, aunque CPU y GPU coincidan exactamente. Es una comprobación algebraica propuesta; no ejecuté el núcleo.

Un clamp dependiente instantáneamente de otras variables, en cambio, puede introducir una restricción algebraica no contemplada. Debe rechazarse hasta implementar ese dominio.

### La transacción debe separar épocas del modelo

En cada tiempo de intervención propondría:

**Llegar al tiempo declarado con el modelo anterior, confirmar su estado, aplicar la transacción completa y reiniciar el cálculo con el nuevo modelo.** No integrar atravesando la mutación y luego devolver una interpolación como si fuera equivalente. ARKODE distingue expresamente avance hasta sobrepasar una salida y avance detenido en un tiempo impuesto. :chatgpt-content-reference{index="12"}

La mutación debe invalidar las estructuras afectadas: Jacobianos, precondicionadores, factores, etapas reutilizadas, interpolantes y grafos CUDA ligados a direcciones de memoria. Los métodos multistep y RK con reutilización FSAL requieren reinicio ante discontinuidades; PETSc lo documenta incluso para cambios de coeficientes que no puede detectar automáticamente. :chatgpt-content-reference{index="13"}

**Preservar identidad no basta para preservar física.** Si cambia \(M\), conservar \(x\) no conserva necesariamente la cantidad \(Mx\). Si se elimina una especie, su contenido debe transferirse, registrarse como retirado o rechazarse la operación, según la semántica declarada. El motor no debe inventar ese balance.

La prueba de referencia sería comparar la mutación en caliente con **dos ejecuciones separadas unidas por el mismo mapa explícito de estado**, incluyendo un cambio de dimensión y una mutación abortada.

## 6. Falsador barato para elegir A antes de escalar

Además del clamp anterior, propongo un problema manufacturado de **32 variables**, sin conectoma:

\[
M=\operatorname{tridiag}(0.2,1,0.2),
\]

\[
M\dot x=M\dot z(t)-D(x-z(t))-(x-z(t))^{\circ3}.
\]

\(D\) es diagonal con escalas fijadas entre \(1\) y \(10^6\); \(z(t)\) es una trayectoria suave conocida, con componentes lentas y transitorios rápidos. Con \(x(0)=z(0)\), la solución exacta es \(x=z\).

Este único caso comprueba masa no diagonal, rigidez heterogénea, no linealidad y Jacobianos generados. Se permutan después los IDs y el almacenamiento, sin cambiar la ecuación.

**Rechazaría A si declara aceptación pero excede el error global previamente fijado, depende de los nombres/orden de variables o no conserva el resultado de las mutaciones prescritas.** Una referencia Radau sobre este problema pequeño debe resolver \(M\dot x=f\) correctamente; DOP853 no sería el árbitro único de la parte más rígida. :chatgpt-content-reference{index="14"}

Este es un ensayo de desarrollo, no una reserva independiente. Las extensiones posteriores de siete y doce estados medirán extensibilidad, con la limitación de autoría que ya declaraste.

---

**Recomendación operativa:** conserva el núcleo declarativo en construcción; añade A mediante ARKStep y álgebra dispersa genérica, empezando por masa constante por época. Antes de escalar, exige el clamp correcto, la equivalencia de mutaciones por tramos y la trayectoria manufacturada rígida. B/C continúan como rivales condicionados por mediciones, no como promesas de aceleración.

Eso permite ampliar especies y mecanismos **sin introducir nombres anatómicos en el integrador**, pero todavía no acredita cobertura biológica, rendimiento de cerebro completo ni una relación fija entre los modos rápido y preciso.
