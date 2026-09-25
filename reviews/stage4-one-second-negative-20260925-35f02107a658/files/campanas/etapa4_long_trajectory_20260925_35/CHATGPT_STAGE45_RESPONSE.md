## Dictamen

**No usaría “2 s con tele-giro de +30° al segundo 1” como prueba única de Etapas 4/5.** Puede ser una perturbación exploratoria, pero mezcla una gran discontinuidad corporal con un horizonte corto y carece del control que atribuya la recuperación al feedback.

La prioridad es interpretar el ensayo de **1 s ya iniciado**: distinguir si mejora realmente el rumbo hacia la fuente, si solo revierte parte del deterioro anterior y si ese cambio supera lo esperable por avance y dinámica basal. Campaña27 confirmó modulación fuente→mando/giro; su cierre mantiene explícitamente abiertas navegación y corrección causal de perturbaciones. 

## 1. Los números no sostienen la interpretación de Gemini

En las referencias de Campaña27, el yaw final fue **+0,0449143° en izquierda y −0,0124062° en derecha**. Los **0,0573205° son su diferencia**, no el giro de cada brazo. Al final, ambos mandos instantáneos eran negativos: aproximadamente −0,181 y −0,398°/s, respectivamente. No procede extrapolar una orientación bilateral sostenida a partir del endpoint diferencial.  

Con los valores de bearing que comunicas, el deterioro a 400 ms fue:

| Fuente | Error inicial | Error final | Cambio |
|---|---:|---:|---:|
| Derecha | 18,37685° | 19,72289° | **+1,34604°** |
| Izquierda | 17,34885° | 18,55769° | **+1,20884°** |

**Eso no demuestra necesariamente repulsión del olor.** Avanzar hacia una fuente lateral sin girar suficientemente puede reducir la distancia y aumentar, simultáneamente, el error de rumbo.

En un modelo planar ideal, sin deslizamiento lateral:

\[
\dot d=-v\cos\beta,\qquad
\dot\beta=\frac{v}{d}\sin\beta-\omega,
\]

donde \(\beta\) es el bearing **firmado** y \(\omega\) la velocidad angular corporal, no simplemente el mando.

Usando la geometría preparada publicada —fuente a aproximadamente **1,13 mm**, avance de **0,2 mm/s**—, el avance inicial produce por sí solo una variación de la línea de visión de aproximadamente **+3,02°/s a izquierda y −3,19°/s a derecha**. El cuerpo necesita contrarrestarla para mantener el bearing; girar algo en la dirección correcta puede ser insuficiente. **Este es un cálculo cinemático, no una reconstrucción de MuJoCo.**  

### Por qué +30° a 1 s no equivale a una prueba concluyente

Quedaría **1 s posterior a la perturbación**. Con el límite declarado:

\[
\left|\int_{1}^{2}\omega_{\rm mando}(t)\,dt\right|\le5^\circ.
\]

Deshacer **30° mediante ese mando** requiere al menos **6 s al techo**, es decir, terminar no antes de **7 s desde ON**. Esto **no acota toda la rotación pasiva de MuJoCo**, ni garantiza que el cerebro llegue al techo. Una respuesta parcial sí podría medirse; exigir recuperación completa sería un criterio mal dimensionado.

Además, un `+30°` no es neutral respecto del lado de la fuente. Si significa giro a izquierda, empeora inmediatamente el bearing de una fuente derecha; puede tener otro efecto para una fuente izquierda.

Finalmente, **tele-giro no es torque físico**. Es una intervención sobre el estado. Puede ser legítima como contrafactual, pero requiere declarar pose, velocidades, contactos y sensores pendientes que se modifican. MuJoCo distingue esos estados de las fuerzas aplicadas. No representa la misma perturbación que el pulso previamente descartado. :chatgpt-content-reference{index="5"}

## 2. Cómo interpretar el ensayo de 1 s

Definiría \(E(t)=|\beta(t)|\) y conservaría, sin reemplazar los criterios existentes:

\[
E(0),\quad E(0,4),\quad E(1),\qquad
J_E=\int_0^1 E(t)\,dt,\qquad
P=d(0)-d(1).
\]

**Hay tres resultados distintos:**

- **\(E(1)<E(0)\):** mejora neta del rumbo, todavía sin atribución causal al olor.
- **\(E(0)<E(1)<E(0,4)\):** recuperación parcial del deterioro; no mejora neta desde el inicio.
- **\(E(1)\ge E(0,4)\):** no hay recuperación del bearing en ese horizonte.

Una reducción de distancia aislada no decide navegación, porque el avance tónico puede producirla. Tampoco trasladaría automáticamente a 1 s la paridad numérica comprobada durante 400 ms.

## 3. Tres controles causales distintos

Son **experimentos del modelo neurocorporal**, no validación en animales vivos. Mantienen cerebro, pesos, lector y cuerpo.

### A. Retirada selectiva del contraste bilateral

**Pregunta:** ¿la información izquierda–derecha contribuye favorablemente al rumbo, o predomina intensidad común/historia basal?

A partir de la entrada consumida en la vida derecha, construir:

\[
C(t)=\frac{c_L+c_R}{2},\qquad D(t)=c_L-c_R.
\]

Comparar dos cintas sensoriales desde el mismo preparado:

\[
u_{\rm lateral}=(C+D/2,\ C-D/2),\qquad
u_{\rm sin\ contraste}=(C,C).
\]

Se conserva exactamente la intensidad común temporal. Ambos cerebros siguen controlando sus cuerpos y recibiendo la propiocepción correspondiente.

**Métrica:** diferencia de \(J_E\), progreso hacia la misma fuente derecha y comando angular; estabilidad y rapidez como controles.

**Falsador:** quitar \(D\) no empeora el resultado, o incluso lo mejora. Eso rechazaría la necesidad beneficiosa del contraste para esa vida, no toda capacidad olfativa.

**Prioridad:** esta es mi primera elección si el ensayo de 1 s **no mejora bearing**. Localiza un problema funcional antes de añadir una perturbación de 30°. El replay lateral que reproduce al donante es aquí un control de identidad, **no prueba de feedback**.

### B. Transferencia de fuente: online frente a yoked

**Pregunta:** ¿actualizar el olor según el movimiento propio mejora la trayectoria más que recibir una secuencia sensorial abierta?

Tras una historia idéntica, cambiar la fuente a una ubicación predeclarada. Ambos brazos experimentan el mismo calendario ambiental:

\[
u_{\rm online}(t)=c(p_{\rm antenas,online}(t),S(t)),
\]

\[
u_{\rm yoked}(t)=c(p_{\rm antenas,donante}(t),S(t)).
\]

La cinta debe evaluarse con **la nueva fuente sobre las poses donantes**; no usar simplemente olores antiguos de otra fuente. El yoked conserva toda su autoridad motora: sus decisiones **sí mueven el cuerpo**.

**Métrica:** ventaja online en error de rumbo y progreso hacia la nueva fuente, calculados desde el cambio; registrar también cuánto se separan realmente ambas entradas.

**Falsador:** online no mejora sobre yoked más allá de la incertidumbre, aunque ambos reaccionen al cambio externo de olor. Un salto sensorial grande demuestra estimulación; no demuestra utilidad del feedback.

**Prioridad:** primera elección si 1 s **sí mejora bearing**. Es el control necesario antes de interpretar esa mejora como aproximación en lazo cerrado, tal como exige la hoja de ruta publicada. 

### C. Desorientación corporal pequeña, emparejada

**Pregunta:** ¿el feedback corrige una desviación nueva de la relación cuerpo–fuente?

Usar una perturbación cinemática pequeña y explícita —por ejemplo, **1° como contrato nuevo**, caracterizado primero en cuerpo— en online y yoked desde estados emparejados. Conservar un brazo sin perturbación para separar recuperación de la evolución espontánea. Una réplica de signo opuesto sería necesaria para ampliar la conclusión a corrección bilateral.

**Métrica:** reducción posterior del exceso de error causado por la perturbación, frente a yoked y al control sin perturbación; no simplemente el yaw absoluto.

**Falsador:** la recuperación aparece igual sin feedback, proviene de la dinámica mecánica, o el efecto impuesto queda por debajo de la resolución del experimento.

**Prioridad:** después de B, o como diagnóstico explícitamente limitado. No aumentar el torque del pulso fallido ni presentar esta nueva intervención de estado como su rescate.

## Recomendación operativa

**Si 1 s mejora bearing: B antes que el tele-giro de Gemini. Si no mejora: A antes de gastar otra vida larga en perturbaciones. C queda para probar robustez cuando haya una respuesta dirigida cuya recuperación merezca examinarse.**

Una sola trayectoria de 2 s podría aportar un negativo útil o mostrar reacción parcial. **No distingue por sí sola navegación, feedback y restauración mecánica**, y no cierra automáticamente las dos etapas.

**Verificación propia:** leí completos `CLOSE_01.json`, `HOJA_DE_RUTA_CAUSAL.md`, `CAMPOS.json` y `GEOMETRY.json`; del CSV de referencia, cabecera y primeras muestras, líneas 398–404 y 798–801. Ejecuté únicamente cálculos Python de diferencias angulares, geometría y límites de integral de mando. **No ejecuté NPZ, MuJoCo, organismo ni el ensayo de 1 s; el pulso fallido y los bearings comunicados se mantienen como evidencia local vuestra.**
