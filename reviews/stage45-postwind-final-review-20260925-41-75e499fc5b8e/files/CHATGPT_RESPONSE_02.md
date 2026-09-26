## Alcance y decisión

**No conseguí leer ninguno de los cuatro archivos:** `REPORT.md`, `run_replay.py`, `verify_clock.py` y `VERIFIED.json`. Intenté los RAW, rutas alternativas de GitHub y descarga directa; obtuve fallos de acceso y resolución. Por tanto, **no doy por revisados el código, el manifiesto ni sus hashes**. Tampoco descargué o ejecuté NPZ. La validación de `7d9b949f` que comunicas corresponde a vuestra ejecución local.

Mi exposición incluye tu resumen y mi respuesta anterior; de campaña32 sólo conozco el antecedente que acabas de proporcionar. El dictamen siguiente es **condicional a los resultados que informas**, no una certificación de las fuentes.

**Mantendría `PROMETEDOR_NO_CONFIRMADO` y A como prioridad.** Pero modificaría el siguiente paso: antes de otro replay o vida neural, separaría **dirección del mando entregado** de **respuesta angular realizada**, utilizando las trazas existentes. Hay dos puntos que cambian la interpretación.

## 1. El giro realizado también empeoró la orientación; falta saber si obedecía al mando

Con tus cifras:

\[
\Delta e=\Delta(\beta-\psi)
=-5{,}905064^\circ-2{,}239007^\circ
=-8{,}144071^\circ.
\]

Eso es compatible, dentro del redondeo, con pasar de \(e=-19{,}414163^\circ\) a aproximadamente \(-27{,}558233^\circ\), cuyo **valor absoluto aumenta**.

Por tanto, para el balance integrado posviento, **retiro mi hipótesis anterior de “giro correctivo superado por el cambio geométrico”**: tanto \(\Delta\beta\) como \(-\Delta\psi\) contribuyeron en la dirección adversa. Esto no dice que cada instante fuese adverso.

Pero hay un salto que todavía no está justificado:

> Giro corporal neto equivocado ≠ mando neural necesariamente equivocado.

Podría haber una consigna angular correctiva con respuesta corporal adversa por dinámica, desfase o semántica de aplicación; o una consigna ya equivocada al salir del filtro/interfaz. **La autoridad contrastada en32 no distingue esas posibilidades en esta trayectoria perturbada.**

Las ablaciones aportan un resultado causal más concreto: bajo el montaje histórico y las órdenes restantes congeladas, retirar el giro o retirar el avance mejoró el error final. **No localizan por sí solas el defecto en el codificador.**

En particular, los \(3{,}309522^\circ\) de mejora al retirar giro **no equivalen** a los \(2{,}239007^\circ\) de yaw de identidad: la intervención puede cambiar tanto yaw como trayectoria y bearing. Lo mismo vale para retirar avance. Para cada intervención \(j\), con prefijo común:

\[
e_j(T)-e_I(T)
=\big[\beta_j(T)-\beta_I(T)\big]
-\big[\psi_j(T)-\psi_I(T)\big].
\]

Ése es el contraste que impide etiquetar automáticamente «giro» como efecto exclusivamente angular y «avance» como efecto exclusivamente geométrico. Para comparar las ablaciones, usaría **su instante efectivo común de intervención, 1021 ms**, sin confundirlo con el final del pulso a 1020 ms.

**Decisión que cambia:** no aumentar amplitud ni ampliar B para resolver una supuesta falta de fuerza neural. Primero determinar si el mando final pedía corregir o agravar el error.

## 2. La identidad histórica es válida como referencia, pero no certifica una restauración físicamente equivalente

El mecanismo que describes es técnicamente plausible: la documentación de MuJoCo exige cálculos cinemáticos previos para que los Jacobianos sean consistentes con `qpos`; identifica `mj_kinematics` y `mj_comPos` como etapas mínimas. Esto respalda la explicación de auxiliares fríos, **no verifica vuestra implementación concreta**. :chatgpt-content-reference{index="0"}

Restaurar en1000 ms para reproducir el reinicio original **no es un arreglo ilegítimo de la identidad**. Reproduce la historia efectivamente ejecutada. Las ablaciones posteriores pueden seguir siendo válidas aunque esa historia contenga un defecto.

El riesgo concreto es **mezclar políticas de restauración entre ramas**: identidad con auxiliares fríos históricos y una intervención con auxiliares recalculados. Entonces dejaría de cambiarse únicamente giro, avance o viento. Sin leer `run_replay.py`, no puedo verificar que esa condición se conserve en todas las ramas.

Hay que mantener separados dos objetivos:

**Reproducción histórica:** conservar la semántica original, incluida la primera aplicación ineficaz, para interpretar campaña40.

**Motor prospectivo:** que restaurar el estado permita aplicar correctamente la próxima fuerza externa. Una identidad exacta obtenida reproduciendo el defecto no satisface ese segundo objetivo.

No atribuiría los \(8{,}14^\circ\) al defecto, ni convertiría «800 llamadas» en evidencia de 800 aplicaciones efectivas. Asimismo, **sin el control sin viento no está identificado cuánto del deterioro depende del pulso histórico**. Nada de esto invalida automáticamente los contrastes giro0/avance0 si comparten exactamente la misma historia anterior.

**Decisión que cambia:** conservar41 como diagnóstico del sistema histórico, pero no trasladar su identidad exacta como garantía de corrección del reinicio para la próxima vida.

## Discriminador siguiente para A: mando solicitado frente a giro realizado

**Elegiría un análisis de las trazas ya existentes, sin otro rollout.** No repite autoridad ±5°, no ajusta ganancias y no amplía el presupuesto consumido.

Sea \(e=\beta-\psi\), firmado y continuo en el tramo analizado, y \(u\) la **consigna angular realmente entregada después de EMA, umbral y saturación**, en rad/s y con la misma convención de yaw. No usar el comando crudo ni el torque como sustitutos.

Para \(V=e^2/2\):

\[
\dot V
=\underbrace{e\dot\beta}_{\text{geometría}}
+\underbrace{(-eu)}_{\text{dirección del mando}}
+\underbrace{e(u-\dot\psi)}_{\text{diferencia entre consigna y giro real}}.
\]

El último término **no es “mecánica pura”**: incluye seguimiento, temporización e interfaz. La fórmula es una identidad contable, no una identificación causal neural.

Puede calcularse sin derivadas ruidosas. En cada intervalo físico, definir:

\[
\bar e_i=\frac{e_i+e_{i+1}}2,\qquad
U_i=\int_{t_i}^{t_{i+1}}u(t)\,dt,
\]

\[
G_i=\bar e_i\,\Delta\beta_i,\qquad
C_i=-\bar e_i\,U_i,\qquad
M_i=\bar e_i\,(U_i-\Delta\psi_i).
\]

Entonces, usando ángulos coherentes y sin saltos de rama:

\[
\boxed{\sum_i(G_i+C_i+M_i)=\frac{e_T^2-e_0^2}{2}}.
\]

Esta igualdad permite comprobar el cálculo algebraico; **no valida por sí sola el reloj del mando**, porque \(U_i\) se cancela entre \(C_i\) y \(M_i\). Su correspondencia con los intervalos de aplicación debe venir del verificador y de la semántica real del registro.

Lo evaluaría por separado en identidad, giro0 y avance0 desde1021 ms, conservando también la evolución temporal para no ocultar cancelaciones. La lectura que decide el siguiente trabajo de A es:

| Resultado | Siguiente foco |
|---|---|
| \(C>0\): el mando final aporta en dirección adversa al error observado. | Signo, marco de referencia, sincronía y transformación crudo→EMA→salida. Todavía no demuestra que el origen sea neuronal. |
| \(C<0\), pero \(M>0\) lo contrarresta y el giro realizado es adverso. | Aplicación y seguimiento de la consigna en el cuerpo perturbado. No repetir la prueba general de autoridad. |
| El mando y el giro realizado son correctivos, pero \(G\) los supera en algún tramo. | Interfaz conjunta avance+giro y evolución geométrica de la fuente. No confundir detener avance con navegación. |

Para identidad, tus incrementos netos ya descartan que **todo** el deterioro posviento sea simplemente geometría superando un giro neto correctivo. Este análisis añade justamente lo que falta: **si ese giro adverso fue solicitado o ocurrió pese a una solicitud correctiva**.

No aplicaría el umbral de \(0{,}5^\circ\) a estos componentes —sus unidades son rad²— ni los usaría para sustituir el criterio final registrado. Son un discriminador diagnóstico.

**Conclusión:** los resultados comunicados sostienen que mantener esas órdenes congeladas empeoró el desenlace frente a sus ablaciones, no que exista recuperación olfatoria ni que esté localizado un fallo neural. El siguiente avance útil es separar mando adverso de seguimiento adverso con la traza actual. El sin-viento permanece pendiente para otra ronda; B/VNC sigue acotada y C/patas posterior. El bloqueo de las fuentes deja sin completar la revisión del montaje y del código.
