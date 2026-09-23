**Conservaría el contraste invertido como hallazgo descriptivo. Priorizaría B: aislar el efecto del pulso olfativo heredado de la preparación, antes de intervenir un grupo elegido por el ranking.** No porque el sesgo sham demuestre esa causa, sino porque permite una intervención precisa sin modificar conexiones ni ganancias.

## 1. Qué sostiene la lectura y qué corregiría

**No encontré un error algebraico de signo:** las fórmulas y tablas son coherentes con ORN antisimétrico \(+0,510775\) y entrada DNa02 antisimétrica \(-108,982240\). Un sesgo aditivo común se cancela en \((L-R)/2\); por tanto, el sham elevado **no explica por sí solo** esa inversión. Los recibos comunican reconstrucción de la suma aferente con errores inferiores a \(2\times10^{-11}\); no la recalculé desde NPZ.  

**La principal corrección es la interpretación del yaw.** `yaw_grados` contiene la **media del ángulo absoluto** entre 161 y 311 ms —aproximadamente −21,25°—, no el desplazamiento angular ni el valor a 200 ms. No pude corroborar el −0,217632° citado. Reportaría separadamente `yaw(200)-yaw(ON)` y `yaw(311)-yaw(161)`, con unidades y origen explícitos. Un ángulo negativo común no demuestra giro negativo ni cuerpo bloqueado. 

Hay dos límites adicionales del código: la ventana usa **151 muestras con media aritmética**, no la integral trapezoidal de análisis anteriores; y `primero()` aplica \(10^{-8}\) a magnitudes de unidades distintas. Sus fechas son detecciones numéricas descriptivas, **no latencias causales comparables**. No cambiaría retrospectivamente esas métricas; las etiquetaría correctamente. 

Los porcentajes de hold corresponden a **decisiones fuente×ventana seleccionadas**, no a porcentaje de neuronas prescindibles. La corrección a tres canales sensoriales conserva los contrastes; para fechar específicamente el ON olfativo debe distinguirse qué canales son olfativos, porque la comparación actual examina cualquiera de los tres.  

## 2. Un solo experimento causal: pulso heredado presente frente a ausente

Desde **el mismo checkpoint anterior al primer intervalo**, comparar dos preparaciones: la histórica y otra que sustituya únicamente la exposición olfativa consumida en ese intervalo por aire limpio. Mantener tercer canal, duración, dinámica neuronal, cuerpo, intervención PFG existente y parámetros. Después ejecutar izquierda/derecha/uniforme/sham con el mismo ON y horizonte de 335 ms. **No llamar “simétrica” a la segunda preparación ni prolongarla hasta conseguir simetría.** La prehistoria no neutra está declarada en la procedencia. 

Fijaría como resultado principal el cambio entre preparaciones del contraste antisimétrico de **entrada nativa DNa02**, en 161–311 ms. Sham y uniforme distinguen cambios basales/comunes; el yaw es secundario, sin criterio de rescate.

**Falsador:** retirado el pulso, con contraste ORN comparable, la inversión DNa02 permanece dentro de la incertidumbre numérica. Eso debilita **ese pulso como explicación suficiente**, no toda hipótesis de historia. Cambiar únicamente el nivel sham tampoco explicaría la inversión antisimétrica.

## 3. Medición numérica mínima

Antes de interpretar el efecto, medir su estabilidad con **motor protegido y refinamiento temporal sobre las ramas que forman el contraste principal, hasta el horizonte interpretado**, no solo durante los primeros milisegundos. Conservar \(10^{-4}\) y los demás límites vigentes; registrar marcas/eventos, entrada DNa02 y mando, no únicamente conteos finales. El efecto atribuido a la preparación debe distinguirse de la discrepancia numérica entre resoluciones.

No exigiría alcanzar 1 s/60 s ni certificar todos los estados para este diagnóstico; tampoco declararía etapa 3 superada.

**Lectura:** completos índice, revisión aplicada, BUGFIX, `RESULTADO.json`, `INTERFACES.csv` y `HOLD_DIAGNOSTICO.csv`; cuerpo del análisis hasta el inicio de `selftest()` y primeras 12 líneas de `GRUPOS.csv`. **No ejecuté código, NPZ ni fixtures nuevos.**
