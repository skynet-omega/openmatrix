# Contraste lateral antes de ampliar la duración

Limitación observada: en el brazo derecho de 1 s el error de orientación creció 3,739936°, con contraste antenal sostenido, giro corporal −0,03110° y mando integrado −0,02280°. El techo de ±5°/s no saturó. El control corporal previo sí responde. Etapas 4 y 5 siguen abiertas.

Tres alternativas de Codex, formuladas con exposición a la revisión anterior de ChatGPT:

| Alternativa | Operación e información | Discriminador y falsador |
|---|---|---|
| A: contribución sensorial lateral | Reproducir la cinta consumida del donante con L/R originales frente a su media bilateral. Mantener intensidad común, tiempo, tercer canal, circuito, lector y cuerpo. | Si quitar D mejora, el contraste no ayuda en esta condición; si empeora, sí aporta, aunque pueda ser insuficiente. La identidad debe reproducir el donante antes de interpretar. |
| B: transferencia neural a movimiento | Recalcular cancelación y signo del lector desde estados DN ya registrados, contrastando autoridad física medida. | Una saturación o discordancia de la ecuación identificaría interfaz; ambas faltan en el donante. La cancelación observada es descripción, no explicación causal. |
| C: corrección tardía | Extender la vida espacial sin cambiar modelo a 2 s. | Sólo vale para una predicción temporal nueva; otra aproximación a la fuente con avance tónico no demuestra navegación. No se eligió todavía. |

Decisión preliminar: B ya identifica mando pequeño y cancelación; preparar A. ChatGPT recomienda A, pero declara no haber ejecutado los NPZ. Se le solicita contraste sobre horizonte, resolución numérica y condiciones para etapa 5. La ablación sería diagnóstico causal del modelo, no una demostración de feedback espacial ni de equivalencia biológica.

Presupuesto propuesto antes de nuevos organismos: dos brazos de 40 ms de preparación y 1.000 ms de ensayo; 4.300 s de pared por brazo, 8.600 s agregados; 18 GiB RAM y 12 GiB GPU, máximo 3 GiB de salida; cero cambios al motor o parámetros biológicos. Ejecución secuencial, con parada si el control de identidad falla. No se usará el checkpoint final como una reanudación validada. No se suma evidencia de dos vidas para afirmar una vida continua de 2 s.

## Contraste recibido y revisión puntual

ChatGPT coincidió en B seguido de A y entregó un verificador NumPy. Corrigió el alcance del margen numérico: 0,001° por brazo es una meta, no una resolución demostrada a 1 s; la reproducción exacta no sustituye una referencia de discretización. La criba 0,022° permanece provisional. Jev eligió A con confianza 0,63; su selección no constituye una medida ni decide validez.

La revisión bibliográfica puntual del 25-09 confirmó lo ya preservado en `etapa4_diseno_20260923_17/EVIDENCIA_VIVA.md`: [Yang et al., Cell 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/) vincula varias DN con giro, sin calibrar nuestra ley `tanh(250·Δq)`; [Rayshubskiy et al., eLife 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12279373/) registra directamente la relación bilateral DNa02–giro durante marcha. No es un descubrimiento nuevo ni licencia para sustituir el lector tras observar esta trayectoria.

El recibo real de preparación35 identifica los cuatro canales del lector como `[10045,10056,10118,10065]`; los dos usados para yaw son **DNb05 izquierda y derecha, en ese orden**. No se debe etiquetar los dos primeros canales de `DN_q_actual` como DNa02: son IDs distintos. La preparación heredada también iguala tau/theta de cada par PN y DNb05; se conserva idéntica en ambos brazos. Por tanto, aquí «sin cambio biológico» significa sin cambio respecto de ese modelo ya intervenido, no equivalencia con todas las propiedades fisiológicas de una mosca real.

## Comprobación barata ante la crítica de Gemini

Sin cambiar ningún brazo ni seleccionar parámetros, se evaluó la misma ley lectora sobre las 1.000 muestras DN consumidas del donante35. Ganancias multiplicadas por 1, 4, 16, 64 y 256; techo conservado en 5°/s. Esto es sensibilidad algebraica sobre una trayectoria expuesta, **no** una vida nueva ni una predicción validada del cuerpo en lazo cerrado.

| Multiplicador | Mando neto integrado (°) | Módulo integrado (°) |
|---|---:|---:|
| 1 | −0,022796211 | 0,157389961 |
| 4 | −0,088152045 | 0,622103689 |
| 16 | −0,233570935 | 2,140080254 |
| 64 | −0,257958509 | 4,105597888 |
| 256 | −0,410913533 | 4,814433214 |

546 muestras negativas y 454 positivas: el límite algebraico de ganancia infinita, con estos estados congelados, es −0,460° netos. No es una cota de toda ganancia intermedia ni del organismo realimentado. Aumentar fuerza amplifica también las correcciones de signo contrario; calibrar sigue siendo una alternativa útil, pero «mando débil» no demuestra que baste multiplicarlo. La dirección a la fuente cambió −3,771° en la vida original. No se ajustó una ganancia para intentar aprobar.

Reproducción CPU (desde la raíz, sólo lee el donante):
```python
import numpy as np
with np.load('campanas/etapa4_long_trajectory_20260925_35/native_minus_02/traces.npz', allow_pickle=False) as z:
    q, b = z['DN_q_usada'][40:], z['DN_baseline'][40:]
    x = (q[:, 2] - b[:, 2]) - (q[:, 3] - b[:, 3])
    for f in (1, 4, 16, 64, 256):
        u = 5 * np.tanh(250 * f * x)
        print(f, .001 * u.sum(), .001 * np.abs(u).sum())
    print('limite ganancia infinita', .005 * np.sign(x).sum())
```

## Decisión tras la nueva revisión externa

El usuario pidió contrastar la crítica de Gemini con ChatGPT. La respuesta de navegación99d72212… leyó protocolo y código, sin ejecutar NPZ; se conserva junto a la revisión del motor en CHATGPT_GEMINI_REVIEW.md. Recomendó completar esta pareja una sola vez dentro del presupuesto y corrigió su consejo anterior: la ablación no es puerta universal para calibración independiente. Admitiría un control más corto en un diseño prospectivo que justifique equivalencia de la frontera; no transformar este control incompleto en PASS del contrato completo.

Se mantiene la ejecución ya congelada, sin nuevas rondas ni ajustes de ganancia. Interpretación precisada: conservar C en concentración antenal NO conserva necesariamente actividad neural común, por no linealidad y adaptación. El contraste mide el efecto total de igualar entradas; no identifica aisladamente un comparador bilateral ni una sinapsis de signo incorrecto. Un efecto beneficioso insuficiente prioriza identificación independiente del lector; uno perjudicial pide localizar cambios de actividad/historia antes de invertir signos; uno por debajo de resolución queda inconcluso. Estos son criterios de interpretación, no modificación de los umbrales congelados.

La calibración requiere datos compatibles y correspondencia declarada entre unidades fisiológicas y q. Puede prepararse en paralelo, pero no se autoriza elegir parámetros para rescatar el yaw de esta trayectoria. No se cambia el motor estable ni se atribuye el turno interrumpido a una causa de interfaz sin evidencia.

## Datos para calibración: disponibilidad comprobada

La consulta puntual de la biblioteca del25-09 encontró código de análisis DNb05 ya adquirido en `investigacion/dnb05_20260923` y registros DNa02 históricos. El código calcula filtros temporales, pero no contiene grabaciones ni una correspondencia q→fluorescencia. La sección Data and code availability de [Yang et al., Cell2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/) indica que los datos se comparten solicitándolos a los autores; [Zenodo12775493](https://zenodo.org/records/12775493) publica código. No se contactó a terceros ni se descargaron datasets grandes. No se presume que registros de otra clase celular calibren automáticamente DNb05. Esta limitación afecta la afirmación de calibración fisiológica; no impide comparar lectores como hipótesis funcionales explícitas con controles retenidos.

## Incidencia de recursos durante la ablación

Al comienzo de no_contrast_01 se observaron pasos aproximadamente dos veces más lentos que en identity_01, incluida la preparación sin olor. La comprobación de procesos encontró dos contextos CUDA simultáneos, PIDs436959 y438273. La tarea Motor C++/CUDA informa que está ejecutando una referencia de100ms, después de su candidata. La GPU RTX4070TiSUPER estaba al97%,2820MHz y50°C; no evidencia de caída de frecuencia en esa muestra. No se atribuye ese aumento de tiempo a la intervención sensorial ni se usa esta pareja para medir aceleración del motor. Se mantiene el límite temporal original. No se detuvo ni modificó la otra tarea.

## Actualización del motor, posterior al comentario de Gemini

La tarea Motor C++/CUDA publicó `motor_nuevo/neurocore_continuidad_20260925_02/RESULTADOS.md`: una variante de punto medio exponencial que conserva el método de referencia y comparte la primera evaluación pasa100ms con diferencias cero en los campos comparados. Recuento de evaluaciones112620→93850, un16,667% menos; ambas ejecuciones cierran en0. La reducción temporal observada34% no se atribuye al motor porque hubo carga compartida. Esta es otra variante, no una rehabilitación retrospectiva del RK3(2) comentado por Gemini, ni demuestra que nextafter explicara aquel fallo. En esta tarea se leyó el informe; no se reejecutó ni verificó su evidencia binaria. No se cambió el motor de la pareja36. Al llegar la ablación a127–145ms, el segundo proceso CUDA ya no figuraba y sus pasos volvieron al rango del control.

La contención reapareció cerca de514ms de la ablación con PID447329 (nueva candidata100ms de membranas). Se comunicó el coste y el presupuesto a la tarea Motor C++/CUDA. Su responsable preservó la candidata activa y suspendió sólo su supervisor antes de lanzar la referencia; después comunicó que la candidata terminó correctamente y que reutilizaría una referencia congelada tras comprobar identidad de entradas/fuentes antes de comparar. No alteró36. Desde626ms sólo figuraba nuestro contexto CUDA. La coordinación evita nuevas corridas concurrentes, pero no elimina retroactivamente la contención registrada ni autoriza afirmar velocidad a partir de esta pareja.

## Cierre de la pareja y siguiente decisión

Ambos brazos completaron 40+1000 ms con salida 0 y sin errores de cierre. Presupuesto consumido 8029.575/8600 s; ambos por debajo de 4300 s. El control reproduce la traza donante y ambos preparados coinciden en sus 588 arrays. Verificadores normal y -O idénticos; el código de ChatGPT ejecutado localmente coincide en el efecto final **+0.128338101°**, favorable al contraste original en este modelo discretizado. Clasificación PROMETEDOR_NO_CONFIRMADO: falta comprobar resolución numérica de los dos segundos completos, y ambos brazos fallan el mínimo histórico de orientación. No hay promoción de etapas ni nueva corrida.

La media de concentración se conserva exactamente, pero la media del par PN cambia hasta 0.00459105983 y la del par DNb05 hasta 3.49590414e-05 en unidades q. Se verifica la advertencia de no linealidad de ChatGPT: este efecto no aísla un comparador bilateral. Los datos nuevos apoyan una contribución beneficiosa insuficiente; no prueban que el lector sea la única causa.

Próximas tres alternativas, prospectivas y todavía sin ejecutar:

| Alternativa | Operación y datos permitidos | Discriminador y falsador |
|---|---|---|
| A: identificar el lector dinámico | Contrastar un mapeo actividad→mando con registros independientes compatibles y una conversión de unidades declarada; si sólo hay prueba funcional, etiquetarla como tal. | Predecir condiciones retenidas sin elegir parámetros por esta trayectoria; falla si sólo rescata la vida expuesta. Es la prioridad de preparación. |
| B: localizar transformación e historia neural | Analizar las trazas PN/DN consumidas ya archivadas y después una intervención acotada que separe actividad común de diferencia lateral. | Una predicción de capa y tiempo debe sobrevivir a control emparejado; una mera correlación o inversión de signo elegida a posteriori no basta. |
| C: utilidad del feedback espacial | Comparar fuente online con reproducción de entrada bajo misma preparación, cuerpo y perturbación, sólo tras justificar separación observable. | Si no hay ventaja online, no se admite control correctivo en ese horizonte. Extender duración por sí solo no es un falsador útil. |

Autocrítica: fue costoso repetir un control completo de 1 s. ChatGPT admite una calificación más corta de la frontera en un futuro protocolo justificado; no se convierte el prefijo en equivalencia medida del resto. Las tolerancias estrictas deben relacionarse con la decisión experimental, sin presentar un fallo de paridad como inutilidad universal ni ruido biológico como permiso numérico. Se coordinó la GPU cuando se detectó contención; los tiempos de esta pareja no son un benchmark de rendimiento. Se detiene al completar el discriminador presupuestado y conservar los resultados.
