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
