# Resultado: controlador por bloques en GPU

**El motor completo aún no alcanza 1 segundo simulado por minuto. Etapa 3 sigue abierta.**

Se implementó un controlador adaptativo genérico dentro de CUDA, con propuesta, rechazo y confirmación privados por bloque. El adaptador real conserva las 51 matrices originales, todos los voltajes/compuertas, fronteras de intercambio, tolerancias y regla de eventos. No depende de la cota de agrupación anterior.

| Organismo real | Tiempo simulado | Avance acumulado | Proceso con carga/guardado |
|---|---:|---:|---:|
| reference20_01 | 20 ms | 82.820 s | 96.356 s |
| device20_01 | 20 ms | 60.317 s | 73.808 s |

La media de pasos 2–20 mejora **1.374×**; el trabajo nativo de membranas mejora **5.212×**, de 28.055 a 5.383 s acumulados. Son mediciones de 20 ms del organismo, no un segundo ejecutado. No se repitieron para escoger el mejor tiempo.

## Fiabilidad medida

La comparación de 20 ms usa la referencia con fronteras de eventos y operador original sin compresión. Las decisiones se reconstruyen desde arrays con el verificador y contrato congelados, no desde el booleano del runner.
PN: 6.8871e-08 mV; estado normalizado: 4.29251e-05; compuertas: 5.87559e-05; guiñada: 4.04498e-10 grados. Conteos, relojes, metadatos y banderas exactos: True. Criba: True.

También se comparó la ventana de 5 ms con la referencia fina conservada. Esa referencia comparte algunos mecanismos heredados: la concordancia no sustituye una prueba independiente de toda la neurofisiología.

La discrepancia al reconstruir se localizó: el cargador histórico perdía los valores efectivos de tau/theta preparados. El registro explícito del operador repone valores, sin recalcular intervenciones. Los estados guardados de continuación fría y después del fallo tienen, respectivamente, 0 y 0 diferencias respecto del control de 1 ms. La sesión corporal fallida continúa bloqueada; no se implementó recuperación automática del cuerpo.

Un modelo adicional de tres estados usa el mismo controlador CUDA. La prueba de células reales verifica independencia de orden/lote, no publicación tras un fallo y reintento exacto. Esto acredita infraestructura reutilizable en esos casos, no soporte universal de cualquier cerebro.

## Rival de puertos y límite pendiente

Se implementó por separado la respuesta de bloques afines a saltos y cambios de operador ordenados. Pasa referencias de exponencial matricial, eventos tardíos, constantes iguales/casi iguales, prefijos y conductancias no conmutativas. Admite un modelo de siete estados sin cambiar el núcleo. **No se conectó como sustituto del CNS no lineal:** falta un control del error de su recurrencia y receptores.

ChatGPT aportó un falsador relevante: un pico entre muestras puede perderse aunque coincida el voltaje final. Los relojes locales no solucionan automáticamente ese problema del detector heredado; el caso se conserva en `DEVICE_CHECK.json`. No se amplió ninguna tolerancia para admitir la candidata.

## Autocrítica y decisión

Bajar de nivel no elimina un algoritmo que repite trabajo global ante eventos locales. En esta ejecución, CNS ocupa 2.103 de 2.994 s por ms; PN aporta 0.310 s/ms. Otra aceleración aislada de membranas tiene un techo bajo. La siguiente ronda debe cambiar la planificación de dependencias y los puertos con error controlado; el Schur acoplado permanece como rival matemático, no como una tercera implementación oculta.

Se conserva A como **PROMETEDOR_NO_CONFIRMADO**: mejora real y criba de 20 ms, sin promesa de segundos ni admisión de etapa 3. B es un componente probado para sistemas afines, sin ganancia global atribuida. C no se implementó. Se cierra este hito de dos prototipos y reparación de identidad para revisión; el objetivo de eficiencia general sigue pendiente.

ChatGPT revisó fuentes anteriores y su comentario cambió la reparación/pruebas; se le envían fuentes y arrays nuevos. No se atribuye reproducción CUDA a esa revisión. Jev hizo una llamada de clasificación acotada (1064/214 tokens), sin decidir tolerancias ni veredictos. No se usaron subagentes Codex.

[Contrato y arquitectura](ARCHITECTURE.md) · [Presupuesto previo](PLAN.json) · [Verificación desde arrays](VERIFIED.json).
