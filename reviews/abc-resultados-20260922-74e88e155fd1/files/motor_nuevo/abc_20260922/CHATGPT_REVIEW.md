**Leí íntegramente** el índice, `PLAN_ABC.md`, `recurrence.py`, `recurrence.cu`, `contract.json`, `connections.cu` e `initial_provenance.json` del commit fijado. **No ejecuté esos archivos ni examiné los resultados de las 16 corridas.** Estas son las seis observaciones accionables.

### 1. C tiene una equivalencia algebraica condicionada al signo

`coefficient()` y `sums()` separan excitación/inhibición según el signo de **`w*scale*s`**; `scatter()` decide únicamente por **`w`**. Coinciden si `scale≥0` y las transmisiones comunicadas permanecen no negativas, pero esas condiciones no se comprueban en cada etapa. Un `s` intermedio negativo puede producir sumas diferentes, aunque el estado final vuelva al dominio. **No afirmo que haya ocurrido.**

Comprueba esa condición en los resultados instrumentados o prueba el kernel con cruces de cero. Para equivalencia general, actualiza diferencias de las partes positiva y negativa del producto, no su clasificación fija por peso. **No recortes estados para ocultarlo.**  

### 2. El acumulador de residuo puede ocultar NaN

`residual()` utiliza `fmax(maxima[i],fabs(...))`. CUDA devuelve el argumento numérico cuando el otro es NaN: un residuo inválido puede dejar intacto el máximo anterior. También hay `fmax/fmin` en los coeficientes y la liberación. La comprobación de finitud cada 0,5 ms no cubre necesariamente esas operaciones intermedias.

Añade un indicador persistente de no-finitud, actualizado **antes** de las operaciones que pueden ocultarla, y una prueba de corrupción específica. El máximo residual no debe certificar validez si ese indicador se activó.   :chatgpt-content-reference{index="4"}

### 3. El ejecutor no es todavía el adjudicador del contrato

La fórmula diagonal de `trapezoid()` es consistente con el trapezoide q/s indicado, **condicionada a resolver el acoplamiento recurrente**. Ocho iteraciones no demuestran convergencia; correctamente se calcula después el residuo completo, pero `run()` solo lo registra.

En los archivos examinados tampoco se aplican las cotas de refinamiento, error candidato o dominio. Tu verificación pendiente debe exigir explícitamente: las 16 salidas previstas —o fallos identificados—, tiempos/formas compatibles, refinamiento ≤\(10^{-6}\), errores q/s ≤\(10^{-4}\), residuo A ≤\(10^{-9}\) y dominio contratado, en ambas condiciones. **Un archivo `timing.json` completo no equivale a PASS.** La precisión queda limitada a los tiempos guardados.  

### 4. Verifica C mediante su estado comunicado, no solo q/s

`previous` se actualiza únicamente cuando se transmite un cambio: eso permite acumular variaciones subumbral y es coherente con la propuesta. Pero `aa`, `bb` y `previous` son **estado numérico adicional**, ausente de `states.npz`.

Como prueba focalizada, compara las sumas incrementales contra `raw_sums(previous)`; separadamente, compáralas contra `raw_sums(s)` para distinguir acumulación numérica de aproximación por umbral. Usa una secuencia que avance y retroceda entre estados, como las etapas RK4. Para captura GPU, compara una muestra capturada con la misma secuencia sin captura, incluyendo esos buffers. No veo una congelación evidente del pulso: el código modifica el buffer y sincroniza antes del lanzamiento. 

### 5. El coste publicado necesita una frontera explícita

`started` comienza **después** de `reset`, construcción CSC, sumas iniciales y captura del grafo; la compilación también queda fuera. `wall_with_sampling_s` mide avance y muestreo, no construcción ni reinicialización del motor. Además, cada condición ejecuta referencias y candidatas en orden fijo, una vez.

Conserva la selección predeclarada, pero informa estos costes excluidos y su amortización antes de interpretar una ventaja arquitectónica. No atribuyas automáticamente una diferencia pequeña al método ni al residir en GPU: **A, B y C se ejecutan mediante grafos residentes** en este banco. 

### 6. La extensión de un segundo excedería RAM con este almacenamiento

Si se reutiliza `run()` durante 1 s muestreando cada 0,5 ms, acumulará **2001 snapshots de 333.400 FP64: 5.337.067.200 bytes**. `states=np.asarray(snapshots)` crea otro bloque equivalente mientras conserva la lista: **9,94 GiB entre ambos**, antes del resto del proceso, frente a 8 GiB contratados. La comprobación de memoria ocurre después.

Antes de esa extensión, utiliza almacenamiento preasignado o por flujo y comprueba el presupuesto previamente. **Esto no invalida por sí solo las corridas actuales de 5 ms.**
