# Revisión no autoral del observador49

Revisión iniciada27-09-2026 13:40:16UTC; presupuesto60sCPU/600s pared, ceroGPU y pasos. Se leyeron `operand_observer.py`, `coefficient_capture.cu`, `capture.cu`, `verify_operands.py` y el contrato; se inspeccionó un milisegundo real de `observer_parity_02`. No se editaron fuentes congeladas ni datos originales. Matrix Astra realiza la comparación completa de endpoints y reconstrucción de los4ms; no se duplicó esa tarea.

## Dictamen del productor

No encontré un error concreto de propietario, indexado, alias o secuencia de copia en el productor examinado. Esto es una revisión acotada, no certificación general del motor.

* Cada una de las seis filas tiene una sección propia en el CSR compactado. El índice `observed_ptr[j] + e - ptr[row]` mantiene el orden de aristas de esa fila aunque el listado de destinos no esté ordenado numéricamente. Los tipos de punteros/índices corresponden a int64/int32.
* Los valores `w[e]`, `release[c]`, `caps[c]` y la máscara se copian **dentro del mismo bucle** que calcula la suma, desde los argumentos FP32 consumidos; no se reconstruyen desde la publicación neuronal posterior. Las seis identidades y su ausencia de sustituciones especializadas se comprobaron en07; el alcance de49 conserva ese modelo.
* Tras la llamada completa a `coefficient`, `capture_rhs` toma objetivo/tasa finales y derivada; por eso se pueden distinguir los coeficientes genéricos de una sustitución posterior. El instrumento no escribe en esas variables ni añade una segunda evaluación.
* `copy_stage` conserva las cuatro etapas en buffers independientes. `copy_trial_operands` usa el contador antes de que el kernel posterior `capture_trial` lo incremente. Ambos se ejecutan en el flujo de lanzamiento del mismo integrador. La descarga de cada época usa el stream de ese núcleo y comprueba tamaño, capacidad y estados. No apareció una lectura de memoria sobrescrita por el siguiente ensayo.
* La capacidad256 produce un fallo explícito si se supera; no cambia el paso para ocultarlo. La falta de no interferencia no puede descartarse sólo leyendo código: requiere la paridad real realizada por Matrix Astra.

## Corrección confirmada: cuarto RHS

El verificador inicialmente leído esperaba fracciones `[0,.5,.75,1]`. En el archivo real la cuarta fracción es0 porque `GraphRK23` pasa su reloj izquierdo de extremo, `nextafter(end,start)`, al último RHS. Los tiempos observados lo confirman. Se avisó inmediatamente; Matrix Astra había identificado y reparado el mismo supuesto de manera independiente. El v3 revisado exige `[0,.5,.75,0]` y el extremo izquierdo exacto. No se cambió ni debe cambiarse el integrador/captura para satisfacer el verificador anterior.

## Límites demostrados del verificador v3

Se cargó `observer_parity_02/operands_001ms.npz` una vez y se hicieron corrupciones **sólo en memoria**, restituyendo el valor después de cada caso. El archivo original permaneció intacto. El SHA del archivo y del verificador usado están en `OBSERVER_REVIEW.json`.

| Cambio deliberado | Resultado de `verify()` | Consecuencia |
|---|---|---|
|Duplicar un peso consumido del mayor término del primer RHS|Rechazado: suma FP32 de10360|Sí detecta una corrupción material de los operandos aritméticos.|
|Cambiar `pre_ids[0]`|Aceptado|La reconstrucción aritmética no autentica la identidad de las fuentes presinápticas.|
|Invertir `committed[0]`, predictor→comprometido|Aceptado|No certifica por sí sola la atribución temporal a la trayectoria comprometida.|
|Duplicar `gain` de DNb05 en un RHS con margen positivo|Aceptado|No reconstruye la rama positiva de la ley neuronal: comprueba rango, derivada y rectificación negativa.|

Son límites de validación demostrados, **no pruebas de que las capturas originales estén corruptas**. Tampoco contradicen una paridad real de los endpoints. Deben distinguirse al describir qué certifica el verificador.

Antes de atribuir sumas a tipos celulares, el lector debe enlazar `rows/positions/pre_rows/pre_ids/ptr` con la selección CSR congelada; no basta con comparar los seis IDs de destino. Antes de llamar a una evaluación comprometida, debe comprobar metadatos de época contra el contrato midpoint125us, reloj de origen y milisegundo registrado. Además de la época comprometida, una lectura de trayectoria debe seleccionar ensayos aceptados; los ensayos rechazados también se guardan intencionalmente.

Para la ley positiva hay dos salidas válidas: verificar su ecuación con una referencia y un criterio numérico fijados, o limitar explícitamente la afirmación a sumas/margen/tasa/derivada y rama rectificada negativa. No propongo ajustar tolerancias después de ver los resultados ni repetir adquisiciones. Los hashes de los archivos cerrados ayudan a detectar modificaciones posteriores, pero no sustituyen una verificación semántica de índices o fases.

## Medida y alcance

La prueba CPU consumió **0.373sCPU**, picoRSS137641984bytes, y reconstruyó3120sumas en520RHS del archivo original. Las corrupciones y sus resultados quedaron registrados; no se importó el modelo/CuPy ni se llamó aGPU, CNS o MuJoCo. El resto del trabajo consistió en lectura estática. No se declara superación de etapas4/5 ni validez fisiológica.

Las reparaciones pendientes son offline y pueden aplicarse a los mismos arrays. El productor puede seguir aportando evidencia mientras se conserva esta distinción; no hay un hallazgo aquí que justifique cambiar sus ecuaciones o capturar otra vida.
