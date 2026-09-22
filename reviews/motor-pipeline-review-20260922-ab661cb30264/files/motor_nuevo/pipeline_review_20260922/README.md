# Revisión del pipeline y decisión — 22 septiembre 2026

**Decisión: continuar B, aclarando primero la discrepancia transitoria. No promover todavía `causal_cuda` a una campaña interpretativa de etapa 3.** Motor: PROMETEDOR_NO_CONFIRMADO. Reparaciones de infraestructura: verificadas localmente en el alcance descrito, pendientes de revisión de los archivos nuevos.

## Resultado que cambia la decisión

Dos ejecuciones reales de 1 ms con el mismo checkpoint, condición sham, sin preparación adicional, geometría original y fronteras de eventos activadas completaron correctamente. La identidad del operador efectivo serializado coincide. Sin embargo, la diferencia máxima de estado normalizado es **0.0001785882326256448**, mayor que el límite histórico **0.0001**. Los conteos son exactos y no difiere la guiñada; eso no compensa el incumplimiento. No se relajó la tolerancia.

La coordenada 234831 corresponde al filtro sináptico de la fila neuronal 57073, bodyId 75907, KCg-m izquierda. No es la coordenada de liberación somática. No hay referencia refinada de 1 ms que determine cuál backend es más preciso; tampoco está localizada todavía la primera frontera divergente. El resultado describe discrepancia entre métodos, no error verdadero medido contra la biología.

Las comparaciones favorables anteriores de 5/20 ms conservan su alcance de extremos. **Autocrítica: era excesivo extenderlas a fidelidad de trayectoria o preparación general para etapa 3.** Las ganancias de coste anteriores no quedan anuladas: el organismo de 20 ms pasó de 82.82 a 60.32 s de avance; la meta aproximada de 1 s simulado por minuto real sigue incumplida. Los tiempos de estos smokes incluyen serialización y no se usan como benchmark de velocidad.

## Errores concretos reproducidos y reparados

- El runner de etapa 3 instalaba una implementación anterior sin seleccionar explícitamente el nuevo backend ni activar sus fronteras. Ahora exige `--engine`, verifica la implementación instalada y registra su procedencia.
- Una salida ya existente podía ser sobrescrita por el manejo final de un fallo. Ahora la creación es exclusiva, antes de entrar en ese manejo.
- Interrupciones y errores de carga dejaban estados ambiguos; se registran como INTERRUPTED o INCOMPLETE. Los snapshots se publican mediante renombrado únicamente después de completar archivos y hashes. Serializar no demuestra reanudar el cuerpo completo.
- El registro de operador aceptaba intercambiar rutas de destino y permitía escrituras parciales ante un destino de solo lectura. La versión 2 autentica también las rutas contra el adaptador, prevalida destinos e intenta rollback ante fallo de copia. Un rollback fallido invalida el propietario.
- Un fallo después de publicar eventos podía dejar estado parcialmente visible. Se inyectó el fallo después de un kernel CUDA exitoso: el backend ahora se invalida y rechaza reutilización. No se afirma rollback automático de esa publicación.
- El comparador histórico aceptaba alteraciones artificiales de +1000 mV en el estado espacial y +1000 en velocidad corporal, porque esas variables carecían de criterio. Su contrato original queda intacto. La nueva calificación separa ese resultado parcial de una afirmación de estado completo; detecta las dos corrupciones. En el contraste real de 20 ms existen 66 campos flotantes cambiados sin límites declarados: son cobertura pendiente, no 66 errores demostrados.

Pruebas y recibos: `REPAIR_CHECK.json`, `RUNNER_FAILURE_CHECK.json`, `PUBLICATION_FAILURE_CHECK.json`, `COVERAGE_BEFORE.json`, `COVERAGE_AFTER.json`, `SMOKE_CHECK.json`. Los negativos anteriores y las fuentes ejecutadas se conservan.

## Medición que orienta la siguiente arquitectura

Instrumentar las decisiones CNS no alteró ninguna hoja exportada de cerebro, cuerpo o trazas respecto del smoke causal: comparación exacta en `TRACE_EQUIVALENCE.json`.

En los ocho intercambios aceptados de 125 microsegundos hubo 118 ensayos, todos aceptados: 56 recortados por eventos, 54 por la propuesta adaptativa y 8 por fin de época; ninguno por máximo. De los 54 limitados por propuesta, 52 tenían error inferior a 0.1. El controlador vuelve a crecer desde pasos reducidos por eventos. Esto justifica estudiar su política, pero modificarla sería un cambio numérico prospectivo, no una reparación cosmética.

Una frontera con siete fuentes tiene 1264 receptores CSR directos y 166311 neuronas alcanzables entre 166700. El mapa solo incluye CSR efectivo: no contiene todas las rutas sustituidas, filtros o máscaras. **Alcanzabilidad no demuestra necesidad de recalcular toda la red.** No basta para elegir partición ni para descartar una respuesta de puerto acotada.

Alternativas conservadas: A, diagnóstico pequeño de etapa 3 cuando la interpretación numérica esté sustentada; B, resolver el transitorio y el trabajo global medido; C, arquitectura distinta con comparación común si B muestra una barrera. No se inició un tercer prototipo ni se afinó fisiología para obtener PASS.

## Revisión externa y presupuesto

ChatGPT reprodujo en CPU dos defectos del registro anterior, revisó código publicado y recomendó B. Después de conocer el fallo de 1 ms, priorizó aclararlo antes de optimizar CNS. Esa última respuesta interpretó cifras comunicadas, sin ejecutar los nuevos archivos. Textos y alcance en `CHATGPT_REVIEW.md`, `CHATGPT_DECISION.md` y `CHATGPT_TRANSIENT.md`.

Jev ejecutó una consulta real de clasificación de tareas: 1095 tokens de entrada y 218 de salida. No es una aprobación científica. El usuario pidió además investigar awesome-fly; esa investigación se documenta separadamente en `../fly_resources_20260922`.

Se consumieron las cuatro cargas de organismo previstas, con tope de 240 s por proceso, sin ampliar el presupuesto al aparecer el fallo. La tercera completó el cálculo pero falló al serializar enteros NumPy del informe de instrumentación; la cuarta corrigió únicamente ese reporte. Ambos resultados se conservan. Cero prototipos numéricos nuevos. No hay campaña larga ni cola automática iniciada.

## Reproducción del paquete de revisión

El ZIP incluye fuentes, dependencias locales de las comprobaciones breves y arrays reales para reconstruir discrepancias y decisiones. No incluye el conectoma/cuerpo estático completo ni los grandes snapshots de sesión. Sus manifiestos quedan como procedencia; no sustituyen sus archivos para una reanudación. No se reivindica reproducción autónoma del organismo completo desde este paquete.

Desde una extracción limpia, con Python 3.10, NumPy 1.26.4 y SciPy 1.15.3:

```bash
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -O motor_nuevo/pipeline_review_20260922/test_repairs.py
python -B motor_nuevo/pipeline_review_20260922/test_runner_failures.py
python -B -O motor_nuevo/pipeline_review_20260922/check_coverage_after.py
python -B -O motor_nuevo/pipeline_review_20260922/reconstruct.py
# Opcional: CUDA 12 + CuPy 13.6; fixture pequeño, no carga el organismo.
python -B motor_nuevo/pipeline_review_20260922/check_publication_failure.py
```

Las pruebas esenciales usan excepciones; el runner rechaza Python optimizado porque los loaders históricos conservados dependen de comprobaciones que no deben desaparecer. Las fuentes del runner completo se incluyen para inspección; ejecutarlo requiere los recursos históricos locales. No ejecutar ahora otra campaña para buscar un extremo favorable.

Siguiente decisión acotada: fijar comparación de transitorios y convergencia con igual historia efectiva; discriminar detección temporal de pico, filtrado y orden de publicación antes de elegir una modificación. No empezar por relajar precisión ni por una reescritura general sin ese diagnóstico.
