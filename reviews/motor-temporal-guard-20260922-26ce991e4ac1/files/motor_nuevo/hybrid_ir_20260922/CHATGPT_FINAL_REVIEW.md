**Conservaría la mejora medida como ingeniería acotada, pero no promovería todavía la supresión de cortes como política numérica general.** El clasificador y su fallback pasan las pruebas CPU de lecturas declaradas. Sin embargo, **reproduje un caso con RHS continuo, certificado favorable y estimador cero que omite una respuesta superior a \(10^{-4}\)**. Es el mecanismo de evento tardío ya conocido, ahora contrastado con el nuevo planificador; no un fallo nuevo del filtro.

**Ejecuté pruebas CPU, no CUDA ni el organismo.** La ejecución corregida consumió **1,74 s de pared y 1,73 s de CPU**, con 103,54 MiB de memoria máxima. No descargué el ZIP completo ni reconstruí sus arrays de 20/50 ms.

:chatgpt-content-reference{index="12"}[**Reproductores, fuentes verificadas, resultados y alcance — 22 KB**](sandbox:/mnt/data/MATRIX_SCOPE_REVIEW_CPU_4df0dbc.zip)

## 1. Lectura directa viva y derivada sustituida: la clasificación funciona en los casos ensayados

Ejecuté `check_dependencies.py` sin modificarlo y comprobé la selección de fronteras de `ScopedGraph` utilizando un padre espía CPU: registra qué fronteras recibiría el ejecutor, sin simular CUDA.

| Caso | Resultado |
|---|---|
| \(q,s\) prescritos; el receptor libre lee únicamente \(s\) | Clasifica continuidad y retira cortes. |
| Un estado libre lee directamente \(q\) | Clasifica dependencia viva y conserva la frontera suministrada. |
| Modo `baseline` | Conserva la frontera. |
| Declaración de continuidad contradicha por el callback | Rechaza la construcción. |

La prueba original también rechaza los cuatro índices malformados que incluye. **No refuté esas reglas de clasificación.** Eliminar una dependencia porque su **derivada completa** está sustituida es distinto de ignorar una lectura cuyo resultado sigue vivo; el IR hace esa distinción mediante `zero_derivative_rows`.   

La garantía continúa siendo condicional a que las lecturas estén declaradas correctamente. `model_event_contract.py` contiene un mapa manual vinculado a fuentes comprobadas, **no una extracción automática de todas las dependencias del programa**. 

## 2. Contraejemplo ejecutado: continuidad no basta para el estimador actual

Usé el mismo caso de tres estados:

\[
q'=-q/\tau,\qquad s'=(q-s)/\tau,\qquad z'=(s-z)/\tau.
\]

Los puertos \(q,s\) se proyectan exactamente; \(z\) es libre. Estado inicial nulo, masa identidad, \(h=\tau=125\,\mu s\), y un salto `ADD(0,5)` en \(q\) a \(0,9h\).

El clasificador devuelve correctamente `continuous_free_rhs=True`: el RHS libre depende de \(s\), que es continuo. Pero las evaluaciones temporales que alimentan a \(z\) ocurren antes del evento.

**Ejecuté los métodos publicados `coeff()` y `midpoint()`, extraídos por AST sin alterar sus operaciones, sobre NumPy.** Los puertos se calcularon mediante `scipy.linalg.expm` 2×2 y la referencia mediante la exponencial del sistema completo 3×3. No ejecuté el controlador C++ ni su aceptación CUDA. 

| Observable | Resultado CPU |
|---|---:|
| \(z\), paso completo | **0** |
| \(z\), dos medios pasos | **0** |
| Diferencia usada por el estimador | **0** |
| \(z\), referencia independiente | **0,00226209354509** |
| Valores no finitos o fuera de \([0,1]\) | Ninguno |

El estimador resulta cero para cualquier `atol/rtol` positivos. **Esto no contradice el certificado de continuidad; contradice usarlo, junto con ese estimador, como garantía general para retirar cortes.** Tampoco demuestra que las trayectorias reales publicadas hayan sufrido esa infracción.

### Cambio concreto

En `ScopedGraph.advance()`, la decisión actual depende solamente del modo y de `continuous_free_rhs`. Separaría:

**continuidad estructural** de **cobertura temporal del efecto del evento**. 

Para la segunda hace falta un control sensible al efecto no muestreado —defecto posterior al evento o cota de la contribución al receptor—, o conservar el corte cuando esa cobertura no esté establecida. **No propongo restaurar ciegamente todos los cortes ni cambiar \(10^{-4}\).** La prueba anterior debe impedir que una cobertura temporal insuficiente se confunda con error estimado pequeño.

## 3. Las tres perturbaciones uniformes son falsadores, no una prueba de completitud

Comprobé además este límite:

\[
F_{\mathrm{libre}}=q_1-q_2,\qquad q_1=q_2=0,2.
\]

Cambiar **ambas fuentes simultáneamente** a 0, 0,37 y 1 produce tres diferencias cero. Cambiar solamente \(q_1\) a 0,37 produce **0,17**.

Esto **no demuestra que vuestro mapa omita ese lector**: una declaración completa lo detectaría. Sí demuestra por qué los tres sondeos uniformes de `ScopedGraph` no pueden sustituir la declaración revisada. Añadiría al fixture una perturbación diferencial o por fuente; no una batería de miles de corridas del organismo. 

## 4. Masa: el puente conserva la matriz local, pero no amplía el certificado

En `ir_waveform_bridge.py`, el bloque recibe `model.mass.toarray()` y el callback devuelve `raw_rhs`, **antes de resolver masa**. No encontré allí una diagonalización ni una doble aplicación de \(M^{-1}\). El puente exige una población de una entidad, un puerto escalar y ausencia de clamps; la prueba publicada contrasta explícitamente la aplicación de masa una vez. **Lo revisé documentalmente, no reejecuté ese puente.**  

Su masa no diagonal es **interna al bloque**. No acredita mezcla de masa entre bloques ni entre un puerto prescrito y variables libres. En ese último caso aparece, por ejemplo,

\[
M_{ff}\dot x_f=F_f-M_{fp}\dot p,
\]

y la continuidad de \(F_f\) por sí sola no resuelve el efecto de una transición de \(p\). El certificado actual excluye masa y retardos expresamente; mantendría esa exclusión, sin convertirla en un requisito de implementación universal para esta ronda. 

## 5. Qué conservar y decisión sobre etapa 3

**Conservaría:**

- La mejora publicada de **2,0814× en avance a 50 ms**, con su comparación acotada y procedencia.
- La clasificación de lecturas y fallback, bajo sus supuestos declarados.
- El puente CPU de masa local y el prototipo de trayectorias como control funcional, **no como acelerador**.

Los recibos separan correctamente `prospective_short_screen=True` de `full_state_qualified=False` y `stage3_admission=False`. No reinterpretaría los 67 campos sin criterio como 67 defectos, ni el contraste favorable como calificación completa.  

**¿Conforme para etapa 3?** Para **diagnósticos acotados**, sí, con alcance numérico explícito. Para una **campaña interpretativa**, todavía no: falta cubrir el efecto temporal de los eventos y contrastar los observables relevantes en el horizonte que se interpretará. No hay aún un segundo medido. La meta de velocidad sigue siendo operativa, no un criterio biológico.

No abriría otra campaña general ni ampliaría retrospectivamente las cuatro cargas consumidas. El cambio inmediato es **una condición temporal adicional y su falsador pequeño**, manteniendo A/B/C para la siguiente decisión.

### Alcance exacto

**Leídos completos:** `README.md`, `dependency_ir.py`, `model_event_contract.py`, `scoped_graph.py`, `compare_round.py`, `ir_waveform_bridge.py`, `check_ir_bridge.py`, `check_dependencies.py`, `run_scoped.py`, `EXTERNAL_VERIFIED.json`, `legacy_runtime/block_midpoint.py`, `legacy_runtime/event_coupling.py` y `campanas/etapa3_motor_nuevo_20260922/graph_core.py`.

**Lectura parcial:** secciones finales de `COMPARISON_20.json` y `COMPARISON_50.json`, incluidos rendimiento, campos sin criterio y veredictos.

**Ejecutado:** prueba original de dependencias; selección/fallback con dobles CPU; rechazo de declaración contradicha; métodos numéricos publicados sobre el caso sintético; límite de las perturbaciones uniformes. Cuatro fuentes coinciden con sus blobs Git. Un primer intento falló al preparar un doble de importación; se conserva y se corrigió únicamente el arnés. **No ejecuté CUDA, arrays del organismo ni verificación del ZIP.**
