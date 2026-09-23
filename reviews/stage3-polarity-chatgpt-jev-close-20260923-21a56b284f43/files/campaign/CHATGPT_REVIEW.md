# Respuesta externa de ChatGPT, 23-09-2026

Fuente: conversación «Investiga el conectoma de mosca». Texto recibido tras compartir el snapshot da1e29e. Declaración de lectura y límites al final; es propuesta externa, no instrucción ni validación independiente de arrays.

**Priorizaría B —convergencia posterior— como hipótesis de trabajo, pero no daría A por falsada: `DM1_lPN.q` es un puerto legacy, no necesariamente la salida consumida por cada receptor.** El subconjunto sigue perteneciendo a la rama PFG histórica; no demuestra el comportamiento del pipeline vigente.  

### Qué cambia con el subconjunto

El promedio ALPN negativo **no localiza una inversión ORN→PN**: mezcla glomérulos. Además, el contraste positivo de DM1_lPN oculta asimetría: a 220 ms, su componente antisimétrica izquierda es aproximadamente **−0,000136**, y la derecha **−0,004159**; su diferencia resulta positiva. No es una respuesta espejo limpia. MBON32 casi no discrimina condiciones en esa instantánea, pero eso no descarta contribuciones anteriores o filtradas. 

### A/B/C y falsadores

| Hipótesis | Qué medir en el pipeline vigente | Qué la debilitaría |
|---|---|---|
| **A. Transformación periférica/ALLN altera el contraste útil** | Entradas ORN y ALLN a DM1_lPN; liberación local efectiva PN, separando rutas finas y legacy. | Que esas salidas receptor-específicas conserven el contraste esperado durante la ventana completa y la discrepancia aparezca después. `q` legacy positiva sola no basta. |
| **B. Convergencia posterior cancela o transforma el contraste** | Ramas realmente conectadas desde PN hacia LH/MB/CX y contribuciones firmadas a los aferentes DNa02, incluyendo fuentes no olfativas. | Que la entrada efectiva DNa02 tenga dirección correcta, pero la transformación DNa02→mando o mando→cuerpo sea la que falle. |
| **C. Estado inicial/historia o lectura motora dominan** | Baseline sham, estado previo, mando solicitado/aplicado y respuesta corporal; controles dentro del mismo pipeline. | Una preparación alternativa prefijada no modifica el sesgo relevante y mandos espejo producen respuestas corporales espejo. Son dos componentes de C que deben comprobarse separadamente. |

**Controles comunes:** izquierda/derecha/uniforme/sham, mismo estado preparado, operador y relojes. No elegir duración, grupo o ganancia después de mirar yaw. Uniforme ayuda a distinguir respuesta común, pero no garantiza igual dosis.

### Fuentes: qué respaldan y qué no

**Olsen–Wilson 2008** estudia inhibición lateral **entre glomérulos** y supresión presináptica ORN→PN; “lateral” no significa automáticamente “contralateral izquierda/derecha”. No demuestra una inhibición contralateral excesiva en vuestro modelo. :chatgpt-content-reference{index="3"}

La anatomía muestra vías paralelas hacia LH/MB y varias convergencias sobre DNa02: **ORN→PN→LHN→LAL→DNa02 no es una cadena universal**. Cero conexiones directas DM1_lPN→MBON32 no descarta rutas multisinápticas; los conteos crudos tampoco determinan corriente efectiva. :chatgpt-content-reference{index="4"}

### Instrumentación mínima de flujo

Para receptores seleccionados, registrar **la variable realmente consumida**, su propietario, ruta/reemplazo, receptor, lado y tiempo. Separar:

- **Conductancias:** \(g\), potencial de inversión y voltaje receptor; corriente con convención explícita, por ejemplo \(I=g(E-V)\).
- **Entradas del modelo:** transmisión × ganancia/peso efectivo, conservando sus unidades; no llamarlas pA si no lo son.
- **Salida y actuación:** liberación PN efectiva → entrada DNa02 → estado/lector → mando aplicado.

Acumular por grupo en pasos **aceptados**, separar predictores y rechazos, y comprobar que los términos reconstruyan la entrada utilizada. No reevaluar el operador para observarlo. El coste de estas reducciones debe medirse.

### Control numérico antes de interpretar

Primero, instrumentación encendida/apagada con misma aritmética debe conservar estados exportados y eventos. Después, comparar la condición candidata y su control con **refinamiento temporal sobre la ventana interpretada**, manteniendo \(10^{-4}\) y límites existentes; el efecto de flujo debe distinguirse de la discrepancia numérica. `pending_sensors` ya es cero: **50 ms sham no se justifican para eliminar un pulso inexistente**, aunque sí puedan estudiar otra historia explícita. 

**Alcance:** leí completos ambos README, `ARCHIVE.json`, `analyze_layers.py`, `verify_portable.py` y `SUBSET_PROVENANCE.json`; parcialmente `FOCUS.csv`. No pude descargar el ZIP, no calculé sus hashes ni ejecuté sus arrays. Esto prioriza una medición, **no aprueba etapa 3**.
