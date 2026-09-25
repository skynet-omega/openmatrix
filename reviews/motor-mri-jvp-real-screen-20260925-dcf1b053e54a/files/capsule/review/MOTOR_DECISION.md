# Decisión del motor tras dos cribas reales de localidad y una JVP

**No se promovió un motor nuevo.** El motor acoplado observado sigue alrededor de 2,46 s de pared por ms simulado en el par sham20ms; la meta del usuario es 0,060–0,120 s/ms para cinco segundos completos. Etapa 4 de navegación y etapa 5 siguen abiertas. Los siguientes resultados son diagnósticos sobre el mismo bloque sham MaleCNS aceptado de 125 µs; no miden comportamiento largo.

## A. MRI con proveedor rápido: realizaciones directas descartadas

La repetición [v4](mri_real_frozen_fast_04/MRI_REAL_FROZEN_FAST_RESULT.json) reparó la semántica izquierda/derecha de los eventos, verificada con ADD/SET CUDA en Python normal y `-O`. Conservó exactamente los arrays `initial/reference/high/low` y el negativo v3: endpoint normalizado **1,865**, defecto muestreado **146,659**, embebido **0,186**, cinco `F_full` y 229 `F_fast`. El arreglo no rescata el rápido diagonal porque sus subpasos no transmiten el evento hacia coordenadas libres; tampoco demuestra que el bug de `side` sea inocuo para un rápido recurrente. El estado completo del organismo, eventos, cuerpo y salida final quedaron exactos frente al control.

El código MRI de ChatGPT (SHA `03ffdf60…`) mostró además un contraejemplo CPU con evento tardío: embebido y dos auditorías muestreadas valen **0**, pero el endpoint independiente tiene error normalizado **322,689**. Lo ejecutamos en Python normal/`-O`; su `FAIL_RETAINED` conserva el fallo. Por tanto, muestrear el resto lento no certifica error continuo.

La topología base de siete eventos alcanza 1.263 receptores directos, 14.598 neuronas en dos saltos y 102.786 en tres. Su objetivo de mayor error `hDeltaA` está a dos saltos, no unido directamente a las siete fuentes. Un `F_fast` que recompute todas las salidas del primer salto en cada una de las 229 llamadas necesitaría **7,519** barridos base incluyendo cinco completos y sin contar setup ni propietarios: supera seis.

ChatGPT propuso una operación distinta: seleccionar **ecuaciones receptoras finales**, no fuentes. Su cribador CPU `zone_cost_gate.py` (SHA `37b1ee96…`) pasó controles sintéticos normal/`-O`, pero no ejecutó neuronas reales. Medimos el coste necesario para esa selección con datos reales:

| Selección de filas q | Filas | Aristas base activas de entrada | Barridos equivalentes optimistas con 5 full + 229 fast |
|---|---:|---:|---:|
| Receptores anatómicos directos | 1.263 | 373.716 | 8,345 |
| Receptores **efectivos observados** con A/B/A | 175 | 112.659 | 6,008 |
| Defecto MRI muestreado >1 | 472 | 106.236 | 5,951 |
| Unión de los dos conjuntos anteriores | 640 | 215.168 | 6,926 |
| Unión quitando **todas** las filas marcadas como especializadas | 547 | 135.553 | 6,213 |

El A/B/A de siete eventos fue reversible bit a bit y dejó intacto el organismo; cambió 182 filas q en total: 175 receptoras base directas y siete puertos prescritos. Es sensibilidad local a una perturbación `+0,001` en un estado/tiempo, **no** prueba ausencia permanente de influencia en otras filas. El defecto muestreado >1 afecta **1.521 coordenadas libres**: 472 en las primeras 166.700 q y 1.049 en estados s/adicionales. El presupuesto de seis barridos permite a lo sumo **111.715 aristas activas por llamada rápida** si cinco barridos completos son obligatorios y setup/propietarios cuestan cero. Incluso la unión q sin filas especiales excede ese margen con 229 llamadas; aún faltan las 1.049 coordenadas no q, el cierre de propietarios y trabajo real de GPU. Esto descarta la **realización directa de un kernel de filas base por cada callback al conteo observado**, no toda MRI: una compresión temporal certificada, menos llamadas o un operador analítico serían algoritmos nuevos que requieren presupuesto propio.

## B. JVP/Krylov: condición local aprobada, integración pendiente

Se fijó antes del ensayo la primera etapa lenta interior (`t=41,667 µs`), dirección libre proporcional a `F_full−F_fast` y amplitud de **una tolerancia normalizada**, sin ajuste de amplitud. La preprueba confirmó dominio `[0,1]` para `z±v`. Se hicieron sólo seis consultas del operador completo: `F(z)`, `F(z±v)`, `F(z±v/2)` y repetición de `F(z)`. El primer `F(z)` coincidió bit a bit con el archivado, la repetición fue exacta y la salida corporal/neuronal del organismo no cambió. La discrepancia normalizada prospectiva de linealización fue **2,0156×10⁻¹¹ ≤ 0,1**, sobre una respuesta JVP no nula (máximo temporal normalizado ≈0,02389). El verificador reconstruyó el cociente desde seis arrays y rechazó corrupción en Python normal y `-O`.

Esto autoriza una **sonda acotada de coste y dimensión de Krylov/JVP** como alternativa B. No prueba que pocos vectores basten, que una JVP compilada sea barata, que la matriz de masa especializada pueda omitirse, ni que exista ganancia temporal. Una JVP por diferencias centrales consume dos llamadas completas; si la dimensión requerida es alta, B fallará el presupuesto antes de integrar el organismo.

## C. Agenda QSS/influencia

Permanece independiente: debe manejar estados graduales, propietarios y recurrencia, con cota de error continuo y costo de actualizaciones. Los dos CSC cuantizados anteriores no alcanzaron coste/precisión; no se reactivan cambiando umbrales. La expansión del cono a 76,95 % de aristas activas salientes en tres saltos es un riesgo, no un no-go dinámico.

## Decisión operativa y límites

La siguiente ronda debe comparar un prototipo B de JVP completo y una formulación A **temporalmente comprimida y certificada** o C de agenda, con presupuesto previo y máximo dos prototipos completos. Antes de invertir en el integrador, medir el coste de una JVP en el operador efectivo completo, cuántos vectores necesita en un bloque real y cómo se tratan eventos, masa y propietarios. La ruta A por máscara directa ya tiene dos negativos y no recibe otro ajuste de filas. En paralelo, el propietario celular vigente cuesta ≈0,263 s/ms, más del doble de **todo** el techo 0,120 s/ms; incluso CNS gratuito no alcanzaría la meta. Los relojes CUDA anteriores eran contradictorios y Nsight en este WSL no atribuyó kernels, así que falta un perfil fiable antes de modificar ese propietario.

Jev11, con los nuevos números, escogió A temporal (probabilidad 0,52; confianza **0,36**) frente a B 0,22 y owner 0,23: sólo asesoría débil, sin ejecutar datos. ChatGPT revisó el código publicado del cono y propuso la criba JVP; su revisión de la nueva semilla real aún no está hecha. Ninguno constituye auditoría biológica independiente.

Recibos: [cono](EVENT_CONE_RESULT.json), [receptores estructurales](ROW_EVENT_READER_RESULT_57.json), [receptores efectivos](effective_event_readers_01/EFFECTIVE_EVENT_READER_RESULT.json), [cinco RHS](mri_slow_inputs_01/MRI_REAL_FROZEN_FAST_RESULT.json), [coste de la semilla](MRI_ROW_ZONE_COST_RESULT_61.json), [JVP real](jvp_real_01/JVP_REAL_RESULT.json) y [verificador](JVP_REAL_VERIFY_63.json). Cada organismo de 1 ms usó directorio único, menos de 38 s de pared y menos de 12 GiB RAM; sus salidas originales quedaron exactas frente al control. Los diagnósticos no son una corrida de cinco segundos ni validan etapas 4/5.
