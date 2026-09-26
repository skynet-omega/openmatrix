# Dataset y correspondencia con el modelo 11

Revisión iniciada el 25-09-2026 y cerrada el 26-09-2026. Grafo, parámetros y estado realmente usados por `event_memory_rk3_20260925_11`, previa al ensayo de 2 s. Históricos y motores 07/11 permanecen intactos. No se importó el organismo, no se asignó GPU y no se ejecutó una vida nueva.

**El motor contiene el grafo canónico preparado de MaleCNS v1.0, con 166.700 neuronas; no contiene una reconstrucción fisiológica completa de cada célula ni todos los endpoints del archivo original.** La correspondencia de IDs, filas y orientación es correcta en los arrays inspeccionados. Las principales restricciones son de alcance del modelo: frontera anatómica excluida, signo y escalas hipotéticos, componentes especializados parciales y bastante estado heredado. La red es muy recurrente: el mayor componente fuertemente conectado del soporte efectivo principal contiene el 97,12 % de las neuronas.

No se encontró una corrupción del dataset que justifique cambiarlo antes del ensayo. Sí debe describirse y congelarse la combinación **anatomía + estado + operador efectivo + reglas de propietarios**, no sólo el nombre del conectoma ni `W`.

## 1. Cadena real de carga

| Capa | Fuente efectiva y significado |
|---|---|
| Motor 11 | [run_trial.py:10](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/run_trial.py:10) fija MATRIX, campaña 40 y campaña 15. Comprueba [SOURCES.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/SOURCES.json), carga el organismo y restaura la preparación antes del primer paso. |
| Loader anatómico | [motor_runtime.py:10](/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922/motor_runtime.py:10) llama a `AntennalContactRuntime.load`, lee `data/male_v10/nodes.parquet` y aplica la preparación declarada. |
| Checkpoint constructor | [matrix_diagnostico_olfativo_v1.json](/home/daroch/AXIOMA_FLYWIRE/matrix/config/matrix_diagnostico_olfativo_v1.json) fija `work/stage234_settling_extension_20260915/settled_700ms`; su `core_carrier/brain` aporta los IDs, CSR y parámetros históricos. `700ms` es el nombre de aquella extensión, no el reloj absoluto del organismo. |
| Grafo original preparado | [manifest.json](/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain/manifest.json) conserva `scope=all`, release, particiones, hashes y la intervención morfométrica. Es el grafo canónico MaleCNS, aunque el repositorio se llame AXIOMA_FLYWIRE. |
| Estado inicial del ensayo | [prepared_state/MANIFEST.json](/home/daroch/AXIOMA_ASTRA/campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state/MANIFEST.json), reloj **44.486.000.000 ns**. Incluye sesión, prótesis, frontera, publicación y operador efectivo. |
| Restauración | [restore_prepared.py:21](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/restore_prepared.py:21) exige `effective_operator.npz`, conserva el grafo constructor, restaura estados y operador, y comprueba la identidad del operador. La carga anatómica sola no reproduce el estado del ensayo. |
| Ejecución heterogénea | [runtime_session.py:23](/home/daroch/AXIOMA_ASTRA/motor_nuevo/pipeline_review_20260922/runtime_session.py:23) instala propietarios CNS, PN, células y eventos. El propietario de eventos real procede de `campanas/etapa3_pn629_intervention_20260923_15`, fijado explícitamente en el runner. |

La revisión nueva guarda hashes de los archivos realmente leídos en [GRAPH_AUDIT.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/GRAPH_AUDIT.json) y [STATE_AUDIT.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/STATE_AUDIT.json). El resultado existente de 1 s registró comparación exacta del estado inicial en las cinco ramas serializadas; esto verifica esa restauración, no la fisiología del estado.

## 2. Qué significa «completo» en este dataset

La [procedencia canónica](/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/provenance.json) identifica **MaleCNS v1.0, flat connectome oficial, minconf-0.5**. Sus tres fuentes son anotaciones de cuerpos, anotaciones de neurotransmisor y pares con conteos sinápticos. `node_ids` son `bodyId` exactos de esa release; `flywireType`, `mancBodyid` y nombres de tipos no convierten esos IDs en IDs de otro dataset.

Fuentes primarias de esa importación: [anotaciones oficiales de cuerpos](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather), [anotaciones oficiales de neurotransmisor](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather) y [conteos oficiales por par](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather). Se utilizaron los derivados locales existentes y su procedencia, sin redescargar esos archivos. Los hashes actuales de C, IDs y nodos coinciden con los registrados por el importador. Esas fuentes respaldan anatomía/anotaciones; no aportan las leyes temporales que el motor añade.

La regla principal conserva anotaciones con `superclass` presente y no vacía, incluidas **94** tentativas `tbc`. De **211.577** anotaciones, **166.700** pasan y **44.877** quedan fuera. Se preservan aparte anotaciones sin superclass y gliales; no se les simula una dinámica por estar archivadas. La importación no aplica el filtro adicional `>=5` contactos: conserva pares de un contacto y autocontactos. La selección oficial de confianza 0,5 ya fue realizada aguas arriba.

| Magnitud | Canonical interno | Frontera no canónica registrada |
|---|---:|---:|
| Pares dirigidos internos | 25.582.938 | La frontera queda como agregados por nodo |
| Conteo de contactos internos | 124.177.617 | — |
| Contactos recibidos desde anotados sin superclass | — | 661.930 |
| Contactos recibidos desde IDs sin anotación | — | 5.614.376 |
| Contactos enviados a anotados sin superclass | — | 476.820 |
| Contactos enviados a IDs sin anotación | — | 170.414.577 |

El artefacto `scope=all` tiene cero nodos **canónicos externos** porque incluye todos los canónicos. Eso no anula la frontera del archivo upstream. Los endpoints no anotados pueden incluir fragmentos u otras entidades; **no son un conteo demostrado de neuronas ausentes**. Tampoco se debe interpretar su actividad desconocida como reposo fisiológico. [anatomical_preparation.py:138](/home/daroch/AXIOMA_FLYWIRE/matrix/src/anatomical_preparation.py:138) distingue estas dos fronteras; el modelo de tasas declara actividad externa cero como supuesto.

Algunas poblaciones presentes: 89.403 `ol_intrinsic`, 32.164 `cb_intrinsic`, 13.161 `vnc_intrinsic`, 9.201 `visual_projection`, 6.370 `vnc_sensory`, 6.098 `ol_sensory`, 4.868 `cb_sensory`, 1.846 ascendentes, 1.314 descendentes y 708 motoras VNC. Hay **11.751 tipos únicos** y **2.194** nodos sin tipo en la procedencia. Un nombre de familia sirve para seleccionar e interpretar; no aporta automáticamente parámetros eléctricos, receptores ni un motor corporal.

## 3. IDs, orientación, duplicados, ceros y pesos

Verificación CPU sobre los arrays reales:

- `node_ids.npy`, `nodes.parquet.bodyId` y `brain/state.npz.node_ids` son idénticos; IDs int64 únicos y estrictamente crecientes. `nodes.node_index` coincide con la fila canónica.
- `C=counts_pre_post` es CSR canónico int64, **fila presináptica y columna postsináptica**. Su transpuesta CSR coincide exactamente en `indptr` e `indices` con el `W` cargado: **fila postsináptica y columna presináptica**. El recorrido `W.indices[e]` identifica a la fuente, no al destino.
- No hay conteos negativos, cero o no finitos en C. El formato canónico confirma que los pares internos no están duplicados; el importador registró cero filas internas duplicadas agregadas. Esto no afirma una auditoría de duplicados de todos los pares excluidos upstream.
- Hay **101 pares autápticos / 474 contactos**. El peso anatómico mediano es **2 contactos**, p99 **46**, máximo **2.591**. **10.299.701** pares tienen un único contacto. `nnz` cuenta pares agregados, no sinapsis individuales.

Ejemplos del mapa comprobado:

| bodyId | Fila canónica | Tipo |
|---:|---:|---|
| 10.065 | 52 | DNb05 |
| 10.118 | 104 | DNb05 |
| 10.176 | 159 | DM1_lPN |
| 10.208 | 189 | DM1_lPN |

La construcción original es `W_post,pre = 0,03 * C_pre,post * signo(NT_pre)` en FP32. ACh recibe +1; GABA, glutamato e histamina −1; el resto 0. Es una **hipótesis de corriente por emisor**, no un signo medido de cada receptor. Los ceros se conservan intencionadamente para preservar el registro anatómico. [anatomical_rate_brain.py:94](/home/daroch/AXIOMA_FLYWIRE/matrix/src/anatomical_rate_brain.py:94).

Hay **3.718 neuronas con signo cero**: 2.999 `unclear`, 178 sin etiqueta, 392 dopaminérgicas, 101 octopaminérgicas y 48 serotoninérgicas. Sus salidas producen **1.023.803 entradas nulas** en el CSR base. Esto no demuestra que las células sean fisiológicamente silenciosas: pueden recibir señal y existir lectores especializados, incluido el mecanismo modulador de plasticidad. El `nt_consensus_nt` tampoco equivale a medición: la procedencia conserva por separado predicción, ground_truth y consenso; 81.216 nodos no tienen campo upstream `ground_truth` y 20.126 etiquetas predichas difieren del consenso cuando ambos existen.

El checkpoint ya no es exactamente el peso inicial: **1.802 entradas** del W histórico difieren de `0,03*C*signo`, con diferencia absoluta máxima 4,4281. Conserva una historia de plasticidad; regenerar W desde anatomía la borraría.

En la preparación de campaña 40, los pesos efectivos CPU y CUDA son **exactamente iguales entre sí** y difieren de W histórico en **1.852 entradas**. La cifra corresponde a las políticas receptoras explícitas R8→Mi1 (**826**) y R8→Mi4 (**1.026**), que convierten el signo efectivo seleccionado a excitatorio. La segunda implementación declara `abs(stored_weight)` y conserva magnitud/anatomía; estas políticas son hipótesis de cotransmisión/receptor, no prueba eléctrica de los pares machos. [r8_mi4_visual_brain.py:69](/home/daroch/AXIOMA_FLYWIRE/matrix/src/r8_mi4_visual_brain.py:69). No es un error de transposición ni 1.852 conexiones nuevas.

El operador principal preparado tiene **14.746.989 entradas positivas, 9.812.146 negativas y 1.023.803 cero**. No describe por sí solo toda la transmisión efectiva: APL sustituye pesos temporalmente por liberación regional, PN tiene puertos locales y varias clases recalculan objetivos sobre filas. [gpu_coefficient_layout.py:20](/home/daroch/AXIOMA_FLYWIRE/matrix/src/gpu_coefficient_layout.py:20) y [kc_spatial_brain.py:91](/home/daroch/AXIOMA_FLYWIRE/matrix/src/kc_spatial_brain.py:91) muestran esas escrituras y reemplazos.

La revisión anterior ya encontró **5.845 pesos distintos** entre un CSR capturado y un loader fresco con topología igual; eran 5.062 filas, diferencia máxima 14,3151. Se conserva como advertencia concreta de identidad del operador, no como fallo actual sin reparar. [graph_weight_diff_01/RESULT.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/owner_fallback_cost_20260924_01/graph_weight_diff_01/RESULT.json). El motor 11 exige el operador de su propia preparación y declara sus posiciones mutables; comparar sólo anatomía o sólo hashes de código no basta.

## 4. Macro y meso: recurrencia y posibles rutas

| Medida nueva | Grafo anatómico C | Soporte principal efectivo preparado |
|---|---:|---:|
| Aristas no nulas | 25.582.938 | 24.559.135 |
| Componentes fuertemente conectados | 1.385 | 4.803 |
| Nodos del componente mayor | 165.314 (99,17 %) | 161.897 (97,12 %) |
| Aristas dirigidas con arista recíproca, incluyendo autápticas | 7.647.621 (29,89 %) | No recalculado |

El grafo es disperso en almacenamiento y ampliamente recurrente. Grado de salida mediano **112**, p99 **794**, máximo **11.203**; grado de entrada mediano **98**, p99 **858**, máximo **11.526**. Hay 1.220 nodos sin salida interna y 370 sin entrada interna; los 217 aislados internos de la procedencia no equivalen a neuronas sin conexiones originales, porque existe frontera.

Las conexiones dentro de una misma superclass suman **18.446.856 pares / 84.813.440 contactos**. Las clases no definen bloques independientes. Algunas rutas anatómicas agregadas, sin atribuirles caudal funcional:

| Fuente → destino | Pares | Contactos |
|---|---:|---:|
| OL intrínseco → OL intrínseco | 8.799.556 | 35.691.182 |
| CB intrínseco → CB intrínseco | 6.815.307 | 34.365.451 |
| VNC intrínseco → VNC intrínseco | 1.831.764 | 11.042.345 |
| OL intrínseco → proyección visual | 1.898.506 | 7.714.143 |
| Proyección visual → CB intrínseco | 551.521 | 3.847.546 |
| CB intrínseco → descendentes | 299.360 | 2.732.731 |
| Descendentes → VNC intrínseco | 246.077 | 1.718.761 |
| VNC intrínseco → motor VNC | 155.068 | 2.214.610 |
| VNC intrínseco → ascendentes | 274.737 | 1.636.238 |
| Ascendentes → CB intrínseco | 229.460 | 1.872.674 |

Existen por tanto rutas ascendentes, descendentes y bucles; no se desprende del número de contactos cuál domina una conducta, ni que CNS→VNC→MN sea el lector corporal actualmente usado. El ensayo necesita seguir por separado la decodificación y la prótesis que finalmente aplican fuerzas.

Se reutilizó el [cono estructural existente](/home/daroch/AXIOMA_ASTRA/motor_nuevo/owner_fallback_cost_20260924_01/EVENT_CONE_RESULT.json): siete fuentes reales de un bloque sham alcanzaban 0,76 % de nodos a un salto, 8,76 % a dos, 61,66 % a tres y 97,84 % a cuatro. A tres saltos la salida del conjunto ya cubría el 76,95 % de aristas activas. No es una velocidad de propagación: no incorpora tiempos, amplitudes, cancelaciones o todos los operadores especializados.

El [ensayo anterior de lectores efectivos](/home/daroch/AXIOMA_ASTRA/motor_nuevo/owner_fallback_cost_20260924_01/effective_event_readers_01/EFFECTIVE_EVENT_READER_RESULT.json) perturbó siete puertos en un único extremo de bloque y cambió 182 filas CNS, de ellas 104 lectores genéricos del primer vecindario. Es evidencia del operador y sus propietarios; no una intervención temporal biológica. Este contraste impide usar la máscara CSR como sustituto automático del mapa de dependencias efectivo.

## 5. Micro: células y circuitos realmente representados

| Parte | Estado/modelo presente en la preparación | Límite verificable |
|---|---|---|
| CNS general | 166.700 coordenadas neuronales dentro de **359.373** coordenadas híbridas FP64, además de estados especializados externos. **105.269 IDs visuales** y **5.529 IDs de entrada fotoreceptora**. | El número de coordenadas no es número de neuronas. El esquema general visual usa voltaje graduado y el resto tasas normalizadas; varias filas son reemplazadas después. No hay 166.700 reconstrucciones de membrana individuales. |
| PN nativa | Una DM1_lPN, **ID 10208 / fila 189**, acoplada a receptores, geometría, calcio y puertos; 178.818 coordenadas exteriores y 20 internas reducidas del modelo de masa. | **`full_PN_replacement=false`**. El volumen original omitido por reducción tiene 38.757.888 coordenadas, no neuronas. La eliminación algebraica posterior reconstruye coordenadas del modelo reducido; no valida la reducción física original. |
| Entradas PN | 74 ORN, 217 pares ACh adicionales, 1 APL y 158 pares inhibitorios: **450 rutas no nulas**, con 43 `zero-direct` pendientes. | La amplitud por par/familia y la transferencia de receptores siguen siendo hipótesis. La cifra 450 es cobertura de rutas del modelo, no reconstrucción de toda sinapsis/receptor. |
| Salidas PN | 466 receptores eléctricos KC/APL; otros **629 consumidores** están mapeados. | **`general_outputs.enabled=false`** al inicio y el runner lo exige. Los 629 mantienen la transmisión genérica. «PN629» no son 629 células PN modeladas físicamente. La PN contralateral 10176 conserva otra representación. |
| KC/APL | 4.064 filas: **4.062 KC + 2 APL**. APL tiene **15 regiones por célula**. | Regiones/contactos no son distancias electrotrónicas ni áreas medidas. El acoplamiento APL es un cierre pasivo por fracciones de contactos; la escala absoluta APL sigue sin identificar. |
| KC gamma espacial | **1.557** instancias con `delta[1557,17]` y `gates[1557,17,4]`; liberación axonal `q/s[1557,12]`. | Se reutiliza una referencia **WT9 de 17 puertos**. PN se distribuye uniformemente en dendritas; APL por ROI y otros inputs en soma proxy. No son 1.557 morfologías eléctricas nativas identificadas. Las KC restantes conservan el componente LIF. |
| PN→KC y APL→KC | 299 fuentes PN en filtros; 9.866 posiciones PN para escalas eléctricas y 1.557 posiciones APL; receptor PN→KC tiene su propia selección de 1.375 destinos. | Miniatura somática y contactos/18 son anclaje y regla de transferencia declarados, no conductancia unitaria identificada de cada par. Las selecciones de cada propietario no deben sumarse como neuronas distintas. |
| Visión especializada | Fototransducción en 3.021 destinos; frontera retiniana externa para 1.771 L1/L2; Mi9→T4 en 31.092 pares; GABA T4 en 6.860 destinos; otros componentes CvN7, PVLP, R8/Mi1/Mi4 y descendentes. | Existen priors y aparatos retinianos protésicos. Presencia de un componente y `enabled=true` no prueba sustitución fisiológica de toda visión. La frontera L1/L2 añade entrada óptica sin nuevas neuronas canónicas. |
| Visión→KC | 1.349 pares VPN/LVIN→KCgamma-d repartidos en 206 filas. | Traslada conductancia existente a puerto dendrítico uniforme; sin posición de cada garra visual ni calibración eléctrica pareada. |

Los manifiestos y formas se leyeron de los arrays del estado, resumidos en [STATE_AUDIT.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/STATE_AUDIT.json). El orden efectivo de reemplazo se observa en [gpu_coefficient_layout.py:74](/home/daroch/AXIOMA_FLYWIRE/matrix/src/gpu_coefficient_layout.py:74); una lista de clases habilitadas no establece por sí sola quién determina una fila.

La prueba más directa del alcance PN está en [pn_general_output_brain.py:44](/home/daroch/AXIOMA_FLYWIRE/matrix/src/pn_general_output_brain.py:44): exige 629 receptores generales distintos de las filas dinámicas y partición de 1.095 consumidores totales. [run_trial.py:79](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/run_trial.py:79) exige que ese reemplazo general permanezca apagado.

## 6. Unidades, normalizaciones e historia

| Cantidad | Interpretación válida |
|---|---|
| C | Conteo entero de contactos anatómicos agregados por par. No pA, nS ni probabilidad de liberación. |
| W | Eficacia de interfaz inicial `0,03*C*signo`, luego modificaciones registradas. Distintos receptores usan la misma anatomía con escalas e interpretaciones diferentes. |
| Estado CNS base | Tasas normalizadas o voltaje visual afín. Para la coordenada visual base `V_mV=80*u−80`; salida saturante `clip((80*u−15)/40,0,1)`. No aplicar esta conversión a cada coordenada de todo el vector. |
| `q`, transmisión y tasas publicadas | Liberación/actividad normalizada y filtros; `q*cap` es un proxy en Hz en interfaces heredadas. APL regional es adimensional; no debe interpretarse como frecuencia de disparo. |
| Conductancia visual genérica | `abs(W)*(1/30)*release` es conductancia relativa a la fuga en esa ley. No asigna directamente nS por contacto. |
| Membranas y puertos especializados | V en mV, C en nF, G en nS, carga en pC cuando así los declara el propietario. Convertirlas exige el puerto correspondiente. |
| Relojes | ns enteros para sesión/acoplamiento; segundos para ecuaciones y eventos dentro de bloque. Fechas de archivos y tiempos de adopción no son observaciones fisiológicas. |

Las ecuaciones base y las conversiones están en [hybrid_visual_brain.py:26](/home/daroch/AXIOMA_FLYWIRE/matrix/src/hybrid_visual_brain.py:26), [gpu_visual_brain.py:23](/home/daroch/AXIOMA_FLYWIRE/matrix/src/gpu_visual_brain.py:23) y [pn_cns_ports.py:29](/home/daroch/AXIOMA_FLYWIRE/matrix/src/pn_cns_ports.py:29). En 11 el RHS neuronal común es `rate*(target−state)`; cambiar integrador no identifica las ecuaciones ni sus parámetros.

Los priors originales son normales independientes truncadas a positivo: tau 20±2 ms, gain 1±0,1, theta 7,5±0,6 y cap 200±10 Hz, semilla 0. La heterogeneidad aleatoria no equivale a heterogeneidad celular medida. El checkpoint heredó una normalización **global por volumen de segmentación**: `s=size/mediana`, `gain/=s`, `theta*=s`; 166.678 volúmenes observados y 22 no observados preservados sin imputación. Mediana **189.014.792,5 voxeles fuente**. El volumen no es capacitancia, superficie ni resistencia de entrada; tampoco es la mediana de una población seleccionada en otro estudio. [anatomical_morphometry.py:78](/home/daroch/AXIOMA_FLYWIRE/matrix/src/anatomical_morphometry.py:78).

Parámetros efectivos inspeccionados: tau entre 11,012 y 29,464 ms; theta entre 0,002515 y 4.632,314 unidades heredadas; `rate_gain=gain/cap` entre 7,838e−6 y 13,226. Son finitos y positivos; su amplitud muestra por qué no se debe atribuir rigidez, saturación o actividad exclusivamente a la topología. `preparar_candidata` homologa tau/theta de los dos PN y dos DNb05. El loader de motor14 fuerza ambas escalas ipsi/contra a 1, aunque el plan histórico conserva 1,4 y 0,1. [matrix_olfactory_diagnostic.py:110](/home/daroch/AXIOMA_FLYWIRE/matrix/src/matrix_olfactory_diagnostic.py:110), [motor_runtime.py:21](/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922/motor_runtime.py:21).

La plasticidad de esta preparación está **deshabilitada**, aunque guarda 41.654 actualizaciones históricas, 1.839 posiciones candidatas KC→MBON05, factores y trazas de elegibilidad. La regla afecta una selección gamma4/PAM08, no todos los pares del conectoma. Desactivar aprendizaje conserva historia y actualización de elegibilidad; no restaura pesos iniciales. [anatomical_plasticity.py:81](/home/daroch/AXIOMA_FLYWIRE/matrix/src/anatomical_plasticity.py:81). Ni historia heredada ni plasticidad habilitable acreditan aprendizaje nuevo durante el ensayo actual.

## 7. Tiempo: lo observado por el simulador y lo no identificado

La tabla de aristas upstream contiene `body_pre`, `body_post`, `weight`. **No contiene timestamps de espigas, retardos por par, respuestas receptoras ni historial de plasticidad.** Los tiempos que usa el motor proceden de hipótesis, otros recursos o decisiones numéricas:

- Filtro común de transmisión de **5 ms**: prior de modelo, no medición por célula. La migración histórica inicializó la transmisión igual a la liberación instantánea porque no disponía de historia anterior.
- PN→KC conserva filtros de 0,478758/4,85 ms y una interpretación explícita de la subida de una miniatura somática; no son tiempos extraídos del conectoma.
- APL→PN y familias inhibitorias declaran retardo **125 µs**, tau rápida **15 ms** y alternativa lenta **30/400 ms**; conductancia y cinética transferidas/hipotéticas. El modo guardado es `fast`.
- El manifiesto PN conserva `coupling_ns=15625` de una adopción anterior. **El ejecutor actual instala bloques de 125.000 ns**, avanza PN media época antes y después y ejecuta predictor/restauración. [runtime_session.py:35](/home/daroch/AXIOMA_ASTRA/motor_nuevo/pipeline_review_20260922/runtime_session.py:35), [block_midpoint.py:102](/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922/block_midpoint.py:102). No usar el campo heredado aislado para describir el paso efectivo.
- `control_ns=1.000.000`, `physical_dt_s=25 µs` y `motor_delay_ns=1.000.000` son contrato de control/cuerpo. Tampoco son retardos axonales observados.

El registro de eventos de 11 guarda ID y fila, productor `gamma_cuda` o `nongamma_lif`, tiempo relativo, salto y valor posterior. Su referencia temporal requiere bloque y reloj de origen; un `time_s` solo no es tiempo absoluto. [event_coupling.py:67](/home/daroch/AXIOMA_ASTRA/campanas/etapa3_pn629_intervention_20260923_15/event_coupling.py:67). El resultado de 1 s suma 77.960 registros en 16.000 bloques del esquema predictor/aceptado. **No debe convertirse ese total en tasa poblacional real ni en eventos comprometidos únicos** sin separar predicciones descartadas. Sus 102.219.560 aceptaciones celulares se declaran `cell_trials`, no neuronas, espigas ni pasos globales. [candidate_1000ms_01/RESULT.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/candidate_1000ms_01/RESULT.json).

El estado también conserva textos de adopciones anteriores: el manifiesto ORN antiguo describe entrada held de 7 Hz, mientras la capa endógena y periférica vigente usa la actividad canónica y `9+83,667*clip(drive/80,0,1)` Hz con modulación recurrente acotada. No son dos mediciones compatibles que deban promediarse; prevalecen banderas, propietarios y código ejecutados.

Un conectoma estático permite calcular caminos, recurrencia, fronteras y distribución de contactos. No determina por sí solo orden de activación, desfases, oscilaciones, sincronía, memoria, latencia de corrección o flujo causal efectivo. Esos resultados pueden medirse **en el modelo**, conservando la etiqueta de simulación; para validación biológica requieren observables/intervenciones independientes pertinentes. Más contactos no prueba más corriente, excitación o importancia conductual.

## 8. Hallazgos que deben quedar visibles antes de 2 s

| Hallazgo verificable | Implicación práctica |
|---|---|
| IDs, transposición y CSR son coherentes; conteos y parámetros inspeccionados finitos | No cambiar importador u orientación para corregir un fallo no observado. |
| `all` es canónico seleccionado, con frontera upstream grande y signo cero parcial | Describirlo como «modelo heterogéneo del grafo canónico MaleCNS v1.0», y reportar sus fronteras. |
| Pesos de anatomía, W histórico, W efectivo y reemplazos por propietario son objetos distintos | Congelar todos los niveles; las estadísticas de C no validan la corriente que se integra. |
| PN general está apagado, plasticidad deshabilitada, visión/morfología contienen prótesis | El ensayo puede evaluar conservación numérica y respuesta del organismo definido; no fisiología cerebral completa, aprendizaje nuevo o sustitución íntegra de PN. |
| Textos/tiempos de adopción permanecen en capas heredadas | Resolver cada parámetro por estado y precedencia de ejecución; conservar procedencia sin interpretar cada texto histórico como configuración activa. |
| Red muy recurrente, gran variedad de grados y operadores especializados | No fijar regiones rápidas o eliminar términos por distancia anatómica, tipo celular o ausencia de espigas sin control de error e influencia efectiva. |

La revisión no valida todos los hashes del árbol transitivo ni reproduce todo mecanismo especializado: usa el contrato de carga, los hashes propios de los artefactos inspeccionados y evidencia anterior identificada. No atribuye una señal a una fuente causal por mera coincidencia estructural.

**Estos límites no bloquean una comparación de motores** que conserve el mismo modelo, preparación, intervención y observables. PN general apagado, fronteras y priors deben ser iguales y explícitos en ambos brazos; no hay razón de dataset para cambiarlos antes de las dos continuaciones de 2 s. La comparación numérica no necesita resolver primero toda la biología que el modelo deja pendiente.

## 9. Oportunidades de arquitectura derivadas de estos datos

Son tres direcciones rivales, no una selección por analogía biológica ni permiso para modificar el ensayo congelado:

| Alternativa | Patrón que la motiva | Discriminador y falsador |
|---|---|---|
| **A. Compilar operadores heterogéneos con propiedad explícita de filas/posiciones**: bloques de filas por ley, CSR inmutable separado de deltas de receptores, buffers persistentes con versión | Muchas filas se recalculan tras la evaluación genérica; sólo 6.474 posiciones declaradas mutables de 25,58 M, pero hay reemplazos fuera de W. Grados muy variables y 1 M ceros explícitos | Medir trabajo genérico realmente reemplazado y ahorro integral. Falla si omite una dependencia, cambia precedencia o mueve más memoria de la que evita. Los ceros pueden omitirse del cómputo sólo si el contrato demuestra que no van a activarse; se conserva el registro anatómico. |
| **B. Operador incremental por fuentes, CSC de cambios más fondo continuo**: eventos actualizan columnas/colas y el componente graduado conserva integración continua | Lectores directos de un evento son pequeños, pero la recurrencia amplía rápido su influencia. La criba histórica `event_sparse` cubría sólo 885.587 aristas, 3,46 % de W | Exigir coste total de mantener deltas y recomponer todos los propietarios, comparando corrientes y trayectoria. Falla si trata ausencia de espiga como ausencia de señal o si el fondo continuo domina y elimina la ganancia. No extrapolar velocidad de la primitiva al CNS. |
| **C. Integración por bloques recurrentes con residual y reconstrucción de fronteras**: partición computacional, corrección Krylov/Schur o distinta resolución temporal por error | Gran componente recurrente, grupos con fuerte conectividad interna, una PN con matriz de masa y 1.557 plantillas KC repetidas | Probar bloques y residuos sobre el operador efectivo, incluida memoria/receptores, sin asumir que superclass define independencia o bajo rango. Falla con evento tardío omitido, error de frontera no detectado o coste de correcciones superior al ahorro. Los negativos MRI/RHS congelado previos siguen vigentes. |

La decisión inmediata de datos es mantener el checkpoint completo y hacer que futuros módulos declaren **IDs, unidades, soporte de lectura/escritura, historial, retardos, fronteras y nivel de evidencia**. Eso permite sustituir una prótesis sin perder rutas silenciosamente. No requiere descargar otro conectoma ni inventar una nueva fisiología antes de medir el motor actual.

## 10. Evidencia y reproducción de esta revisión

Se reutilizaron revisión 10, cono estructural, lectores efectivos y diferencia de pesos de `owner_fallback`. Los únicos cálculos nuevos de dataset fueron inspección de grafo y estado, con límites prospectivos de **120 s / 6 GiB por cálculo**: grafo **5,576 s / 1.586.632 KiB RSS**; estado **1,649 s / 601.072 KiB RSS**. El primer resumen de estado falló al calcular un cuantil booleano; se corrigió sólo el resumen, se conservó [STATE_AUDIT_REPAIR.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/STATE_AUDIT_REPAIR.json) y se repitió ese modo bajo el mismo límite. La corrección no afecta el cálculo de grafo entero/flotante; su recibo conserva el hash del script anterior a la corrección.

[Plan](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/PLAN.json), [script](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/audit_dataset.py), [grafo](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/GRAPH_AUDIT.json), [estado](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/STATE_AUDIT.json). El Python de inspección CPU es distinto del runtime congelado del organismo y se registra en la evidencia; no se atribuye reproducción del motor a estos cálculos.

```bash
python3 /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/audit_dataset.py graph --out /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/reproduction_01
python3 /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/audit_dataset.py state --out /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/data/reproduction_01
```

El script entregado añade salida exclusiva y rechazo de sobreescritura; los recibos originales conservan el hash del script que los produjo. Al repetir, elegir otro directorio si el anterior existe. Esta protección de entrega no cambia cálculos ni fuentes.

No se consultaron ni descargaron fuentes externas nuevas: la decisión se resuelve con artefactos locales reales y su procedencia. Los textos fisiológicos heredados se reportan como hipótesis/alcances del modelo, sin convertirlos en validación independiente.
