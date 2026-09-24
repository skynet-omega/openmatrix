# Producto recurrente por eventos: alcance matemático y prueba barata

**Resultado.** Para los puertos `q/s` que sólo decaen y reciben saltos, sus aportes a una matriz **fija durante el segmento** pueden propagarse analíticamente y actualizarse por las columnas CSC de las células que disparan. Esto es una optimización exacta en aritmética real de **ese subconjunto**. No reemplaza el producto recurrente de todo el CNS actual: las 166.700 coordenadas de transmisión `s` de base reciben una liberación graduada que cambia continuamente con el estado, y varios adaptadores cambian la ponderación efectiva o reemplazan filas. La hipótesis «un barrido CSR por evento en vez de por trial para todo el cerebro» queda falsada por el contrato de ecuaciones existente, antes de medir velocidad.

## Contrato que se inspeccionó

- `graph_core.py:42-49` construye dos soluciones de punto medio exponencial por trial (ruta completa y dos medias), con **seis evaluaciones de coeficientes**. Se evalúan estados especulativos y se puede rechazar el trial: un acumulador con mutación cronológica debe poder volver atrás, o evaluar cada estado tentativo de forma pura.
- `gpu_coefficient_layout.py:225-240` usa, para el grafo CNS base, `state[transmission_start:]` como liberación que entra en el CSR. La misma función fija el objetivo de cada transmisión como una función del estado somático/visual y su tasa como `1/synaptic_tau_s`. En `synaptic_visual_brain.py:1-8,79-97`, la ecuación es `tau_s ds_j/dt = r_j(x_j)-s_j`, con `tau_s=0.005 s` como candidato común. Las células visuales tienen `r_j=clip((80*x_j-15)/40,0,1)` y las demás `r_j=x_j`; el objetivo de `x_j` depende a su vez del producto recurrente y, para filas no visuales, de `tanh` (`gpu_visual_brain.py:13-42`).
- `event_ports.py:1-20` proyecta algunos `q_j,s_j` desde el inicio del segmento y la lista ordenada de saltos. El adaptador los sitúa tanto en filas somáticas como de transmisión, y les asigna tasa cero antes de la proyección (`organism_adapter.py:23-36`). En este puerto sí rigen `q'=-q/tau_q`, `s'=(q-s)/tau_s`, con `tau_s` común y `tau_q` por fila. El evento registrado incluye salto **efectivo tras clipping/SET**, no un impulso nominal (`physical_events.cu:12-18`, `event_waveform.py:61-76`). `event_ports.py` consume tiempos y saltos efectivos; su prueba numérica debe conservar la convención `event_time <= query_time` y el orden de SET repetidos.
- Para una fila visual, el objetivo necesita **conductancias positivas y negativas por separado**, no sólo la suma firmada (`gpu_visual_brain.py:23-40`). Las filas especiales de KC, retina, GABA, PN y adaptación reemplazan partes del operador en `gpu_coefficient_layout.py:64-223`.
- `pn_general_output_brain.py:77-98` multiplica pesos seleccionados por una salida PN efectiva durante cada evaluación; `kc_apl_dynamic_brain.py:197-204` instala pesos efectivos de APL al inicio de un bloque de acoplamiento y los restaura. El peso anatómico puede estar fijo mientras `W_eff` de la evaluación cambia. La sincronización de plasticidad también actualiza el vector de pesos (`gpu_visual_brain.py:69-71`).

## Ecuaciones para el subconjunto impulsivo

Para un conjunto de fuentes impulsivas `E`, sea `W_ij` la contribución efectiva de `j` a la fila `i`, sin cambios de peso dentro del segmento. Si todas las fuentes de un grupo `g` comparten `tau_q,g` y `tau_s`, definimos

`Q_i,g = sum_{j in E_g} W_ij q_j`, `S_i,g = sum_{j in E_g} W_ij s_j`.

Entonces, entre eventos, `Q'=-Q/tau_q,g` y `S'=(Q-S)/tau_s`. En un intervalo `h`,

`Q(t+h)=Q(t) exp(-h/tau_q,g)`,

`S(t+h)=S(t) exp(-h/tau_s) + Q(t) C(h;tau_q,g,tau_s)`,

`C=[exp(-h/tau_q)-exp(-h/tau_s)]/[1-tau_s/tau_q]`, con límite `C=(h/tau_s) exp(-h/tau_s)` cuando los tiempos son iguales. Es la misma convolución que implementan `event_ports.py:10-19` y `event_waveform.py:7-16`. Ante un salto efectivo `Delta q_j` en `t_e`, `Delta Q_i,g=W_ij Delta q_j` para cada destino en CSC columna `j`, mientras `S` es continuo en `t_e`. Si se registra un **SET** `q_j := q_post`, primero se calcula `Delta q_j=q_post-q_pre` con los eventos previos del mismo timestamp ya aplicados. Usar sólo `W Delta s` en el instante de la espiga sería incorrecto: `Delta s=0` justo allí.

Si los `tau_q,j` difieren, la reducción a un solo `(Q_i,S_i)` deja de ser cerrada: `Q'_i=-sum_j W_ij q_j/tau_q,j`. Se necesitan grupos de valores **exactamente iguales**, un estado modal por fuente o una aproximación certificada. Agrupar tiempos «casi iguales» cambia el modelo. También hay que mantener `Q+`, `Q-`, `S+`, `S-` por separado en filas visuales, clasificadas por el signo del peso/conductancia; la suma firmada sola no reconstruye `target` y `rate`.

Para la transmisión graduada base, `I_i=sum_j W_ij s_j` satisface, si `tau_s` es común y `W` fijo,

`I'_i = [sum_j W_ij r_j(x_j)-I_i]/tau_s`.

El término `W r(x)` no desaparece: `x` evoluciona por recurrencia y `r` puede ser saturada o recortada. Por tanto el decaimiento de `I` no es una solución cerrada entre espigas. Un ejemplo de una sola conexión con `x(t)=exp(-t/tau_x)`, `s(0)=0`, `x(0)=1` y **ningún evento** da `s(t)=[exp(-t/tau_x)-exp(-t/tau_s)]/[1-tau_s/tau_x]`; si `tau_x=tau_s`, `s(t)=(t/tau_s)exp(-t/tau_s)`. En `t=tau_s` vale `1/e`; un agregado que sólo decae desde `s(0)=0` predice cero. El error aparece sin eventos ni rigidez.

Hay una identidad algebraica alternativa válida para cualquier cambio de estado: `I_new=I_old+W Delta s`. Con CSC evita trabajo sólo si el coste ponderado `rho=sum_{j:Delta s_j != 0} degree_out(j)/nnz(W)` es bastante menor que uno. Si cambian casi todas las fuentes, el scatter CSC toca casi todas las aristas y añade conflictos de escritura y un orden de suma distinto. Un umbral `|Delta s|<epsilon` vuelve esa identidad aproximada; la cota local es `|delta I_i| <= sum_{j omitidos}|W_ij| |Delta s_j|`, pero debe propagarse por la red recurrente y sus posibles cruces de evento antes de aceptarla.

## Condiciones de admisión y falsadores

1. Separar en el IR las fuentes `impulso_qs`, `graduada_s`, `peso_efectivo` y operadores que sustituyen filas. El núcleo puede mantener agregados exactos para la primera clase, pero conserva la evaluación requerida para las otras. Los pesos de PN/APL y plasticidad obligan a reconstruir o corregir el agregado en cada frontera donde cambian. Una variación de peso dentro de un trial sin invalidación falsaría la caché.
2. Mantener la semántica de fase: puerto se proyecta al inicio, medio y fin de cada trial, el evento en `t_e` ya es visible para consultas `t>=t_e`, y una fase SET sustituye `q` antes de añadir eventos posteriores. Un evento tardío dentro de un trial, el caso `tau_q≈tau_s` y dos SET simultáneos son tres casos mínimos de regresión. La ruta del agregado debe soportar predictor, mitad, paso fino y rollback tras rechazo.
3. La equivalencia anterior es matemática, **no bit a bit FP64**. La propagación CSC y la reducción CSR cambian orden de sumas, y `conv` usa una serie cerca de tiempos iguales. Comparar corrientes, `target/rate`, estados finales, eventos y comandos con el contrato congelado; nunca reducir tolerancias a posteriori para admitirla.
4. Falsador de utilidad: si el ahorro del subconjunto impulsivo queda oculto por la evaluación graduada y los adaptadores, o `rho` de la actualización CSC es alto en trials reales, esta técnica no es la reforma completa del motor. Conservarla, a lo sumo, como operador especializado dentro de una arquitectura general.

## Medición barata con datos reales, sin reejecutar el organismo

Inspeccioné los `session.npz` guardados del brazo `reference_minus_01` de la campaña de etapa 4. `array_261` es el estado `hybrid` según su `session.json`; `n=166700`, `photo_ids` tiene 5529 elementos y `transmission_start=n+2*5529=177758`. Entre `state_100ms` (`time_ns=44586000000`) y `final_state` (`44886000000`), **300 ms**, cambiaron exactamente 128.289/166.700 transmisiones de base; 123.265 superan `1e-9` y 104.706 superan `1e-6` en valor absoluto. El máximo fue `0.5507571191` y la mediana `3.1728273e-6`. Cambiaron exactamente 128.286 estados neuronales. Son datos reales del modelo, pero **no miden sparsity por trial**: no extrapolar estas fracciones a un paso de microsegundos. Sí refutan usar los conteos de espigas como conteo completo de fuentes que cambian durante un intervalo largo.

Prueba decisiva siguiente, presupuesto previo recomendado: un único segmento guardado de **1 ms**, sin cuerpo ni campaña conductual; instrumentar las seis evaluaciones de coeficientes de cada trial ya ejecutado para registrar sólo `(Delta s_j, degree_out(j), W r(x), eventos)` y la corriente agregada, hasta 250 trials y sin cambiar aceptación. Calcular `rho` exacta y por tolerancias `1e-12,1e-9,1e-6` sólo como diagnóstico; comparar el agregado impulsivo reconstruido por CSC con la reducción CSR en las mismas marcas temporales. Primera puerta: corriente, `target/rate`, estado y lista de eventos dentro del error originalmente congelado en la ruta exacta; segunda puerta: tiempo GPU de CSR frente a CSC+propagación **incluyendo** construcción, invalidaciones y sincronización. Si `rho` o el tiempo no mejoran materialmente, cerrar la alternativa CSC global sin una campaña larga. La prueba no requiere inventar una regla celular ni ajustar el olor.

Proveniencia de esta nota: fuentes `graph_core.py` SHA256 `898dbf27...`, `event_ports.py` `e4faa68c...`, `organism_adapter.py` `355666d4...`, `gpu_coefficient_layout.py` `2598415d...`, `gpu_visual_brain.py` `fd6bc5ec...`, `kc_apl_dynamic_brain.py` `b1cdd582...`; snapshots `state_100ms/session.npz` `dae2c195...`, `final_state/session.npz` `ad6fef2f...`. Todas las rutas se refieren al árbol `/home/daroch/AXIOMA_ASTRA` y esta inspección fue sólo de lectura salvo la presente nota.
