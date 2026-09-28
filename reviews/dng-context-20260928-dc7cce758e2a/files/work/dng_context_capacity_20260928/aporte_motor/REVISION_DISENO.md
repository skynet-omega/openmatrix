# Revisión de diseño: aporte y capacidad directa hacia DNg100

28-09-2026 · Motor C++/CUDA · sin CNS ni GPU.

**El diseño es aplicable a los operandos49, con tres precisiones operativas antes del cálculo. Mantendría los grupos elegidos.** No calculé aportes, capacidades, rankings ni nuevas sumas de las seis neuronas. Revisé las fuentes del consumidor, el observador, el integrador y el codec; decodifiqué las diez capturas sólo para comprobar dominio, identidad, máscaras y soporte temporal. [Comprobaciones y procedencia](COMPROBACIONES_DISENO.json).

## 1. La variable acotada es la transmisión consumida; el máximo requiere la fila completa

El consumidor de DNg100 ejecuta `included * W * (transmission * cap)` en FP32. `included` procede de `connected || !visual[pre]`. Estos destinos son filas genéricas: **no se aplica aquí el factor de conductancia `scale` de la rama visual**. La captura guarda los cuatro operandos de la misma llamada. Las capacidades son positivas y los pesos/capacidades/máscaras permanecen constantes dentro de las diez capturas inspeccionadas. Los parámetros y el operador entre capturas deben seguir ligados a la selección congelada mediante el verificador existente.

Para una fila y evaluación, la cota algebraica del grupo G es:

`U_G = Σ(i en G) included_i * max(W_i * cap_i, 0)`.

La positividad de `cap` permite elegir `q*=1` cuando W>0 y `q*=0` cuando W<0. Para W=0 cualquier q vale; conservar su q facilita la trazabilidad. Esto combina excitación y silenciamiento de aportes negativos: **no es la respuesta de «activar todo el grupo»**. Es un máximo de operandos libres en el dominio del modelo, manteniendo el resto y el operador congelados. No acredita que un estímulo, la cinética o el circuito puedan producirlo.

Para la cota operacional, sustituir únicamente esas transmisiones, formar de nuevo **toda la fila en su orden original** y reutilizar `warp`. No comprimir las aristas elegidas: cambiaría sus carriles. Tampoco obtener el nuevo neto mediante `net_actual − aporte_G + máximo_G`; la suma FP32 no es distributiva. Mantener el producto `W*(q*cap)` y calcular el margen con el mismo orden de `net+drive−theta`.

Para describir el aporte del grupo y del resto, declarar la convención: por ejemplo, sumar en FP64 los términos individualmente redondeados por el consumidor y mostrar aparte su diferencia respecto al neto FP32 registrado. No exigir que dos reducciones separadas, sumadas después, reproduzcan exactamente el neto original. Esa diferencia aritmética no es otro mecanismo neuronal.

La reducción fija es monótona en sus términos finitos, por lo que los extremos anteriores dan el máximo operacional de esta intervención en el espacio de operandos. Comprobar finitud del resultado extremo y ganancia positiva del destino; esta última se cumple para ambas DNg100. No hay duplicados presinápticos dentro de las filas congeladas ni signos opuestos entre los aferentes compartidos de las dos DNg. Aun así, presentar primero cada cota por destino, sin inferir realizabilidad neuronal conjunta.

**Fuentes:** [consumidor, línea 16](../../../campanas/etapa45_operands_20260927_49/coefficient_capture.cu#L16); [reducción y verificaciones, líneas 12–56](../../../campanas/etapa45_operands_20260927_49/verify_operands.py#L12); `frozen_selection.npz`.

## 2. El dominio 0–1 está confirmado en las muestras elegidas, no por el mero nombre del integrador

`RealCNS` declara límites 0–1, pero el kernel `estimate` comprueba el candidato final y la finitud de las derivadas; eso solo no prueba que todos los estados intermedios RK estén dentro del dominio. El verificador49 comprueba finitud de los operandos y rango del target, **pero no exige explícitamente 0≤transmission_consumed≤1**.

Comprobé directamente ese dominio en todos los RHS aceptados de las épocas comprometidas de las diez capturas y en los tres grupos propuestos: se cumple. Añadir esta condición específica al nuevo análisis es pertinente; no hace falta reparar ni repetir la adquisición. Una extensión a otras muestras debe volver a comprobarla, sin recortar datos fuera del dominio.

El valor que se debe cambiar para la cota es la transmisión ya consumida, no `state` somático ni una tasa publicada multiplicada nuevamente por `cap`. Las transformaciones, filtros y buffers anteriores al CSR pueden vincular transmisión y pesos en un organismo vivo. Por eso congelar esos otros operandos define una **capacidad directa instantánea de esta frontera**, no el máximo de un cambio libre del estado neuronal completo.

Las sobrescrituras especializadas tampoco se deben adivinar desde el tipo celular. El verificador49 ya exige que target y tasa del CSR coincidan con los finales de estas seis filas; reutilizar esa condición. La ausencia de una sobrescritura posterior en estos registros no equivale a que todas las rutas o funciones del organismo sean genéricas.

**Fuentes:** [rango comprobado al final del intento](../../../motor_nuevo/full_pipeline_review_20260925_12/engine/graph_runtime.py#L25); [target/rate y operandos](../../../campanas/etapa45_operands_20260927_49/verify_operands.py#L33); `src/gpu_synaptic_visual_brain.py`, `src/gpu_coefficient_layout.py` y `engine/fp32_operator.py`.

## 3. Conservar identidad anatómica, reloj y la distinción entre historia y causa

Las selecciones exactas en `data/male_v10/nodes.parquet` contienen 1.314 descendentes, 2.383 ascendentes/sensoriales ascendentes y 97 MBON. Son disjuntas. Todos los IDs presinápticos de la selección49 tienen correspondencia canónica; sus filas coinciden con el orden de los 166.700 IDs. Las dos filas DNg no tienen autoaristas en esta selección.

Mantener las etiquetas exactas: no incorporar clases `_tbc`, nombres similares o identidades de otros conectomas. `class` falta en 6.567 de las 7.227 posiciones presinápticas de las **seis filas**, contando cada arista; no son 6.567 neuronas únicas. Esa ausencia debe figurar en cobertura. No equivale a clase conocida distinta de MBON ni autoriza completar la anotación según el resultado. Separar tamaño anatómico del grupo, presencia en las filas, inclusión por máscara y transmisión con peso efectivo distinto de cero.

El emparejamiento temporal propuesto sí tiene soporte: en las diez capturas, el primer intento aceptado de cada época comprometida empieza en tiempo local cero, y las ocho épocas comprometidas por ventana tienen los mismos inicios absolutos entre sham/profile. Usar explícitamente **etapa RHS 0** de ese intento y mantener el instante absoluto como clave. No elegir simplemente cualquier RHS con `stage_fraction=0`: la cuarta evaluación también tiene fracción cero, pero representa el límite izquierdo del extremo final.

Las envolventes de todos los RHS aceptados describen la cobertura registrada; no son intervalos de confianza ni promedios temporales. Las ventanas son los milisegundos 1, 5, 10, 20 y 50 de colas OFF con historias anteriores distintas. Una diferencia de aporte entre ellas no demuestra que el grupo contextual haya causado la diferencia entre historias. El nuevo resultado debe ser aporte/capacidad/cobertura, no otra afirmación de causalidad del grupo.

**Fuentes:** [reloj y propiedad de las épocas](../../../campanas/etapa45_operands_20260927_49/verify_operands.py#L69); [cuarto RHS](../../../campanas/etapa45_operands_20260927_49/verify_operands.py#L39); `capture.cu`, `CAPTURE_CONTRACT_v4.json` y las diez capturas decodificadas.

## Dictamen y límite de esta revisión

Procedería al cálculo con el diseño y los grupos actuales, incorporando las precisiones anteriores. Un máximo que permanece subumbral excluye suficiencia directa de ese grupo bajo el resto congelado **en las evaluaciones cubiertas**. No excluye otros instantes, recurrencia, neuromodulación ni una regla neuronal distinta. Un máximo favorable sólo habilita una posibilidad algebraica; no el inicio de marcha ni una promoción de etapa.

El verificador49 y su codec son CPU y pueden reutilizarse sin cargar CuPy ni el CNS. Su dependencia implícita `HERE/frozen_selection.npz` debe permanecer declarada por contenido en la nueva receta; no basta con declarar únicamente el archivo del verificador.

Coste instrumentado de esta revisión de diseño: **1,2684763 s CPU** (lecturas breves y redacción no cronometradas individualmente), de los 60 s CPU compartidos con la futura revisión de tabla. Cero nuevos CNS, cero GPU y ningún cambio fuera de `aporte_motor`. Queda presupuesto suficiente para esa única revisión final; no la inicié.
