# Procedencia e integridad de la anatomía — 23-09-2026

**No encontré corrupción de descarga ni alteración de los archivos comprobados.** Sí hay transformaciones de modelado y mezclas de preparaciones que impiden tratar el organismo como una mosca individual calibrada. La conducta inesperada no es evidencia de que el conectoma esté mal descargado.

Revisión de sólo lectura del histórico: cero GPU, cero pasos neuronales, ninguna descarga masiva. El chequeo principal consumió 4,70 s CPU. Se consultaron páginas primarias y cabeceras de objetos oficiales; no se repitió la importación completa ni se validaron todos los coeficientes funcionales del organismo.

## Qué datos son y qué se verificó

El grafo de 166.700 nodos procede de **MaleCNS v1.0, macho**, no del grafo femenino FlyWire, aunque el repositorio se llame AXIOMA_FLYWIRE y las anotaciones contengan correspondencias `flywireType`. La descarga oficial distingue esas fuentes. MaleCNS incluye cerebro y cordón nervioso; las correspondencias entre tipos no convierten sus identificadores en neuronas del mismo individuo. La versión 1.0 fue publicada el 08-06-2026 con revisiones de trazado y anotaciones. [Datos oficiales](https://male-cns.janelia.org/download/), [versiones](https://male-cns.janelia.org/release/), [comparación entre sexos](https://male-cns.janelia.org/build/dimorphism_overview/).

Comprobaciones realizadas:

- **Tres tablas originales** —anotaciones, neurotransmisores y conectividad minconf-0.5— y **cuatro SWC** de las dos PN: sus SHA-256 coinciden con los recibos locales; sus MD5, tamaños y generaciones coinciden también con las cabeceras oficiales consultadas ahora. No se descargaron otra vez sus contenidos.
- **57 comprobaciones locales por SHA-256**, todas coincidentes: artefactos del importador, geometría/cables PN, recortes de etiquetas, activos congelados del runtime PN y archivos del cerebro en el checkpoint. Son comprobaciones de identidad respecto de manifiestos, no pruebas de que la segmentación o la física sean correctas.
- **Reconstrucción independiente de la ruta relevante:** recorrí las 151.856.684 filas de la tabla original y comparé todos los pares entre las 74 `ORN_DM1` y las dos `DM1_lPN` con la matriz canónica y con el archivo de contactos espaciales. Coinciden exactamente los **145 pares no nulos y 13.384 contactos**: 6.608 hacia **10176, derecha**, y 6.776 hacia **10208, izquierda**. Cero diferencias de conteo en ese contraste.
- La matriz canónica tiene 25.582.938 pares y 124.177.617 contactos; su orden de IDs coincide con los metadatos. Es la selección local de segmentos con `superclass` no vacía, incluidos 94 tentativos `tbc`; conserva conexiones de un contacto y autoconexiones. No es toda la tabla de segmentos oficial. Una diferencia frente al número publicado de neuronas puede reflejar selección o versión; no demuestra corrupción. Fuente local: `data/male_v10/provenance.json`.
- **Volúmenes usados en normalización:** la tabla derivada de 166.700 IDs y su manifiesto coinciden con los hashes del checkpoint. La fuente remota conserva generación `1780895325252595` y tamaño 4.648.794.426 bytes. Se extrajeron originalmente sólo ID/volumen; **no está verificado localmente el checksum del objeto completo** y no lo descargué. El manifiesto declara 22 volúmenes no observados y conservación de los parámetros previos para esos casos. Esto es una limitación explícita de la extracción, no evidencia de bytes dañados.

Los metadatos identifican 10118/10065 como **DNb05 izquierda/derecha**; DNa02 corresponde a 523769/10360. No deben intercambiarse las afirmaciones fisiológicas entre ambos pares.

## Transformaciones que importan más que volver a descargar

1. **Conteo anatómico → dinámica.** El origen proporciona contactos y etiquetas de neurotransmisor, no conductancias, receptores postsinápticos, potenciales de reposo ni una ley motora. El modelo convierte la matriz de filas presinápticas a filas postsinápticas y aplica signos/escala; después intervienen parámetros, normalización por volumen y sustituciones de circuitos. La coincidencia de hashes conserva esas decisiones, pero no las calibra. Véanse `src/anatomical_rate_brain.py:111` y el `model_metadata` del checkpoint. No reconstruí en esta revisión la igualdad de cada peso efectivo con todas sus intervenciones históricas.

2. **Geometría PN → cable eléctrico.** Los SWC de alta resolución son objetos oficiales con generaciones verificadas. El importador usa **0,008 µm por unidad de posición** y **0,001 µm por unidad de radio**, no un factor único (`scripts/prepare_dm1_native_geometry.py:60`). La segunda conversión es una inferencia documentada a partir del generador histórico; no está confirmado su commit exacto ni validado cada radio eléctrico. El [código histórico de neuclease](https://github.com/janelia-flyem/neuclease/blob/267b0ecb18c3e07db7bcb4c853c0d47c47511d08/neuclease/misc/skeletonize.py) es un contraste de procedencia, no una certificación del archivo generado. La revisión local previa conserva 117/202 componentes y no añade puentes; por tanto, fragmentación del esqueleto tampoco equivale automáticamente a aislamiento eléctrico biológico. No comprobé de nuevo todos los vóxeles contra EM.

3. **La variante funcional actual ya interviene el modelo.** `etapa3_pn629_intervention_20260923_15/full_sham_01/INTERVENTION.json` desactiva el reemplazo general de 629 salidas de la **PN izquierda 10208**, conserva los 466 destinos dinámicos y no cambia el grafo anatómico. Un cambio conductual después de esa operación informa sobre el acoplamiento implementado; no demuestra que se haya reparado una descarga ni justifica editar contactos originales.

## Diferencias reales frente a moscas vivas

El ajuste pasivo utilizado —Rm=20.800 Ω·cm², Cm=0,79 µF/cm², Ri=266,1 Ω·cm— procede de ModelDB 118662/Gouwens–Wilson 2009 y se transfiere a geometría masculina distinta. Aquel estudio usó **hembras de 2–10 días**, normalmente con antenas retiradas; combinó morfologías de preparaciones diferentes y encontró ajustes distintos entre células. Además, distingue reposo con entrada ORN intacta frente a retirada y efectos del sello del electrodo. Por ello no corresponde comparar directamente cualquier voltaje o tasa de ese estudio con el checkpoint activo como si fueran la misma preparación. [Artículo primario, Methods, tablas 1–2 y Discussion](https://wilson.hms.harvard.edu/sites/g/files/omnuum8421/files/wilson-lab/files/gouwenswilson2009.pdf), [modelo depositado](https://modeldb.science/118662).

La tarjeta local de canales PN declara otro traslado: densidades/kinéticas adoptadas de un modelo KC y formas derivadas de motoneurona, sin densidades medidas para esa PN MaleCNS ni SIZ nativa identificada (`evidence/pn_fine_ionic_20260912/model_card.json`). El operador G/M reducido preserva una transformación numérica declarada, no una nueva medición biológica.

Gaudry 2013 separa conducta en esfera con estimulación antenal bilateral de registros PN con una antena retirada. Su asimetría funcional de liberación no es un multiplicador universal deducible del número de contactos. Comparar sexo, individuo, glomérulo, intervención y observable importa; ninguno de esos factores permite atribuir por sí solo el fallo del modelo a variabilidad biológica. [Artículo primario](https://wilson.hms.harvard.edu/sites/g/files/omnuum8421/files/wilson-lab/files/gaudrywilson2013.pdf).

## Comprobación que cambiaría la decisión

| Explicación rival | Discriminador y consecuencia |
| --- | --- |
| A. Descarga o correspondencia rota | Hash oficial discrepante, ID/lado equivocado o diferencias por par ORN→PN exigirían detener el brazo afectado y reparar sólo ese origen/transformación. **El contraste efectuado no encuentra esos fallos**; repetir las descargas completas no está justificado. |
| B. Transformación local o interfaz causal incorrecta | Para reivindicar un mecanismo PN biológico, contrastar la misma transferencia de entrada→voltaje/salida en condiciones de electrodo, entrada y preparación emparejadas, y comprobar la dependencia del resultado respecto de radios/continuidad y del reemplazo de salidas. Si el efecto exige una conversión o ruta injustificada, conservarlo como ingeniería funcional y corregir esa dependencia en contrato nuevo. No convertir una auditoría completa de radios en puerta obligatoria para cualquier ensayo funcional. |
| C. Diferencia entre preparación y observable | Antes de etapa 4, medir con las poses reales si la geometría prospectiva produce muestras antenales online distintas del replay tras la perturbación. Si no las produce de forma resoluble, el experimento no identifica retroalimentación aunque la anatomía sea íntegra. Si las produce, ejecutar el contraste funcional acotado; una calibración con sujetos vivos emparejados sería una reivindicación adicional, no un requisito implícito de ese piloto. |

**Recomendación:** conservar los datos anatómicos actuales, continuar sólo con el alcance funcional explícito y concentrar el próximo discriminador en la interfaz sensorial/corporal. La integridad comprobada reduce la plausibilidad de una descarga defectuosa; no valida la electrofisiología transferida ni resuelve por sí sola etapas 3/4.

## Recibos y hashes

`PROVENANCE_CHECKS.json` contiene los 57 hashes y la reconstrucción por ruta; SHA-256: `78fc5203e1d6891b2a2b0e02f1ae3976f6a7ee16eb4183aaf526839830e95de1`.

`PROVENANCE_REMOTE_HEADERS.json` conserva las siete comparaciones oficiales completas; SHA-256: `eb14376706344a70e32d6f81d7750bb89f6ffaab8b3764b44b87190f720c9228`. **Corrección del verificador propio:** el primer recibo convertía cabeceras repetidas en un diccionario y perdía el MD5, dejando indicadores `false`. No eran discrepancias de contenido. Este segundo recibo conserva todas las cabeceras y verifica siete MD5 coincidentes; se preservó el primero y no se alteraron datos fuente.

`PROVENANCE_MORPHOMETRY.json` conserva el contraste de volumen y sus límites. El SHA-256 de este informe y de todos estos recibos figura en `PROVENANCE_REVIEW.sha256`.
