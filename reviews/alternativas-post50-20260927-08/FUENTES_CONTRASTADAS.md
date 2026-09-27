# Fuentes y límites que cambian esta decisión

## Fuentes consultadas directamente

- [Yang et al., Cell 2024](https://doi.org/10.1016/j.cell.2024.08.033): cinco parejas de steering, con lateralidad distinta para DNb06. [Zenodo 12775493](https://zenodo.org/records/12775493) contiene código; no se verificaron datos neuronales crudos públicos. Los datos de calcio no calibran q. Usamos las identidades para una inspección exploratoria fijada antes de consultar sus valores en cuatro checkpoints, no para seleccionar el mayor efecto.
- [Braun, Hurtak, Wang-Chen y Ramdya, Nature 2024](https://www.nature.com/articles/s41586-024-07523-9): redes descendentes reclutadas por algunas neuronas de comando. Corrige la atribución a Westeinde de la respuesta ASTRA. No prueba que una mediana de cinco DN sea un controlador ni que todas las DN requieran el mismo contexto.
- [Fujiwara, Brotas y Chiappe, Neuron 2022](https://doi.org/10.1016/j.neuron.2022.04.008), [depósito 6365304](https://zenodo.org/records/6365304): se recuperaron metadatos, Readme y dos scripts, con MD5 coincidente. El README declara fases y posiciones 2D de articulaciones femur-tibia y tibia-tarso de tres patas izquierdas, además de Vm y velocidades corporales. Hay cinemática parcial pertinente. No equivale a ángulos 3D de seis patas listos para nuestro transductor; el mapeo sigue sin resolver. No descargamos Data1/Data2, que suman aproximadamente 1,11 GB. Recibos en `fuentes/FUJIWARA_*`.
- [Apostolopoulou y Lin, PNAS 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7368247/): compensación a exceso de inhibición en mushroom body adulto, con mecanismos y subtipos distintos, en días. No identifica un setpoint universal ni regulación en segundos de DNb05/DNg100. Esto reduce la prioridad biológica de C2, aunque una regla local explícita pueda estudiarse como ingeniería.
- [Pospisil et al., código conn2eff](https://github.com/dp4846/conn2eff): simulaciones y estimadores de efecto causal; el README incluye ejemplos de conductancias. No es un effectome medido de nuestras DN ni una calibración de pesos. Usar como método externo, no como componente cognitivo.
- [Ceballos et al., iScience 2026](https://doi.org/10.1016/j.isci.2026.115624), [datos](https://data.mendeley.com/datasets/dh4vghvtpm/1), [código anunciado](https://github.com/pena-rodrigo/ceballos_et_al2026_iscience): contraste de conductancias y modulación axonal en el circuito de escape GF. El ZIP enlazado, 30 kB, se descargó e inspeccionó: sus 80 archivos son metadatos macOS/AppleDouble; no contiene las fuentes ejecutables aparentes. Se conserva ZIP, hash e inspección. No se adoptó ni ejecutó ese modelo; su fisiología no calibra steering.

## Antecedentes locales comprobados

La entrada de biblioteca fue `/home/daroch/AXIOMA_FLYWIRE/matrix/catalogo/README.md` y `recursos_cientificos/README.md`, ambos en sólo lectura. Se consultaron el índice complementario ASTRA, el plan posterior a48 y su revisión motora.

`anatomical_morphometry.py` ya aplica escala por volumen al preparado. `hybrid_visual_brain.py` ya distingue voltaje/conductancia visual y tasas genéricas para otras células. C1 cambiaría la ley de las células ordinarias; no descubre que jamás hubo conductancias en el organismo. Sus parámetros actuales también son hipótesis no calibradas.

`work/gemini_physiological_predictor_review_20260915/README.md` ya rechaza inferir una tasa universal de 30–50 Hz o conductancias individuales desde expresión. Por eso C2 requiere definir qué regula y comparar su dependencia de estado; no se incorpora una homeostasis genérica por analogía.

La captura49 demuestra que, con sus operandos congelados, net+drive de DNg100 es negativo antes de theta. Esa identidad elimina el rescate por modificar únicamente una ganancia positiva o tau; no identifica una causa biológica, ni demuestra que otro estado recurrente deba fracasar.

Las propuestas externas son hipótesis. Los artículos, códigos, datos, parámetros y dominios se distinguen; ningún asesor ejecutó aquí la simulación ni se verificó PRO.
