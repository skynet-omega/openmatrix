# Fuentes y correcciones antes del resultado agrupado

Consulta primaria del 28-09-2026; primero se revisó el catálogo local. No se descargaron nuevos datasets ni se transfirieron parámetros fisiológicos.

- C1: [Braun et al., Nature 2024, Descending networks transform command signals into population motor control](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186778/), DOI 10.1038/s41586-024-07523-9. Intervenciones sobre DNs y conectividad apoyan reclutamiento entre descendentes para comportamientos completos. No identifica aquí una ley ni un contexto específico de DNg100.
- C2: [Chen et al., Nature Neuroscience 2023, Ascending neurons convey behavioral state to integrative sensory and action selection brain regions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076225/), DOI 10.1038/s41593-023-01281-z. Observa señales ascendentes de movimiento propio y acciones hacia regiones cerebrales. Una entrada pequeña en reposo no refuta su función durante locomoción; tampoco demuestra una ruta directa hacia DNg100.
- C3: [Aso et al., eLife 2014, Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC4273436/), DOI 10.7554/eLife.04580. Apoya selección de acción y valencia por conjuntos MBON; no establece iniciación de avance por MBON→DNg100.

Las identidades y grupos se seleccionan sólo con las anotaciones del MaleCNS canónico. No se transfieren IDs individuales entre estudios ni se convierte conectividad o valencia en corriente fisiológica.

## Revisión recibida, anterior al cálculo

ASTRA_V2 y Motor_V2 revisaron el planteamiento, sin examinar nuevos arrays ni modo PRO verificado. Sus respuestas completas están conservadas. Adoptamos la advertencia común: extremos independientes de transmisión constituyen una cota algebraica, no un estado alcanzable. Añadimos la vista secundaria prospectiva C1_external, que excluye ambos DNg100 (10045,10056), conservando C1 original. No selecciona por efecto.

Corregimos un criterio sugerido por ASTRA: U/|margen| no basta para excluir ni admitir capacidad cuando el grupo ya aporta señal, en especial si ésta es negativa. El criterio es el margen total reconstruido por el consumidor con el grupo en su extremo y el resto fijo. Se informa también el incremento disponible respecto al aporte actual. No habrá umbral retrospectivo de efecto material para diferencias descriptivas entre historias.

Motor C++/CUDA revisa el dominio real de transmisión, máscaras, fases y conservación FP32. No se suman reducciones FP32 separadas de grupo/resto como si fueran la reducción original. Las sumas FP64 de productos son un libro contable descriptivo, con residuo de redondeo explícito.
