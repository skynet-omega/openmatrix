# Campaña47: observar la transferencia DNg100 durante el estímulo

El [estado de la cola](QUEUE.json) distingue adquisición en curso de pareja terminada.
El instrumento pasó sus comprobaciones de software y el primer intervalo neuronal; la
conclusión requiere la pareja completa. Se conservan45/46 y sus negativos; etapas4/5 abiertas.

La [revisión externa del código](CHATGPT_CODIGO.md) no identificó un fallo material de adquisición. Sí señaló que el clasificador original confundía ausencia de objetivos positivos con cero literal. `analyze.py` permanece congelado para conservar la identidad de la ejecución. Al terminar, `python diagnosis.py` genera `DICTAMEN.json` con extremos mínimos/máximos, distingue objetivos negativos y sustituye sólo esa clasificación; no cambia datos ni criterios de admisión. No es una nueva simulación.

El postproceso finito `finish.py` está encadenado a la cola existente: espera su cierre correcto,
comprueba fuentes y bloques, genera `DICTAMEN.json`, `CURVAS.npz`, `ENTRADAS.png` e
`RESULTADOS.md`, y actualiza el estado raíz. [Estado del cierre](CIERRE.json). No reintenta
ni integra otra vida si falla un brazo. [Decisión fundada y antecedentes](DECISION.md).

Después se publica el registro mediante el workflow existente deOpenMatrix y se descarga
el paquete para reproducir el análisis desde una extracción nueva. [Estado de entrega](ENTREGA.json).
Esa fase tiene topes propios600s de pared,300sCPU del proceso publicador,300MiB de archivo
y3GiB de espacio de trabajo; no amplía el presupuesto científico ni genera otra vida.
El paquete recalcula el diagnóstico a partir de registros; no reinicia el organismo completo.

**Alcance de las dos verificaciones:** la comparación47↔45 se comprueba **localmente durante
la adquisición**, bloque a bloque, sobre `traces.npz`, `published.npz` y `events.json`.
El ZIP de esta entrega contiene los registros `dng100_observed.npz`, resultados y fuentes
analíticas, pero no aquellos tres conjuntos de ambos ensayos. Su extracción limpia puede
reproducir el **diagnóstico CPU del observador**, no verificar independientemente la igualdad
completa47↔45. `parent_observations_exact` comunica el resultado del runner local; los hashes
y la descarga verificada acreditan identidad de los archivos publicados, no sustituyen los
datos omitidos para repetir esa comparación. Tampoco se compara todo estado oculto de45,
que no se guardó. No se declara una reproducción externa del organismo o de su paridad.

La revisión del catálogo recuperó inhibición de DNg100 ya descrita el7-09, estimulación directa efectiva DN→MN sin marcha el13-09, y el análisis CPG del14-09. Son otras preparaciones e intervenciones; no registran el objetivo realmente consumido durante el olor de45.45 guarda publicaciones cada100ms y observaciones DN/cuerpo cada1ms, pero no estados completos intermedios reiniciables ni operandos de cada evaluación. No se repite el pulso directo ni una calibración genérica del cuerpo.

Hipótesis rivales: A, la entrada/ley mantiene el objetivo DNg100 nulo; B, una discrepancia de implementación o lectura pierde una respuesta válida; C, una orden válida no produce movimiento corporal.45 ya muestra avance crudo cero, por lo que C no explica por sí sola ese cero. La nueva observación separa A/B y sólo habilitaría investigar C si aparece una orden válida.

Una adquisición pareada sham/olor de3s por condición, conservando el primer segundo basal y los2s de estímulo45. No modificar pesos, parámetros, plasticidad, tolerancias, lector ni físico. Observar sólo IDs10045/10056, filas36/46, dentro de evaluaciones existentes. Registrar suma CSR consumida, sumas positiva/negativa auxiliares, drive, theta, gain, objetivo/rate de base y finales del callback, estado y derivada, relojes/etapas, aceptación RK y predictor/comprometido del acoplamiento. Las sumas auxiliares no realimentan la integración.

El scheduler C++ permanece original: se registran error/flags antes de su decisión y se cotejan la clasificación, contadores, continuidad de tiempos y estado posterior. En caso de ambigüedad o divergencia con45, parar la interpretación; no etiquetar un predictor descartado como trayectoria comprometida. Buffer GPU acotado; descarga al terminar cada época, sin segunda evaluación neuronal para observarla.

Resultados discriminantes: objetivos positivos en pasos aceptados comprometidos refutan A; un objetivo contrario a la ley/parametrización documentada localiza un fallo reparable; una evolución coherente con objetivo nulo es una limitación del modelo/contexto, sin justificar bajar umbrales. Cambios excitadores/inhibidores pueden cancelarse: entrada neta constante no prueba ausencia de modulación. Ningún resultado de esta ronda admite navegación o recuperación física.

Presupuesto prospectivo: dos brazos,6000ms totales, máximo6300s de pared por brazo/12600s agregados,18000sCPU de procesos neuronales, RAM24GiB, VRAM14GiB, disco8GiB. Al coste45, referencia de8763s pared. Sin ampliar duración, estímulo ni ganancia tras resultados. Pruebas de software del observador separadas, máximo120sGPU/300sCPU y ninguna vida adicional; análisis de registros máximo300sCPU. Corrección de errores de implementación conserva fallos y respeta el presupuesto agregado. ChatGPT/Jev son asesores externos.
