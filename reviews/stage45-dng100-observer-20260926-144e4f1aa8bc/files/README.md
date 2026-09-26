# Campaña47: observar la transferencia DNg100 durante el estímulo

Estado: implementación, sin vida neuronal nueva todavía. Se conservan45/46 y sus negativos; etapas4/5 abiertas.

La revisión del catálogo recuperó inhibición de DNg100 ya descrita el7-09, estimulación directa efectiva DN→MN sin marcha el13-09, y el análisis CPG del14-09. Son otras preparaciones e intervenciones; no registran el objetivo realmente consumido durante el olor de45.45 guarda publicaciones cada100ms y observaciones DN/cuerpo cada1ms, pero no estados completos intermedios reiniciables ni operandos de cada evaluación. No se repite el pulso directo ni una calibración genérica del cuerpo.

Hipótesis rivales: A, la entrada/ley mantiene el objetivo DNg100 nulo; B, una discrepancia de implementación o lectura pierde una respuesta válida; C, una orden válida no produce movimiento corporal.45 ya muestra avance crudo cero, por lo que C no explica por sí sola ese cero. La nueva observación separa A/B y sólo habilitaría investigar C si aparece una orden válida.

Una adquisición pareada sham/olor de3s por condición, conservando el primer segundo basal y los2s de estímulo45. No modificar pesos, parámetros, plasticidad, tolerancias, lector ni físico. Observar sólo IDs10045/10056, filas36/46, dentro de evaluaciones existentes. Registrar suma CSR consumida, sumas positiva/negativa auxiliares, drive, theta, gain, objetivo/rate de base y finales del callback, estado y derivada, relojes/etapas, aceptación RK y predictor/comprometido del acoplamiento. Las sumas auxiliares no realimentan la integración.

El scheduler C++ permanece original: se registran error/flags antes de su decisión y se cotejan la clasificación, contadores, continuidad de tiempos y estado posterior. En caso de ambigüedad o divergencia con45, parar la interpretación; no etiquetar un predictor descartado como trayectoria comprometida. Buffer GPU acotado; descarga al terminar cada época, sin segunda evaluación neuronal para observarla.

Resultados discriminantes: objetivos positivos en pasos aceptados comprometidos refutan A; un objetivo contrario a la ley/parametrización documentada localiza un fallo reparable; una evolución coherente con objetivo nulo es una limitación del modelo/contexto, sin justificar bajar umbrales. Cambios excitadores/inhibidores pueden cancelarse: entrada neta constante no prueba ausencia de modulación. Ningún resultado de esta ronda admite navegación o recuperación física.

Presupuesto prospectivo: dos brazos,6000ms totales, máximo6300s de pared por brazo/12600s agregados,18000sCPU de procesos neuronales, RAM24GiB, VRAM14GiB, disco8GiB. Al coste45, referencia de8763s pared. Sin ampliar duración, estímulo ni ganancia tras resultados. Pruebas de software del observador separadas, máximo120sGPU/300sCPU y ninguna vida adicional; análisis de registros máximo300sCPU. Corrección de errores de implementación conserva fallos y respeta el presupuesto agregado. ChatGPT/Jev son asesores externos.
