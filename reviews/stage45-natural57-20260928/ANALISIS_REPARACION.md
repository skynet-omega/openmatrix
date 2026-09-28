# Corrección del contador analítico, sin modificar la simulación

El primer chequeo CPU del verificador57 sobre la cualificación2ms rechazó mi supuesto `llamadasCSR = 4 × ensayos registrados`. Fallo conservado en VERIFY_QUALIFICATION.json (0.0272221sCPU), fuente previa en analisis_previo. Ningún brazo CNS falló por esta comprobación, que se ejecutó separadamente.

La fuente ejecutada graph_runtime.py, constructor GraphRK23, líneas88–89, realiza dos ensayos de calentamiento de cuatro RHS antes de capturar el grafo. El observadorDNg reinicia su registro en cada advance; los testigosPN/ORN reinician antes de la construcción inicial y cuentan también esos8 llamados. En la cualificación se observan448/424 llamados frente110/106 ensayos, exactamente4×ensayos +8 sólo en el primer intervalo. El verificador ahora exige esta igualdad exacta, no una tolerancia mayor. Rechazará cualquier otro exceso o falta. Las ventanas51–89 y257–384 y la suma muestreada desde11ms no incluyen el calentamiento inicial.

No se cambia fuente neuronal, evaluador vivo, contrato, tolerancia, entrada o criterio científico. No se relanza CNS. First/last/min/max resumen llamadas predictoras, comprometidas y rechazadas; no son una integral física completa. Los resúmenes temporales de señales se etiquetan como muestras por milisegundo, no como dosis sináptica medida.
# Segunda reparación del análisis: geometría latente del prefijo

Al terminar las tres vidas, el primer verificador completo detectó `common10ms prefix spatial_geometric_concentration`. Ese campo describe la concentración geométrica de la fuente incluso durante el prefijo en que el estímulo está apagado. Las fuentes L/R tienen geometrías distintas por diseño. La concentración **usada**, ORN, CNS, cuerpo y todos los demás campos del prefijo coinciden exactamente; la geometría de cada brazo se reconstruye y verifica por separado.

Se conservó `analisis_previo/verify57_before_geometric_prefix.py` y el coste/fallo `verify_COMMAND_COST.json`. La reparación excluye únicamente la igualdad entre brazos de esa geometría latente; conserva su igualdad exacta con la fórmula espacial, incluyendo la prueba de corrupción de geometría. No cambia observaciones, métricas, ventanas, criterios ni ejecución CNS. Fue posterior a la exposición, y así queda declarada.
