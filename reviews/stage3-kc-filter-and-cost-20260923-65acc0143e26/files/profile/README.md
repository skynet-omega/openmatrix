# Coste de una época completa del organismo

Perfilado CPU confirmado localmente: 40.255 de59.887s (67.2%) transcurren en la llamada nativa C++ del grafo recurrente. Esto ubica el coste; no distingue ejecuciónGPU y esperas. La suma de tiempos acumulados anidados sería incorrecta. El estado científico de las corridas de20ms con perfilador activado/desactivado fue idéntico.

Se consumieron cuatro cargas,44ms de ensayo total,268.687s. Nsight2022 no tenía importador;2024 registróAPI pero no actividad de kernels y advirtió incompatibilidad del driver13.1 frente a sus bibliotecas12.8. No hay un benchmarkGPU interpretable ni optimización promovida.

El CNS aceptó3677 pasos y rechazó0 en20ms. Los1564 eventos registrados darían1884 intervalos como cota combinatoria sin más cortes; no es una cota suficiente de precisión ni autoriza saltar eventos.

A programa residente/compilación y reducción de nodos sigue candidato sólo tras atribuciónGPU; B integración espacial con eventos requiere certificado de error; C implícito no recibe apoyo de rechazos en este prefijo. No invertir en optimizar validaciones o física por el supuesto de que explican el80%. El trabajo pesado observado ya está enC++.

[Plan previo](PLAN.json), [perfil](profile_on_01/PROFILE.json), [neutralidad](CAPTURE_NEUTRALITY.json), [limitacionesNsight](PROFILER_LIMIT.md).
