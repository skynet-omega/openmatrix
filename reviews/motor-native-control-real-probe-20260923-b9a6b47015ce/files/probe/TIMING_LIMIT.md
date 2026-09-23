# Resultado de la sonda: neutralidad aprobada, atribución temporal pendiente

Las tres cargas de1ms consumieron104,372s y conservaron exactamente los estados científicos seleccionados, archivos auxiliares y eventos. Entre modos0/1 coinciden191 decisiones y16 invocaciones nativas. Los tiempos de esos dos modos de C++ son2152,855 y2181,308ms, sin estimación estadística con una réplica.

La suma de intervalos del grafo medidos por eventos CUDA es2279,313ms, superior a los2181,308ms de pared de sus invocaciones contenedoras. El cociente1,044929 **no es un porcentaje admisible de tiempo consumido**. Por época varía entre1,04025 y1,04699. Se conserva el resultado bruto y se marca la inconsistencia en [TIMING_LIMIT.json](TIMING_LIMIT.json); no se normalizan los datos para ocultarla.

La [documentación de NVIDIA](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) advierte que los eventos miden intervalos asíncronos que pueden incluir otras operaciones. Eso no identifica por sí solo la causa de este exceso agregado. No se ha demostrado que sea WSL, frecuencia de reloj, ordenamiento, una limitación de la sonda o un defecto del runtime.

La medida del host registra2120,481ms dentro de sincronización y47,568ms en llamadas de lanzamiento. Esperar puede incluir trabajo GPU/planificación: no se atribuye todo a un coste eliminable. No se ha medido tiempo exclusivo de kernels. Presupuesto de tres cargas agotado; cero repeticiones y ninguna optimización promovida.

El experimento mantiene valor para verificar el envoltorio y estudiar la secuencia adaptativa real, pero no certifica un desglose preciso CPU/GPU. La confirmación neuronal siguiente usa las ecuaciones conservadas, no este cociente temporal.
