# Encargo matemático y de programación: causa de la divergencia KC

Objetivo: una corrección o certificado de error útil para la orientación del organismo, manteniendo las tres alternativas y sin una sucesión de microfixtures inconexos. Se suministran WARP, controlador C++, tolerancias, snapshots/restore y productores. Las fuentes físicas son las de la campaña 13, antes de nueva instrumentación.

Datos reales anteriores y código reproducido: commit f502238c58025870983fbad6bc8cccbb82ebc669 de OpenMatrix; resultado numérico 20 ms en a1ab76706e94c11f6205990d2cf2e279e71ae806. Primera diferencia de evento registrada KC76431 (98437/100000 ns desde inicio); máximo trough KC544736. El reloj 1.482375 s era absoluto; no 1.482 ms post-ON.

Se pide: 1) inspeccionar causas A muestreo/estado detector, B integración, C rollback; 2) código C++/CUDA o comparador portable para un falsador que use las entradas reales que Codex va a extraer; 3) decidir qué campos del detector son estados físicos observables y cuáles dependen inevitablemente de la discretización, y qué condición causal/numerica permitiría un contrato prospectivo motor sin ignorar error relevante. No cambiar el gate histórico ni asumir que KC está aislada. Evitar que la ayuda compute respuestas dentro del organismo.

Codex instrumentará dos épocas reales de 125 microsegundos con on/off y rollback, y puede ejecutar tu código. Propón un algoritmo concreto si detectas defecto, con criterio previo y casos límite; máximo dos prototipos. Fuente WARP y C++ ahora incluida; no hace falta inventar datos. Si no puedes descargar NPZ, lee fuentes/JSON, declara esa limitación y entrega el programa para ejecución local.
