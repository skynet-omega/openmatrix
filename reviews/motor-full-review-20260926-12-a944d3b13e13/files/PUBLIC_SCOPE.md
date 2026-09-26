# Alcance de esta publicación

Revisión12: dos simulaciones continuas de2s completadas. El contrato funcional conserva FAIL por un mando yaw diferente en1/2000 pasos. No es un problema de descarga ni una simulación abortada. El runtime pasa; ahorro47,72%, coste27,19min por segundo simulado.

Este subconjunto público contiene código de la revisión, criterios, resultados, trazas y dictámenes. Omite bibliotecas compiladas, checkpoints científicos grandes y parte de las dependencias históricas. Sirve para revisar la evidencia; no permite anunciar una nueva ejecución completa del organismo ni del verificador integral por sí solo.

La cápsula local completa de evidencia guardada (MOTOR_REVISION_INTEGRAL_20260926_12.zip) tiene1.447.142.648bytes y SHA25688dbe2cf0efa34cc72ac64b0b500890780c2a7c9bfba24f4268f3736289f083b. Una extracción nueva reconstruye PAIR100=PASS y PAIR2000=FAIL, además de ambos contratos runtime. La comprobación bloquea acceso a los árboles originales y no ejecuta GPU. No certifica reinicio genérico.

Entradas de lectura: RESULTADOS.md, METRICAS.json, REVISION_INTEGRAL.md y reviews/math/POSTMORTEM_LONG.md. Las fuentes y criterios congelados permanecen intactos.
