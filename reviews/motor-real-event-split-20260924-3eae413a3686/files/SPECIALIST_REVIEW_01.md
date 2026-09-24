# Revisión breve de precisión y oráculo

Especialista Codex `precision_review`, sólo lectura, sin ejecutar organismo ni GPU. Revisión no autoral de código y recibos, no certificación externa.

1. Señaló que los temporales del grafo del oráculo podían reutilizarse desde el pool general de CuPy, mientras NativeGraph mantiene uno privado. Codex reprodujo el fallo con canarios y lo corrigió en `effective_oracle.py`; `ORACLE_MEMORY_TEST_01.json` conserva tanto el fallo como el pase. Las salidas del oráculo antiguo podían coincidir aun mientras corrompía memoria ajena. La captura de1ms ya ejecutada sí pasó su comparación completa; no se presume que estuviera corrupta.
2. Confirmó que captura/inspección ocurren antes de restaurar APL y conservan PN del midpoint, SET/ADD y fase. El oráculo comparte buffers vivos: requiere fase/versión/intervalo controlados. Los60 target/rate guardados no pueden emplearse como forcing de una trayectoria nueva.
3. Exige definir el defecto entre muestras sobre estados libres; los puertos q/s tienen rate=0 en el integrador porque se proyectan, no porque físicamente permanezcan inmóviles. Para un candidato debe completarse esa norma antes de ejecutarlo; no afecta la neutralidad de una captura.

Validó la cota condicional60/8=7,5× para esquemas con CSR global por corte y la necesidad de separar eventos, recurrencia y entradas mantenidas. No infirió rigidez del Jacobiano a partir de max(rate)*h.

Sobre Gemini: error numérico puede ser estable y sesgado; bitwise valida únicamente las ejecuciones comparadas; una simulación cerebral es abierta/disipativa, no tiene por qué conservar una energía global; redondeo no sustituye a ruido biológico parametrizado. La diferencia de yaw3,42e−7° es una observación, no la tolerancia de0,002° de las campañas26/27.
