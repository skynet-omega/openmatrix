# Runtime CUDA residente: código y recibos de un milisegundo

El grafo CNS real conserva exactamente el estado final completo en 1 ms pero no acelera el organismo (3,808 s referencia; 3,903 s candidato). El fixture sintético de rechazo inicialmente falló por un oráculo cuyo final difería un ULP; la traza y el FAIL original se conservan. Con el oráculo corregido prospectivamente, el mismo binario CUDA coincidió en siete aceptaciones, seis rechazos y un evento. Ninguna ruta matemática del 24-09 fue descartada; ver `files/DONORS_20260924.md`.

Los checkpoints completos (~400 MB por brazo) permanecen locales; sus SHA-256 figuran en `STATE_HASHES.json`. El verificador y su recibo están publicados para inspección, pero esos hashes no sustituyen una auditoría externa de los datos brutos. No se publican credenciales.

Un segundo prototipo incorpora `files/epoch_session.cpp`: coordinador C++ de puertos, eventos y rollback con MuJoCo C. Su único fixture sintético pasó; no ejecutó el conectoma ni midió velocidad. `files/GEMINI_REVIEW_02.md` explica por qué FP32 y `event_sparse` no se multiplican, y `files/OWNER_MIGRATION_MAP_01.md` detalla lo pendiente para el organismo real.

Una revisión de procedencia de la preparación real identifica la homologación PN/DNb05, lector motor de ganancia 250 extraído de un script histórico de Gemini y 40 pasos físicos de 25 µs por milisegundo (`files/PREPARATION_PROVENANCE_01.json`). El verificador de paridad compara esta preparación intervenida. ChatGPT identificó de forma documental el ULP del oráculo, sin ejecutar el código; ver `files/CHATGPT_REVIEW_01.md`.
