# Runtime CUDA residente: código y recibos de un milisegundo

El grafo CNS real conserva exactamente el estado final completo en 1 ms pero no acelera el organismo (3,808 s referencia; 3,903 s candidato). El fixture sintético de rechazo inicialmente falló por un oráculo cuyo final difería un ULP; la traza y el FAIL original se conservan. Con el oráculo corregido prospectivamente, el mismo binario CUDA coincidió en siete aceptaciones, seis rechazos y un evento. Ninguna ruta matemática del 24-09 fue descartada; ver `files/DONORS_20260924.md`.

Los checkpoints completos (~400 MB por brazo) permanecen locales; sus SHA-256 figuran en `STATE_HASHES.json`. El verificador y su recibo están publicados para inspección, pero esos hashes no sustituyen una auditoría externa de los datos brutos. No se publican credenciales.
