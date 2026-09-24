# Cápsula compacta de revisión, no simulación neuronal

Incluye código original de ChatGPT, cuatro JSON originales con geometría/cierres, verificación local, dictámenes y cálculos. No incluye los NPZ completos ni checkpoints: no reproduce el organismo ni la recomputación de trayectoria `analyze_existing.py`. Las trazas originales están en los paquetes públicos de campañas26/27 enlazados en CHATGPT_REQUEST.md. No se afirma ejecución de MuJoCo/CUDA.

Desde una extracción nueva de este ZIP, Python3 sin paquetes adicionales reproduce los dos cálculos externos:

```bash
python3 -B criba_feedback_chatgpt_original.py --files files --horizon 2 --pulse 1.5 --angle 30 > recomputed_gemini.json
python3 -B criba_feedback_chatgpt_original.py --files files --horizon 2 --pulse 0.5 --angle 1 > recomputed_small.json
```

Los resultados geométricos y de coste deben coincidir con los campos `gemini_case` y `chatgpt_case` de EXTERNAL_CODE_RESULT.json, excepto las claves de rutas dentro de `inputs_sha256`, que cambian con el directorio. Los valores de los cuatro hashes se conservan. No evaluar los enlaces `sandbox:` de la respuesta como archivos incluidos; sólo el bloque completo de código recibido fue recuperado y se verificó localmente.

Verificación de empaquetado: una ejecución CPU adicional desde extracción nueva, dos llamadas a ese archivo extraído, comparación de todos los resultados sin las claves de rutas; máximo30s, cero simulaciones. Este presupuesto de reproducción de la cápsula es distinto de los análisis de datos y no permite reintentar organismos ni modificar resultados científicos. Manifiesto del publicador con hashes explícitos; sin credenciales ni cambios en históricos.
