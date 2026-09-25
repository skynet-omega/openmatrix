# Alcance del paquete de resultados

Esta cápsula permite recalcular la comparación desde las trazas numéricas completas de los dos brazos y del donante, y revisar el código ejecutado. Incluye protocolo, fuentes, testigos de eventos/flujo/mando, recibos, preparación intervenida explícita y revisión externa. Las fuentes ejecutadas se conservan con sus nombres de captura y manifiestos.

No incluye los grandes arrays de los checkpoints ni todos los recursos históricos del organismo. Sus manifiestos se incluyen para identificación, no como sustitutos de los arrays. La igualdad de 588 arrays preparados se verificó localmente con esos datos completos. Este ZIP no permite repetir esa comprobación completa ni ejecutar el organismo en un equipo limpio. Los checkpoints locales están serializados; no se ha validado su reanudación.

La revisión de ChatGPT es lectura de código y propuesta matemática, sin ejecución de NPZ/CUDA/MuJoCo en su entorno. Su verificador se ejecutó aquí. Los dos verificadores offline coincidentes no sustituyen una referencia numérica refinada del organismo a 1 s.

## Recalcular desde una extracción limpia

Se requiere Python con NumPy; Matplotlib sólo para la figura. Tras concatenar las partes del ZIP y comprobar el SHA-256 de ARCHIVE.json, extraer conservando carpetas. Desde la raíz extraída:

```bash
cd campanas/etapa4_contrast_ablation_20260925_36
python3 -O chatgpt_verificar_cintas_cd.py \
  --donante ../etapa4_long_trajectory_20260925_35/native_minus_02/traces.npz \
  --control identity_01/traces.npz \
  --sinD no_contrast_01/traces.npz \
  --campos ../etapa4_long_trajectory_20260925_35/CAMPOS.json \
  --out RECOMPUTED.json
cmp EXTERNAL_VERIFIED_01.json RECOMPUTED.json
python3 -O summarize_pair.py --out analysis_reproduced
cmp analysis_01/MEASURES.json analysis_reproduced/MEASURES.json
```

Esto reconstruye el efecto y las medidas del informe desde datos crudos, conservando las limitaciones del diagnóstico. No ejecuta una nueva vida ni certifica navegación. La guía de corrida completa en el host original está en README.md; requiere los recursos locales históricos y directorios de salida nuevos. No repetir simulaciones sólo para empaquetar.
