# Evidencia completa44 — verificación desde extracción limpia

El paquete conserva contratos, fuentes, entradas y fuerzas realmente consumidas, traza, checkpoint1000ms y snapshots completos de inicio/fin. Los hashes deduplican bytes idénticos. El verificador materializa cada ruta dentro de la extracción; no escribe en los repositorios originales. Incluye además fuentes/versiones históricas y recursos de construcción del organismo; no se afirma haber validado una nueva ejecuciónGPU portable.

Requiere Python3.10.18 y NumPy1.26.4. Ambos modos son CPU; no generan pasos neuronales.

```bash
unzip ETAPA45_CONTINUIDAD_CPU_20260925_44_COMPLETO.zip -d extraido44_nuevo
cd extraido44_nuevo
python -B verify_capsule.py --out HASHES_VERIFICADOS.json
python -B verify_capsule.py --full --out EVIDENCIA_RECALCULADA.json
```

Modo corto: hashes y pruebas de corrupción sobre contrato, bandera y traza. Modo completo: igualdad semántica del checkpoint inicial completo, reconstrucción de los35campos del sham durante120ms en Python normal/-O y controles de presupuesto. **No es una reproducción nueva del CNS** y no ejecuta los brazos pendientes. La campaña agotó su presupuesto CPU; no debe reabrirse por extraer este ZIP. El informe y el bloqueo están en `AXIOMA_ASTRA/campanas/etapa45_neural_contrast_20260925_44/README.md` y `STOP.json`.
