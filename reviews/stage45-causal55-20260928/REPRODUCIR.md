# Reproducción y alcance

El ZIP compacto contiene los registros usados por el verificador, contratos congelados, controles, intervenciones, fuentes nuevas y revisión. No requiere importar el organismo de trabajo para recalcular las conclusiones. Requiere Python3.10 y NumPy; las versiones utilizadas están en ENVIRONMENT.json.

Desde una extracción nueva, dentro de la carpeta `ETAPA45_TRANSFERENCIA_CAUSAL_20260928_55`:

```bash
# Corto: manifiesto, integridad científica y cifras desde arrays.
python -B -O check_delivery55.py
# Completo de evidencia: añade alteraciones deliberadas de contexto, entrada,
# soportePN, mando, criterio y bandera; cada una debe ser rechazada.
python -B -O check_delivery55.py --corruptions
```

Si la entrega pública está segmentada, concatenar las partes por su orden en ARCHIVE_PARTS.json y verificar el SHA256 del ZIP antes de extraer. El manifiesto enumera cada fichero. Las partes no son ZIP independientes.

El archivo local `ETAPA45_TRANSFERENCIA_CAUSAL_20260928_55_ESTADOS_Y_FUENTES.zip` conserva los dos estados iniciales48, los seis finales55, fuentes y dependencias locales transitivas, y manifiesto de todos los miembros. `SOURCE_INDEX.json` relaciona rutas originales y copias por hash. Sus miembros se verifican todos por streaming.

Para repetir **toda la simulación** en el mismo entorno CUDA cualificado, usando una carpeta nueva y las dependencias originales verificadas:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python -B repeat_frozen55.py \
  --project-root /home/daroch/AXIOMA_ASTRA
```

Ese modo crea exposición nueva con el mismo contrato finito:703ms previstos, incluido el intento original que debe reproducir la discontinuidad detectada, y su reparación; techo800ms/4000sCPU/3600s de colas. No sobrescribe55 ni selecciona semillas nuevas. El comando está suministrado, **no fue ejecutado de nuevo como parte de55**. Se detiene si falta o cambió una dependencia; no instala nada ni modifica el motor histórico. El tiempo exterior incluye la creación y verificaciones, además del tiempo de colas.

La reanudación GPU totalmente portable desde rutas distintas no está cualificada. No confundir la verificación CPU ejecutada desde una extracción limpia con esa reproducción completa. La cápsula preserva los datos necesarios para trabajar esa portabilidad; no se atribuye un resultado no medido.
