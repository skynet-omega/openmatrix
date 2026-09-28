# Reproducción de la campaña57

## Modo corto desde una extracción nueva

El compacto conserva registros, contratos, observadores y verificador. Python3 y NumPy son suficientes para reconstruir la comparación; Matplotlib sólo se necesita para regenerar la figura. No importa el árbol de trabajo ni usaCUDA.

```bash
cd ETAPA45_SENAL_NATURAL_20260928_57
OPENBLAS_NUM_THREADS=1 PYTHONPATH='' python3 -B -O check_delivery57.py --corruptions
```

Comprueba manifiesto/hashes, cualificación exacta2ms, tres prefijos89ms contra54, entradas/salidas registradas, reloj, lector, contextoRHS, objetivos descendentes, geometría y contraste corporal. Reconstruye estadísticas y veredicto desde arrays y rechaza corrupciones deliberadas. CeroCNS nuevos. No usar `--write` sobre resultados conservados. La comprobación bajo `-O` conserva excepciones explícitas.

## Estados y fuentes completos

`ETAPA45_SENAL_NATURAL_20260928_57_ESTADOS_Y_FUENTES.zip` conserva estado inicial 48/sham, tres estados finales completos, RNG/parámetros/reloj, registros y dependencias locales transitivas identificadas. Incluye también los archivos reales del padre de carga en frío a 700 ms, con topología CSR e IDs canónicos, el catálogo MaleCNS utilizado y la configuración del cargador. Un índice SHA256 evita duplicar contenidos. `SOURCE_INDEX.json` relaciona ruta original con copia conservada de fuentes; `FULL_INDEX.json` enumera además esos datos. Nunca escribe automáticamente en rutas absolutas originales.

```bash
python3 -B -O restore_capsule57.py ../ETAPA45_SENAL_NATURAL_20260928_57_ESTADOS_Y_FUENTES.zip
python3 -B -O restore_capsule57.py ../ETAPA45_SENAL_NATURAL_20260928_57_ESTADOS_Y_FUENTES.zip --restore-to ../ESTADOS57_EXTRAIDOS
```

El primer modo lee y verifica todos los contenidos sin expandirlos; el segundo exige un destino nuevo. La expansión completa puede ocupar más que elZIP. Verificación integral de contenido no equivale a nueva ejecuciónGPU ni a reanudación desde el final3384ms.

## Modo completo en el host original

La siguiente orden crea una exposición nueva, con cualificación2ms y tres384ms, tope1200msCNS/6000sCPUtrabajadores/5000scola. Verifica fuentes y usa directorio único. Requiere las rutas originales y entornoCUDA, y no se ejecuta para comprobar el cierre.

```bash
OPENBLAS_NUM_THREADS=1 /home/daroch/miniconda3/envs/GPU/bin/python -B -O repeat_frozen57.py --project-root /home/daroch/AXIOMA_ASTRA
```

El cargadorGPU no está cualificado desde rutas portables ni para reinicio desde los estadosfinales3384ms; preservamos explícitamente esa limitación. La extensión57 es una continuación desde el mismo estado3000ms, con40pasos25µs y recurrencia corporal exacta cada ms, no una ampliación del guard de carga49. Las comprobaciones limpias de esta entrega reproducen el análisis de registros, no una vida nueva, datos biológicos ni admisión4/5.
