# Reproducir la evidencia registrada de la campaña 50

El ZIP compacto contiene las ocho trazas completas, los registros DNg100, el diseño, el análisis y los controles. Requiere Python3 y NumPy. No requiere CUDA, acceso a los árboles originales ni descarga de datos.

Desde una extracción nueva del ZIP compacto:

```bash
cd ETAPA45_TEMPORAL_20260927_50
python3 verify_complete50.py
python3 -O verify_complete50.py
python3 -O test_analysis50.py --real
python3 -O test_law_supplement50.py
python3 -O verify_supplements50.py
```

El primer comando recalcula los contrastes de las ocho condiciones, el veredicto y el informe, además de verificar identidad, entradas, relojes, latencia, prefijo, parámetros y ley DNg100. Las pruebas separadas rechazan modificaciones de entrada, identidad, contexto, reloj, parámetros, criterio y transferencia. Ninguna comprobación esencial depende de `assert`.

El archivo completo `ETAPA45_TEMPORAL_20260927_50_COMPLETO.zip` conserva también el estado inicial48, los ocho estados finales íntegros, los operandos49 usados en el certificado y las dependencias locales explícitas. Comparte contenidos idénticos mediante recetas sin pérdida; las partes de cada NPZ se concatenan para recuperar exactamente sus bytes originales y su SHA256.

Para materializar el archivo completo en una ubicación nueva, con espacio suficiente:

```bash
python3 unpack_full50.py ETAPA45_TEMPORAL_20260927_50_COMPLETO.zip extraccion50
cd extraccion50/AXIOMA_ASTRA/campanas/etapa45_temporal_factorial_20260927_50
python3 -O verify_complete50.py --without-manifest
python3 -O verify_certificate49.py
python3 -O verify_supplements50.py
```

La extracción comprueba hashes de cada archivo. El constructor verifica además todos los contenidos únicos y la reconstrucción byte por byte de todos los NPZ desde el ZIP terminado. El modo `--prefix` del extractor permite extraer una parte identificada del archivo; una extracción parcial no se presenta como materialización completa.

**Alcance:** estos comandos reproducen el análisis de registros científicos, no una nueva vida del cerebro. Las ocho simulaciones se ejecutaron en el entorno original identificado en los recibos. La recuperación exacta48→continuación fue cualificada en49; el nuevo propietario temporal de un checkpoint final50 todavía no tiene un cargador portable cualificado. No afirmar una reejecución completa del CNS en otra máquina ni volver a integrar ocho brazos como una prueba de empaquetado.

El certificado sobre entradas negativas DNg100 usa capturas49 anteriores. Está referenciado en el compacto, pero su recomputación sin dependencias locales requiere el archivo completo. La cobertura PN del compacto se reconstruye desde una proyección sin pérdida; la anatomía completa y su procedencia se conservan en el archivo completo.

El archivo completo conserva las rutas relativas de los dos proyectos. Algunos cargadores históricos siguen conteniendo rutas absolutas; antes de una nueva simulación fuera del entorno original se debe cualificar su reubicación y fijar un presupuesto nuevo. Es una limitación de portabilidad, no evidencia de fallo neuronal ni autorización para cambiar el contrato científico.
