# Reproducir la evidencia de la campaña52

El paquete compacto contiene todos los registros numéricos de diez brazos y siete cualificaciones, las proyecciones originales51 necesarias para el contraste y dos verificadores. El paquete completo conserva además estados finales, preparación48 y dependencias locales explícitas. Ninguno incluye credenciales. Se distingue recomputar los resultados guardados de volver a ejecutar el organismo en GPU.

## Verificación corta, desde una extracción nueva

Requiere Python y NumPy. Entrar en la carpeta extraída `ETAPA45_REPARACION_20260927_52` y ejecutar:

```bash
python -B -O verify_complete52.py
python -B -O aporte_motor52/verify52.py science --out "$PWD/aporte_motor52/REPRODUCCION_NUEVA.json"
```

El primer comando comprueba el manifiesto, reconstruye cualificaciones, métricas, unidades y trayectorias aceptadas, y detecta diez corrupciones deliberadas sin editar la evidencia. El segundo usa otra implementación del receptor y la copia canónica deIDs incluida. Elegir un nombre nuevo para su recibo si ya existe; no sobrescribirlo.

Para recalcular la figura, en una copia de trabajo de la extracción con Matplotlib instalado:

```bash
python -B plot52.py
```

La figura no sustituye al verificador.

## Paquete completo

El ZIP completo emplea un manifiesto de bloques compartidos para conservar los archivos originales byte por byte sin duplicar estados idénticos. `FULL_CAPSULE.json` registra el hash del archivo y la comprobación de cada bloque yNPZ reconstruido. Para materializar todo en una ubicación nueva, con espacio suficiente:

```bash
python -B unpack_full52.py ETAPA45_REPARACION_20260927_52_COMPLETO.zip reconstruccion52_nueva
```

Este comando reconstruye y verifica todas las fuentes, dependencias locales transitivas, estados y resultados incluidos. No ejecuta el CNS. Las rutas originales quedan bajo `AXIOMA_ASTRA/` y `AXIOMA_FLYWIRE/matrix/` dentro del destino. El histórico original no se modifica.

La reanudación GPU del esquema52 no está cualificada. El runner52 arrancó siempre desde48 mediante el restaurador49 comprobado, con las versiones registradas en `ENVIRONMENT.json`. Volver a ejecutar los diez brazos exige otra carpeta, fuentes/contrato congelados y presupuesto nuevo; `run_queue.py` rechaza una segunda ejecución sobre esta campaña. No se presenta aquí una reejecución portable completa del CNS como si ya hubiera sido probada.

La entrega se valida desde extracciones nuevas local y descargada; esos recibos acreditan reproducción de los datos registrados, no una nueva vida simulada ni admisión de etapas4/5.
