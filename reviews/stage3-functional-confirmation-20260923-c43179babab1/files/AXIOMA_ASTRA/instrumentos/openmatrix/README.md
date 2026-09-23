# Publicación de evidencia en OpenMatrix

El usuario autorizó fuentes y resultados pertinentes del motor y etapa3 en `skynet-omega/openmatrix`. Se usa SSH ya configurado, sin claves en archivos de entrega. No depende de Computer Use ni de reiniciar WSL.

Un cierre de campaña reconstruye primero el informe, enumera archivos explícitos y publica un snapshot inmutable con ZIP dividido en40MiB, manifiesto de hashes y fuentes de texto legibles. El mismo contenido produce el mismo snapshot y no otro commit. No hay calendario, proceso residente ni envío de mensajes implícito.

Cierre de la campaña actual, desde `/home/daroch/AXIOMA_ASTRA`:

```bash
OPENBLAS_NUM_THREADS=1 /home/daroch/miniconda3/envs/GPU/bin/python -I -B motor_nuevo/abc_20260922/finalize.py --publish --receipt motor_nuevo/abc_20260922/PUBLICATION.json
```

`publish.py` también acepta otros manifiestos explícitos dentro de las raíces autorizadas. Rechaza rutas fuera de alcance, recorridos `..`, enlaces simbólicos, archivos ocultos, extensiones no admitidas, patrones de credenciales, colisiones de destinos y cambios de snapshots existentes. No fuerza Git ni cambia configuración global. Valida el destino exacto, árbol limpio, commit remoto y lectura pública del manifiesto. `verify_remote.py` acepta cualquier recibo explícito de este repositorio, descarga las partes por flujo y contrasta todos los miembros con el inventario local. Comprueba duplicados, tamaños, hashes y rutas antes de extraer; no ejecuta el código descargado. La reproducción científica se registra por separado.

Seis rechazos deliberados y determinismo/idempotencia comprobados conPython-O. Dos fallos iniciales conservados en el trabajo: falta de identidad del commit (resuelta con identidad por comando ya usada en el repo) y timestamp variable del manifiesto dentro del ZIP (formato2 fija todos los timestamps). Nunca presentar publicación como revisión ni reproducción externa.

Después de publicar, Codex usa la herramienta de mensajes de la app para enviar a la conversación ChatGPT autorizada el enlace fijado al commit y un encargo concreto. No se modifica el selectorPRO mediante el texto ni se inventa acceso al modelo. Jev clasifica tareas mediante su cliente con caché; no ejecuta decisiones del organismo.

Verificación genérica después de publicar (la extracción y el ZIP opcionales deben tener destinos nuevos):

```bash
python3 -B instrumentos/openmatrix/verify_remote.py CAMPAÑA/PUBLICATION.json \
  --receipt CAMPAÑA/REMOTE_VERIFY.json \
  --extract intercambio/VERIFICACION_NUEVA \
  --archive-out /mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/PAQUETE_NUEVO.zip
```

Cinco corrupciones deliberadas del paquete se rechazan también con `python3 -O -B instrumentos/openmatrix/test_verify_remote.py`. La lectura remota deja un recibo nuevo; no cambia los archivos de la campaña ni su veredicto.
