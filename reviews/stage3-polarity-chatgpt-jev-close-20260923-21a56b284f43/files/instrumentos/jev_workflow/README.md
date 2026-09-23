# Jev en el flujo del laboratorio

Integración solicitada el22-09-2026. Jev sugiere a quién derivar una tarea breve; Codex razona, programa y ejecuta; ChatGPT hace revisión externa en la conversación existente. Los verificadores locales calculan errores y tiempos y conservan el control de las decisiones científicas.

Jev devuelve Choice/Score/Noul; no genera código ni explicaciones extensas. [Alcance oficial](https://docs.typesafe.ai/introduction/coding-agents). El fabricante reconoce limitaciones para aritmética, indirection y entradas adversariales: por eso aquí no determina tolerancias, PASS ni comandos. [Limitaciones](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## Operación

Sólo biblioteca estándar de Python. Una invocación hace como máximo una petición, sin reintentos; hasta8 resúmenes,20KB de solicitud y30s de espera. Versión fijada: `jev-1.13.0`. La respuesta queda siempre como recomendación, incluso si declara confianza1. No hay un umbral calibrado para este laboratorio. La clave se lee desde `TYPESAFE_API_KEY` o por entrada oculta; nunca se guarda en el paquete ni en los resultados.

Las consultas idénticas reutilizan una respuesta local validada: la identidad incluye modelo, preguntas, criterios y tareas completos. Una respuesta reutilizada registra cero llamadas y cero tokens nuevos, y conserva aparte el consumo original. Una consulta en curso o interrumpida deja un marcador `.pending` con la ruta de su recibo; bloquea otro envío idéntico antes de pedir la clave. Revisar ese recibo ante un fallo de red, porque el servidor podría haber procesado la petición. No se borra el marcador ni se reintenta automáticamente. Una caché dañada produce error, no otra llamada pagada.

```bash
python -I -B test_jev_workflow.py
python -I -B jev_workflow.py --input ejemplo_tareas.json --out salida_nueva
# Una solicitud real, con clave ya configurada o introducida sin eco:
python -I -B jev_workflow.py --input ejemplo_tareas.json --out consulta_nueva --live --ask-key
# Reutilizar el piloto conservado, sin clave ni conexión:
python -I -B jev_workflow.py --input ejemplo_tareas.json --out reutilizacion_nueva \
  --cache-dir cache --reuse-result piloto_20260922_01
```

La salida contiene solicitud, respuesta tipada, uso, tiempo, hash y recomendación. No envía otros archivos ni ejecuta tareas. No colocar conversaciones completas, claves o datos innecesarios en los resúmenes. La protección de formato detecta algunos patrones de secretos; no sustituye revisar qué datos se seleccionan.

El cliente envía un `User-Agent` honesto (`AXIOMA-JevWorkflow/1.0`) y `Accept: application/json`; una prueba unitaria verifica ambos encabezados y que no haya reintento. El23-09 una petición autenticada de cuatro tareas de etapa3 terminó correctamente con una solicitud y5,73s, [recibo](../../campanas/etapa3_polaridad_20260923_01/jev_01/receipt.json). No se ha demostrado que el encabezado fuera la causa única de los 403 anteriores: un POST sin clave devuelve también HTTP403 con error de autenticación tanto con UA predeterminado como explícito. El error1010 conservado en una corrida anterior sí señala una regla de firma del cliente, pero el 403 de la ronda inmediatamente previa no guardó cuerpo. No repetir solicitudes fallidas con marcador `.pending` sin aclarar el recibo.

Para fallos futuros, la respuesta HTTP se reduce a `http_status` y una categoría segura (`authentication_error`, `cloudflare_1010` u otra etiqueta restringida). Nunca se guarda el cuerpo remoto ni un eco de la clave. Una prueba con cuerpo artificial comprueba la categoría, la ausencia de texto remoto en la excepción y el marcador de no reintento. Las 15 pruebas locales pasan; no se hizo otra consulta pagada para probar este manejo de errores.

## Prueba real realizada

Una petición con tres tareas redactadas para este piloto: reparación del posprocesamiento→Codex; revisión independiente→ChatGPT; diseño del motor→razonamiento Codex. Las tres coincidieron con la división de trabajo prevista.0,708s;1041 tokens de entrada;210 de salida; coste estimado US$0,000043722 a US$0,042 por millón de entrada. El precio es la tarifa documentada consultada, no una factura. [Modelos y precios](https://docs.typesafe.ai/models).

Catorce pruebas locales cubren rechazo de secretos, entradas acotadas, respuestas incoherentes/no finitas, modelo inesperado, redirecciones, ausencia de reintentos, reutilización sin credenciales, invalidación al cambiar la tarea, corrupción de caché y bloqueo de duplicados. Se importó además la respuesta real del piloto y se reutilizó en dos ejecuciones con cero llamadas nuevas. El piloto prueba conectividad y tres casos simples; no demuestra calibración, ahorro agregado de tokens ni mayor calidad científica. Clasificación: PROMETEDOR_NO_CONFIRMADO para triage.

Para que aporte ahorro deberá filtrar lotes de tareas o pasajes que de otro modo leería un modelo más caro. Invocarlo para cada suma o para decisiones triviales añadiría coste. La selección de documentos puede usar el mismo endpoint con preguntas pertinentes, pero no se ha implementado todavía una búsqueda documental general. [API oficial](https://docs.typesafe.ai/api).

## Coordinación con ChatGPT

La app permite leer y enviar mensajes a «Investiga el conectoma de mosca». El usuario autorizó hacerlo directamente; el envío local de una petición con datos resumidos ya funcionó. Eso no adjunta un ZIP ni da a ChatGPT acceso a los discos locales. La carpeta común es `F:\Downloads\AXIOMA_INTERCAMBIO` (`/mnt/f/Downloads/AXIOMA_INTERCAMBIO`): entrada, salida y recibos. Acceso bidireccional Windows/WSL comprobado con hashes. No hay bucle de agentes permanente ni tareas programadas por esta integración. El adjunto sigue pendiente de recuperar Computer Use y comprobar PRO/razonamiento máximo en la interfaz real.
