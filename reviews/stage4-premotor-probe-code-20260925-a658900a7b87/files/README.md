# Campaña38: reclutamiento causal premotor

Estado: campaña acotada iniciada, sin resultado biológico todavía. Etapas4/5 abiertas.

La alternativa B pregunta si DNp09 bilateral o DNa03 unilateral responde y transmite una señal bajo las leyes vigentes, antes de diseñar otro lector motor. A conserva la alternativa de interfaz identificada; C, coordinación conjunta de avance y giro. El lector normalizado de campaña37 fue descartado y no se rescata ajustándolo.

Cuatro condiciones, cada una con40ms de preparación idéntica y200ms de ensayo: sham, DNp09 bilateral, DNa03 izquierda, DNa03 derecha. Pulso fijo durante[20,60)ms: la primera evaluación nativa deriva la corriente adicional para recorrer25% del rango restante de su target instantáneo. No cambia pesos, umbrales, ganancias, estado q ni línea base. La normalización volumétrica histórica ya está incluida; no se reaplica.

Los mandos corporales y olores provienen de la misma cinta de campaña36. Se verifican estados de integración y fuerzas en cada paso corporal; buffers sensoriales efectivamente copiados en los8 intercambios aceptados de125us y sus8 predictores descartados de62,5us. Los estados recurrentes y la salida PN permanecen libres. El mando neural original se observa sin mover el cuerpo con él en esta prueba de identificación.

La sonda registra22 células elegidas antes de consultar actividad. Sus máximos y conteos abarcan todas las evaluaciones, incluidas las rechazadas: no son ocupación temporal aceptada. Un máximo no positivo limita todas esas evaluaciones; un máximo positivo no garantiza respuesta sostenida. Las rutas estructurales existen hacia las22 células desde las cuatro fuentes, pero ello no prueba transmisión funcional. Una corrección de tipos en la clasificación inicial se conservó en `structure_01`; `structure_02` es la salida corregida.

Presupuesto: máximo4 organismos,5000s agregados,1300s por brazo,18GiB RAM,12GiB GPU,4GiB checkpoints y256MiB observaciones. Un fallo de integridad o límite detiene la cola. No hay calibración por giro ni aprobación de navegación.

ChatGPT proporcionó el núcleo Python original de dosis; Codex lo trasladó al consumidor CUDA real y comprueba ambos cálculos con las mismas entradas nativas. La revisión estática especializada no detectó impedimentos para sham; no ejecutó organismos. La compilación CUDA pasó. No se modifica el motor padre ni se promueve una nueva arquitectura.

El paquete de revisión incluye código del experimento y rutas calculadas, no los checkpoints ni todas las dependencias del organismo. No constituye reproducción autónoma de una corrida completa. Los resultados se añadirán una vez medidos.

Ejecución local, con el entorno GPU ya instalado y el árbol histórico conservado:

```bash
cd /home/daroch/AXIOMA_ASTRA
/home/daroch/miniconda3/envs/GPU/bin/python campanas/etapa4_premotor_recruitment_20260925_38/run_queue.py
```

La cola rechaza un directorio ya utilizado. Consultar `QUEUE.json` y los `RESULT.json` de cada brazo antes de decidir una reanudación; este comando no restaura automáticamente un organismo.
