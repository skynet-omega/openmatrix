# Criba de trayectoria prolongada con el organismo completo — 25-09-2026

La Campaña27 mostró una diferencia bilateral de giro de 0,0573205° a 400 ms, pero el error de rumbo respecto de cada fuente **aumentó** y el avance fue casi común. Esta campaña extendió únicamente el brazo de fuente derecha a 1 s con la misma preparación, organismo PN629-off, campo gaussiano, lector DNb05 y cuerpo. Se pausó el trabajo de microkernels durante la corrida. [Plan prospectivo](PLAN.json), [46 fuentes fijadas](SOURCE_LOCK.json), [runner](run_long.py), [verificador independiente de trazas](verify_long.py).

**Resultado: criba negativa para corrección tardía de rumbo en esta vida y horizonte.** El organismo completó40ms de preparación y1000ms de ensayo en3801,643s reales, dentro del techo4300s. Los primeros400ms de qpos, posición, campo, sensores, mando y yaw son **exactamente** los de la corrida histórica. La [verificación cruda](RAW_VERIFIED_01.json) reconstruyó la trayectoria y produjo el mismo JSON byte por byte bajo `python -O` ([copia](RAW_VERIFIED_02.json)).

| Observable | Preparación | 400 ms | 1000 ms |
| --- | ---: | ---: | ---: |
| Error absoluto de rumbo a la fuente derecha | 18,37685° | 19,72289° | **22,11678°** |
| Distancia corporal a la fuente | 1,13125 mm | 1,05660 mm | 0,94591 mm |

El bearing empeoró3,73994° desde el inicio y2,39389° después de400ms; falló ambos mínimos de mejora≥0,5° preregistrados. La distancia bajó0,18534mm con avance constante0,2mm/s, de modo que ese acercamiento no demuestra búsqueda dirigida. El cuerpo mantuvo cuatro contactos activos y upright mínimo0,97746; el olor reconstruido coincide con las antenas, el lag de1ms y el lector motor exacto pasaron.

La [descomposición descriptiva posterior](DIAGNOSIS_01.json) también se reprodujo byte a byte en `-O` ([copia](DIAGNOSIS_02.json)). En la trayectoria observada, la dirección de la fuente cambió−3,77104°, el yaw corporal−0,03110° y el mando neuronal integrado−0,02280°. El contraste consumido L−R siguió presente (media−0,37985 al inicio,−0,36443 al final); no hubo saturación del límite ±5°/s. Esto localiza una diferencia grande entre la demanda geométrica de orientación y el mando real, pero no atribuye causalmente la pérdida a una neurona, al lector o al cuerpo. La Etapa4 navegación y la Etapa5 causal siguen **abiertas**.

La criba compara tres explicaciones: **A**, la orientación dirigida aparece con retraso; **B**, el giro inicial no se convierte en corrección de rumbo; **C**, la corrida es incomparable por reloj, entrada, soporte o prefijo. El éxito exploratorio exige una reducción ≥0,5° del error de rumbo desde la preparación y desde 400 ms, además de apoyo corporal y concordancia con los primeros 400 ms históricos. Incluso un A positivo necesitaría control sham/yoked para demostrar navegación en bucle cerrado; esta única vida no cierra Etapa4 ni Etapa5. No se aplica viento ni un giro de 30°: con el mando actual ±5°/s no cabría recuperar 30° en el segundo siguiente.

Presupuesto registrado antes de la corrida: un organismo completo de40+1000ms, ≤4300s de pared, ≤18GiB RAM, ≤12GiB GPU, salida proyectada≤3GiB, reserva C:≥30GiB. Un primer lanzamiento falló **antes del primer paso** porque solicitó diez capturas y el observador admite ocho. El [resultado fallido](native_minus_01/RESULT.json) y el [recibo de reparación](PREFLIGHT_REPAIR_01.json) permanecen. La repetición `native_minus_02` conservó los criterios y usó ocho capturas; produjo1,4GiB en Linux. No se repite con ajuste de parámetros para rescatar el negativo.

Tras completar, la verificación lee los NPZ y escribe un recibo nuevo, sin confiar en el campo de éxito del runner:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python campanas/etapa4_long_trajectory_20260925_35/verify_long.py --run campanas/etapa4_long_trajectory_20260925_35/native_minus_02 --out campanas/etapa4_long_trajectory_20260925_35/RAW_VERIFIED_01.json
```

El modo de verificación `-O` usa excepciones explícitas y produjo las mismas métricas. Jev priorizó una criba de autoridad corporal como próximo control consultivo, sin ver esta trayectoria. [ChatGPT respondió](CHATGPT_STAGE45_RESPONSE.md) sin ejecutar el organismo ni la traza de1s: si mejora bearing, propone fuente desplazada online/yoked; si no, ablación del contraste lateral conservando intensidad común; una perturbación corporal pequeña y emparejada vendría después. Su cálculo cinemático recuerda que avanzar hacia una fuente lateral puede reducir distancia mientras **empeora** bearing. Son propuestas, no resultados nuevos. La próxima decisión debe comparar la ablación lateral con una verificación del lector/efector bajo información de entrada emparejada; el control corporal previo de ±5°/s ya mostró una separación física de2,9789° y no es necesario repetirlo sin una pregunta nueva.
