# Campaña41 — diagnóstico físico después del viento

**Clasificación: PROMETEDOR_NO_CONFIRMADO. Etapas4/5 abiertas.**

Reproducción del cuerpo MuJoCo y la prótesis con órdenes congeladas de una sola vida expuesta de campaña40. No se cargó ni simuló el cerebro. Las intervenciones son instrumentos del evaluador; no son políticas propuestas para el organismo.

| Condición | Error final a fuente (°) | Cambio desde1020ms (°) | Mejora frente a identidad (°) |
|---|---:|---:|---:|
| Órdenes originales y reinicio original | 27.558233 | +8.144071 | +0.000000 |
| Giro nulo después del viento | 24.248712 | +4.834549 | +3.309522 |
| Avance nulo después del viento | 22.314833 | +2.900670 | +5.243401 |

Descomposición geométrica desde1020ms: Δdirección a la fuente=-5.905064°, Δyaw corporal=+2.239007° y Δerror firmado=-8.144071°. Se cumple Δerror=Δdirección−Δyaw. Son términos geométricos; no porcentajes aditivos de causalidad.

- Giro nulo después del viento: mejora=+3.309522°; supera el mínimo prospectivo de0,5°: **sí**.
- Avance nulo después del viento: mejora=+5.243401°; supera el mínimo prospectivo de0,5°: **sí**.

El primer replay continuo falló identidad: era exacto hasta1000ms y divergía tras el reinicio histórico. La sonda sin integración midió fuerza generalizada nula en la primera llamada de viento con datos MuJoCo fríos, frente a norma 0.00467269785715347 con cinemática consistente en un estado auxiliar. El replay reparado reproduce el reinicio y su primera llamada nula; no modifica retrospectivamente la campaña40. Su negativo de navegación permanece.

La identidad reparada tiene errores máximos qpos=0, qvel=0, yaw=0°. El brazo sin viento quedó **NO EJECUTADO**: el intento fallido consumió una de las cuatro ejecuciones. No se amplió el presupuesto ni se relajaron criterios.

Presupuesto consumido: 110.957/600s de proceso de los replays (incluye el fallo);8000ms corporales y cuatro intentos. Cero organismos, GPU, entrenamiento o ajustes de ganancia. Tres condiciones útiles; no cohorte de semillas ni confirmación reservada.

Los efectos pueden interactuar. Las órdenes permanecen congeladas al cambiar la trayectoria; por ello no se predice la respuesta neural al retirar avance o giro. Detener el avance puede contener el error angular sin acercarse al objetivo: es un diagnóstico, no navegación. El efecto mecánico emparejado de retirar viento sigue pendiente.

A sigue prioritaria: identificar una interfaz conjunta avance/giro con estímulos independientes y controles sensoriales emparejados. B (CNS→VNC→MN) queda acotada; C (músculos y seis patas) posterior. No integrar patas ni rescatar el filtro con otro umbral sobre esta vida.

Fuentes, contrato, fallo, reparación, trazas, modelo MuJoCo y estados del controlador están incluidos. El paquete reproduce este experimento físico; no contiene el cerebro completo ni acredita aprendizaje o equivalencia biológica.
