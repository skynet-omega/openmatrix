# Suplemento descriptivo y cierre

Los análisis siguientes usan únicamente datos expuestos ya incluidos en la cápsula principal. No añaden replays, selección de parámetros ni capacidad neural.

El reinicio frío cambia el error final en0.001970010° frente al replay continuo con las mismas órdenes. Por tanto, el defecto del primer torque es real y debe corregirse para nuevas intervenciones, pero no explica el empeoramiento de8,144° observado aquí. Es una comparación posterior; no reemplaza el control sin viento pendiente.

La auditoría del filtro reproduce lector, EMA y relé exactamente. La integral posviento pasa de+0,014936810° crudos a+2,19° aplicados; el campo L−R sigue negativo. Es evidencia de amplificación de una señal direccional neta inconveniente en esta vida, no una localización neuronal ni permiso para invertir el lector.

[Decisión y rivales](DECISION.md). [Código del filtro](audit_filter.py), [resultado](FILTER_AUDIT.json), [código del reinicio](audit_restart.py), [resultado](RESTART_ARTIFACT.json).

La cápsula principal: https://github.com/skynet-omega/openmatrix/tree/b572a2661b77ed5f5499995e633812bd7842af0c/reviews/stage45-postwind-diagnosis-20260925-41-ce7c556bd77b

Su ZIP remoto se descargó y verificó íntegro (80archivos), y la extracción remota volvió a reconstruir las métricas byte idénticas. La prueba corporal completa en extracción nueva pertenece a la cápsula local previa: sus fuentes, modelo, estados y trazas coinciden por hash con los publicados. La reproducción del organismo neuronal no forma parte de esta entrega.

## Revisión final y cálculo solicitado

ChatGPT no consiguió abrir ninguno de los cuatro archivos RAW; su revisión final es condicional al resumen. [Original íntegro](CHATGPT_RESPONSE_02.md). Su petición de separar mando solicitado de giro realizado se implementó localmente en [command_accounting.py](command_accounting.py), con [contrato diagnóstico](ACCOUNTING_PLAN.json) y [resultado normal/-O idéntico](COMMAND_ACCOUNTING.json).

En identidad: G=0.042887171708, C=0.015088712492, M=0.000289281188rad²; residual1.39e-17rad². El mando fue adverso en507ms, correctivo en69ms y nulo en404ms. C>0: el mando entregado ya pedía un giro neto adverso; no era simplemente el cuerpo desobedeciendo una solicitud correctiva. M no es mecánica pura, y no se aplicó el umbral conductual0,5° a términos enrad².

Cada orden de la fila1021 gobierna el intervalo1020→1021ms: se usa el estado completado1020ms como frontera inicial y las980 órdenes siguientes. No se desplazaron muestras para mejorar ajuste. La igualdad contable sola no verifica ese reloj; el runner y el verificador comprueban contexto y orden por separado.

| Intervención frente a identidad | Cambio de bearing final (°) | Menos cambio de yaw final (°) | Cambio de error firmado (°) |
|---|---:|---:|---:|
| zero_yaw_after_wind | +1.124929162 | +2.184592746 | +3.309521908 |
| zero_forward_after_wind | +5.254625611 | -0.011224830 | +5.243400781 |

Se mantiene A: revisar signo/marco/tiempo y transformación de la señal hacia el mando, junto con avance. El resultado no localiza aún el origen neuronal, no justifica subir amplitud ni requiere integrar patas. El prefijo físico hasta1020ms es exactamente igual en todas las condiciones útiles, incluida la semántica fría del reinicio.

Reproducir los suplementos tras extraer la cápsula principal y copiar estos archivos en su raíz:

```bash
python -B audit_filter.py --out FILTER_NEW.json
python -B audit_restart.py --out RESTART_NEW.json
python -B command_accounting.py --out COMMAND_NEW.json
```
