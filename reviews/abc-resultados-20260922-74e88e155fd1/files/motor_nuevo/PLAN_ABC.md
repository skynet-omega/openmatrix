# Plan vigente: tres hipótesis rivales

Corrección del usuario del 22-09-2026: mínimo tres hipótesis sustancialmente distintas por decisión arquitectónica, incluido motor y etapa 3. El padre es control y no cuenta como hipótesis. No obliga a construir tres organismos completos simultáneos: comparar primero el discriminador causal más barato y después como máximo dos integraciones completas por ronda. Ningún resultado numérico sustituye validación biológica ni admisión de etapa 3.

| Hipótesis | Operación nueva y limitación atacada | Falsador práctico |
|---|---|---|
| A: integración implícita conjunta | Resolver q y transmisión s en el mismo paso, conservando interacción recurrente; permite pasos mayores si converge. Trapezoide con resolución diagonal exacta e iteración del acoplamiento. | No converge bajo presupuesto fijo, falla error o las iteraciones cuestan más que B. |
| B: ejecución residente en GPU | Todas las etapas RK4 y el estado de la red permanecen en VRAM; elimina recorridos por Python/CPU entre operaciones. | No conserva las ecuaciones o el coste de recorridos completos impide alcanzar el objetivo. |
| C: propagación incremental | Mantener sumas sinápticas y actualizar columnas CSC sólo cuando cambia su transmisión más que un umbral; estados siguen siendo graduales. | Densidad de cambios/atómicos elimina la ventaja, o error acumulado supera el contrato. |

Información legal: conectividad, parámetros y estado de la misma preparación; entradas prescritas compartidas. Ninguna usa una salida correcta ni ayuda conductual del evaluador. Son ingeniería de métodos conocidos (trapezoide, RK4, actualización dispersa), no novedad biológica. A difiere del padre en integración conjunta; B en organización de ejecución; C en propagación de cambios. No se fusionan por votación.

Primer discriminador: recurrente base con 166.700 neuronas y 25.582.938 conexiones almacenadas. Conserva las ecuaciones q/s base y metadatos exportados. **No contiene reemplazos PN/KC/APL/retinales especializados ni cuerpo:** no es una simulación de la mosca completa. La reproducción de ese bloque debe preceder al coste de portar dichos subsistemas; no basta para promover motor.

Contrato y presupuesto prospectivos: `abc_20260922/contract.json`. Comparar contra RK4 refinado, conservar trayectorias y fallos. No ajustar umbrales tras ver resultados. Una alternativa que pase podrá ejecutar 1 segundo del bloque base, claramente separado de la meta de cerebro+cuerpo completo. Posteriormente: portar PN conjunto, eventos KC y acoplamiento corporal con reloj preservado.

Publicar fuentes y evidencia por manifiesto explícito en OpenMatrix; ChatGPT revisa archivos fijados al commit, declarando qué leyó/ejecutó. Jev clasifica tareas acotadas. Cero subagentes Codex. Publicación automática al ejecutar el cierre, sin vigilancia periódica ni daemon implícito.

## Próxima ronda de etapa 3, después de disponer del motor necesario

Tres explicaciones rivales del giro común en ambas lateralidades; son hipótesis pendientes, no conclusiones ni experimentos ejecutados en esta campaña:

- **A, pérdida de señal lateral dentro de la red:** seguir diferencias emparejadas PN→CX→DN con la misma preparación, y localizar dónde se extinguen o invierten. Falsador: lateralidad neural correcta y suficiente hasta la salida motora; entonces buscar el fallo fuera de ese tramo.
- **B, sesgo del estado inicial o adaptación:** comparar olor izquierdo, derecho, uniforme y sham partiendo del mismo estado completo, y contrastar historias de preparación previamente fijadas. Falsador: el mismo defecto persiste pese a controlar la historia y la contribución basal. No rescatarlo reescalando ORN después de ver conducta.
- **C, desacople entre salida neuronal y cuerpo:** comprobar mediante intervenciones diagnósticas emparejadas la correspondencia entre las salidas efectivamente leídas (incluida DNb05) y signo/magnitud del giro. Falsador: cuerpo y lectura responden correctamente a las salidas espejo; el origen vuelve a buscarse en la red.

Estas intervenciones son herramientas externas del evaluador; no se convierten en controlador del organismo. Los registros de moscas vivas deben emparejar preparación, variable observada e intervención pertinente antes de trasplantar cifras. El negativo PFG ya probado permanece negativo. El contrato y presupuesto de esa ronda aún no se han congelado; no activar estas pruebas como obligación previa a mejorar el motor.
