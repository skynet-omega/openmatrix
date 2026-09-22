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
