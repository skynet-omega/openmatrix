# Bloque real, precisión y separación de eventos — 24-09-2026

**Resultado:** captura completa neutral, bug de memoria del oráculo reparado, instancia MRI de ChatGPT descartada por coste y primitiva CUDA dispersa conservada como donante. **No se cambió el motor operativo ni se alcanzó la meta de velocidad.** Etapa3 conserva su confirmación funcional local; navegación de etapa4 y etapa5 siguen abiertas.

| Prueba ejecutada | Resultado | Alcance |
|---|---|---|
| Captura de1ms del organismo, incluyendo un bloque aceptado de125µs | 580 arrays serializados, RNG, eventos, trazas y contadores idénticos al padre; 57,645s con captura/compresión | Neutralidad en este sham; no tiempo de simulación ordinaria ni perfección universal |
| Oráculo de coeficientes, canarios de memoria | La versión anterior devolvía resultados correctos y corrompía buffers vecinos; pool privado conserva ambos | Reparación del instrumento nuevo, no diagnóstico de corrupción del motor operativo |
| MRI33 original de ChatGPT | Fixture CPU pasa, error2,37e−9; con siete eventos reales requiere≥49 evaluaciones frente a60: techo1,2245× | Descartado por coste antes de gastar un replay GPU; no se ejecutó el integrador sobre la mosca |
| CSR reducido de eventos, 60 consultas reales CUDA | 885.587 aristas,4.062 fuentes,6.342 receptores; diferencia máxima de corriente1,46e−11;19,50× menos visitas contando una partición | Sólo la contribución base de esos puertos. No incluye la recurrencia continua ni todos los reemplazos especializados |

La cápsula dispersa se recalculó en CPU en sus60 consultas; cuatro controles comprueban SET/ADD, eventos tardíos, consultas retrospectivas y versiones. Los q/s reconstruidos coinciden con la captura hasta5,55e−17. Sus4.060 valores distintos de tau_q se conservan: no se fabricaron dos grupos biológicos.

**Límite de rendimiento:** preparar la partición costó236,6ms, superior al ahorro nominal de estas60 consultas. Además, la suma de intervalos CUDA (70,355ms) excede la pared del bucle (65,160ms): sus cocientes18,50×/14,72× no se admiten como medición fiable de velocidad. No se repitió el benchmark para buscar un resultado conveniente. La reducción de aristas y las comparaciones de corrientes son independientes de ese problema de relojes.

La pieza dispersa permite evaluar eventos sin recorrer el96,54% de aristas continuas restantes, pero **no elimina la necesidad de calcular esa recurrencia**. El siguiente discriminador debe separar compilación de topología, actualización de pesos/puertos y avance temporal. Reutilizar una partición sólo es legal mientras sus dependencias no cambien. Véase [A/B/C y falsadores](NEXT_SPLIT_ABC.md): MRI con corrección recurrente, QSS2/CSC y Krylov. Máximo dos prototipos completos; no se promociona esta primitiva como motor general terminado.

## Precisión: corrección a la propuesta externa

El ruido biológico debe representarse mediante un modelo estocástico explícito. El error numérico puede divergir, pero también permanecer estable y sesgar el resultado. Coincidir bit a bit en una ejecución no demuestra «cero bugs»: el caso de los canarios lo muestra directamente. No se necesita un integrador universal perfecto para ensayar un modo rápido; sí tolerancias y observables definidos antes de ejecutar, además de controles adecuados. La diferencia de yaw3,42e−7° observada antes no era el umbral: éste era0,002°.

ChatGPT aportó código completo y un límite de coste; Codex verificó su hash y ejecutó el fixture. Su mensaje llegó truncado a20.000 caracteres, pero el código está completo y coincide con su SHA; el ZIP remoto y la suite ampliada no se recuperaron. Un especialista detectó el problema de memoria y revisó el alcance del oráculo; otro implementó la sonda dispersa. Jev priorizó separar eventos/cálculo global en una consulta acotada. Sus opiniones no sustituyen las mediciones ni una auditoría independiente del organismo.

## Evidencia y reproducción

- Captura local: `capture_01/`; inventario completo en `CAPTURE_MANIFEST_01.json`, neutralidad en `NEUTRALITY_01.json`.
- Fuentes de captura originales conservadas en `capture_01/executed_sources/`; `effective_oracle.py` contiene la reparación posterior. `ORACLE_MEMORY_TEST_01.json` conserva el fallo y el pase.
- MRI: `chatgpt_original/mri33_replay.py`, `chatgpt_cpu_01/RESULT.json`, `CHATGPT_REAL_WORK_FLOOR_01.json`.
- Operador portátil: `event_sparse/`; cápsula numérica y corrientes en `event_sparse/gpu_01/`, con hashes y controles. Es autocontenido para esta primitiva; **no** reconstruye el organismo entero.

Entorno ejecutado: Python3.10.18, NumPy1.26.4, SciPy1.15.3, CuPy13.6.0/CUDA12, RTX4070Ti SUPER. Desde una extracción limpia del paquete, con Python/NumPy/SciPy disponibles:

```bash
python -B event_sparse/verify_capsule.py event_sparse/gpu_01 --out VERIFY_NEW.json
OPENBLAS_NUM_THREADS=1 python -B chatgpt_original/mri33_replay.py --selftest --out MRI_NEW
```

El primer comando reconstruye todas las corrientes portadas en CPU y coteja los outputs GPU conservados. El segundo ejecuta el fixture del integrador; ninguno simula de nuevo el organismo. La sonda GPU completa usa `event_sparse/run_probe.py --capture <captura_local_completa> --out <directorio_nuevo>` y requiere el inventario efectivo local que no se duplica en el paquete compacto. Las60 trazas completas y checkpoints del organismo permanecen locales, con hashes; el paquete no los sustituye por enlaces ni afirma contenerlos.

Presupuesto consumido: una captura de1ms/57,645s y812,3MB; un discriminador disperso de4,47s/12,1MB,1,51GB RAM y0,66GB VRAM adicionales observados; cero replays de candidatos sobre el organismo y cero etapas4/5 nuevas. El oráculo corregido sólo pasó aquí la prueba sintética de propiedad de memoria: todavía debe ligarse a versión/fase y contrastarse fuera de trayectoria en un replay real. La clasificación de la pieza dispersa es **PROMETEDOR_NO_CONFIRMADO**; la instancia MRI global por corte queda **DESCARTADO** para el gate10×.
