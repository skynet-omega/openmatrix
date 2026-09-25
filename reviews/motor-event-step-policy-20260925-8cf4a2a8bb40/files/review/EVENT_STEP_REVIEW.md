# Política de paso tras eventos — resultado prospectivo del 25-09-2026

El controlador nativo anterior calculaba la próxima propuesta a partir del paso **realizado** incluso cuando una frontera física había recortado una propuesta mayor. Con error aceptado pequeño, eso reducía artificialmente el tamaño de los pasos posteriores. La candidata v2 conserva la propuesta previa sólo después de un **evento interior** que recortó el paso y cuyo error fue menor que el umbral existente `0.1`. Los cortes, las seis evaluaciones de error, la norma, los parámetros biológicos y el cuerpo no cambiaron. El paso siguiente se vuelve a probar y puede rechazarse.

| Ensayo pareado, sham | Padre | Candidata v2 | Comprobación |
|---|---:|---:|---|
| 1 ms: subpasos CNS aceptados | 191 | 133 | 0 rechazados; estado final máximo 0,192 tolerancias normalizadas; 84 eventos emparejados |
| 1 ms: pared del paso acoplado | 3,560 s | 2,927 s | 17,8% menos, un único par |
| 20 ms: subpasos CNS aceptados | 3.677 | 2.324 | 0 rechazados; estado final máximo 0,932 tolerancias normalizadas; 1.564 eventos emparejados |
| 20 ms: pared de los 20 pasos acoplados | 63,939 s | 49,238 s | 23,0% menos, un único par |

El ensayo de 20 ms mantuvo la misma preparación de pesos (SHA-256 `6d8d339e29e5229ad0089d5512fe4e31b4d2bddf1a61a79611ed23b9e5240cfd`). Orden e identidad de los eventos coincidieron; su máxima diferencia temporal fue `4,50e-10 s`, dentro de la puerta previa `1e-9 s`. La diferencia máxima de yaw fue `2,37e-10°`. La verificación reconstruye cifras desde estados, trazas y auditorías, pasó también bajo `python -O` y rechazó una corrupción de estado.

**Procedencia de fallos y alcance.** El primer intento de arranque importó prematuramente un módulo de eventos y terminó con `0 ms` simulados; queda en `event_step_baseline_01`. La v1 de la candidata redujo 191→104 pasos, pero también conservaba propuestas al final de cada época. Su aparente ganancia no se atribuye sólo a eventos y queda como exploratoria en el plan41. ChatGPT señaló la distinción y la v2 la implementó antes de los pares definitivos; un fixture con evento terminal prueba que allí la política no cambia. Jev priorizó la prueba A como asesoría; no ejecutó código. ChatGPT revisó el controlador y advirtió que un error pequeño de la cola no certifica el siguiente paso; por eso la candidata mantiene la evaluación completa y el rechazo.

**Resultado científico:** `PROMETEDOR_NO_CONFIRMADO` como reparación genérica del planificador temporal. Dos pares sham, de 1 y 20 ms, no validan olor, visión, otras preparaciones, horizonte de 5 s ni velocidad de 5 s en 5–10 min. El mayor error de estado a 20 ms llega a 0,932 del límite 1,0 y exige vigilancia antes de ampliar. La mejora de 23% sobre el paso acoplado deja ≈2,46 s de pared por ms en este ensayo; extrapolar linealmente a 5 s daría horas, no minutos. No se modifica aún el motor operativo.

**Tres rutas vigentes:** A, la política v2, conserva una mejora parcial pero tiene techo por los cortes obligatorios; B, caché de transmisión CSC, tiene dos primitivas CUDA negativas bajo puertas congeladas (0,1095/0,1168 ms por consulta y error E 0,0291), por lo que no recibe más retoques en esta ronda; C, MRI de corrección recurrente o integración implícita/Krylov, debe reducir evaluaciones globales con error y coste integral medidos. La próxima decisión es un discriminador de C con el operador completo y conteo de todos sus productos, no otro ajuste del factor de crecimiento de A. Etapa 4 (navegación espacial) y etapa 5 (perturbación causal) siguen abiertas.

En el paquete compacto, `code/verify_event_step_capsule.py --root .` reconstruye el resultado de 20 ms usando `data/`, `results/` y `plans/`. Los checkpoints completos permanecen locales con sus hashes en `data/EVENT_STEP_CAPSULE_MANIFEST.json`; la cápsula no los sustituye como estado reanudable.
