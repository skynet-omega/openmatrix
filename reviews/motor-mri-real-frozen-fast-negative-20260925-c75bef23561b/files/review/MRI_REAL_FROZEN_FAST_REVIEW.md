# MRI sobre un bloque real: negativo del proveedor rápido congelado

La pregunta fue si el esquema CPU de cinco consultas completas entregado por ChatGPT podía pasar **un bloque aceptado real de 125 µs** usando como `F_fast` la dinámica diagonal `r₀·(a₀−z)`, con proyección exacta de puertos SET/ADD. La corrida sham de 1 ms mantuvo el organismo en el motor padre. El replay diagnóstico se hizo después de calcular el bloque y antes de publicarlo: consultó el operador efectivo completo en el mismo preparado y comparó contra el endpoint padre del mismo bloque. No se ajustaron parámetros tras observar el resultado.

| Medida | Resultado | Puerta prospectiva |
|---|---:|---:|
| Consultas completas | 5 | ≤5 |
| Consultas rápidas locales | 229 | coste completo aún sin puerta válida |
| Eventos físicos del bloque | 7 | sin pérdidas |
| Error final normalizado | **1,865112** | ≤1 |
| Estimador embebido | 0,186039 | ≤1 |
| Defecto muestreado | **146,659324** | ≤1 |
| Pared del replay CPU/GPU | 3,066 s para 125 µs | no candidata de velocidad |

Sólo una coordenada del estado final excedió 1 tolerancia normalizada: índice 29.460, `bodyId 42975`, tipo `hDeltaA` en `nodes.parquet` MaleCNS v1.0 (error 1,865; diferencia absoluta 1,922×10⁻⁶). Le siguió `FB4K` con 0,344. El muestreador de defecto detectó una discrepancia mucho mayor antes del final; el estimador embebido por sí solo habría dado falsa tranquilidad. Esta observación **descarta este proveedor diagonal congelado**, no todas las variantes multirritmo con corrección recurrente.

El par de salidas científicas padre/replay es idéntico al control sin instrumentar: 574 arrays del estado rehidratado, eventos, trazas, pesos, cuerpo y salidas publicadas. El [verificador](verify_mri_real_frozen_fast.py) recalcula endpoint y embebido desde los estados crudos, compara al padre, funciona también bajo `-O` y rechaza corrupción no finita. No puede recalcular de forma independiente el defecto sin el operador vivo; su cifra procede del código congelado y queda con ese límite explícito. El conteo de aristas del donante es sólo **cota inferior** para los cinco barridos CSR: no incluye propietarios PN/KC/visual, operaciones locales ni preparación, de modo que el `work10x` interno del donante **no es una puerta de velocidad válida**.

Procedencia: el plan50 falló por sintaxis antes de cargar el organismo. El plan51 ejecutó el padre idéntico pero el donante rechazó las fechas de eventos en orden de productor (`Fechas/orden`); no hubo trayectoria candidata. El plan52 ordenó únicamente las fechas de corte, dejando el orden SET/ADD del propietario físico intacto. Esa corrida terminó dentro de 36 s, 5,83 GiB RSS y 384 MB de archivos, bajo los topes congelados de 240 s/12 GiB/700 MiB. El archivo de ChatGPT conservó su SHA-256 original `03ffdf6038772226d4797b837c5ed08275d470019925f697c56d5a23857a15fc`.

Siguientes rivales no fusionados: **A**, `F_fast` disperso de la zona de influencia de eventos con recurrencia local y corrección global MRI; **B**, integrador implícito/exponencial Krylov con JVP completo contabilizado; **C**, agenda QSS/asíncrona con cotas de influencia e invalidación. Para A, un proveedor nuevo debe hacer visible coste de proyección, vecinos, propietarios y correcciones y pasar bloque/estados/eventos antes de ensayar el organismo más largo. Ninguna de estas rutas supera hoy la meta de 5 s en 5–10 min ni admite Etapas 4/5.
