# Motor C++/CUDA: resultado de la prueba real del 25 de septiembre

**Hay una implementación ejecutable del núcleo CNS, probada dentro del organismo
completo. No está admitida como sustituto del motor estable: pasa 20 ms y falla
la comparación estricta de 100 ms.** Clasificación global:
PROMETEDOR_NO_CONFIRMADO. La prueba dejó una implementación y un fallo medido;
no demuestra todavía un motor general completo ni la meta de velocidad.

El núcleo genérico (`graph_runtime.py` + `resident_controller.cu`) implementa
RK3(2) FP64, decisiones adaptativas residentes, eventos obligatorios y
restauración de una época fallida. El modelo aporta las ecuaciones mediante
un RHS capturado en CUDA. El núcleo no contiene nombres de neuronas, objetivos
conductuales ni escalas de olor. El adaptador conserva el modelo existente,
incluidas sus prótesis y limitaciones; esta ronda sustituye sólo el integrador
CNS. Los solvers de PN y membranas espaciales conservan sus masas y ecuaciones.

## Mediciones integrales

166.700 neuronas canónicas, 359.373 estados CNS y 25.582.938 pesos originales,
sin normalización. Mismo checkpoint asentado y misma preparación por pareja.
El cuerpo y la entrada sensorial se ejecutan en cada corrida. El estímulo es
el campo lateral estático existente, no una pluma gaussiana ni una ráfaga.

| Prueba | Referencia: avance | Candidata: avance | Comparación completa |
|---|---:|---:|---|
| 20 ms, sin olor, precisión original | 98,785 s | 104,466 s | PASA |
| 100 ms, olor lateral, precisión original | 363,363 s | 287,895 s | FALLA |
| 100 ms, olor lateral, rtol/atol CNS diez veces menores en ambos | 371,029 s | 292,664 s | FALLA |

Los tiempos incluyen el avance neuronal y corporal; carga y guardado quedan
fuera de esa columna y constan separadamente en cada RESULT. La candidata fue
5,75 % más lenta en la primera pareja y consumió 20,77 % / 21,12 % menos tiempo
en las dos parejas de 100 ms. Son observaciones sin exclusividad instrumentada
de GPU; hubo otra sesión activa durante parte del trabajo. No certifican una
ganancia universal ni un rendimiento sostenido durante segundos.

## Qué falló y qué se conservó

Los umbrales se fijaron antes de la primera corrida y no se relajaron:

| Medida máxima, pareja refinada de 100 ms | Observado | Límite |
|---|---:|---:|
| Error CNS normalizado, todas las coordenadas/muestras | 3,248455 | 1 |
| Diferencia de voltaje espacial | 0,0000281492 mV | 0,000020 mV |
| Diferencia de compuertas espaciales | 0,000000450557 | 0,0000002 |
| Diferencia temporal de evento | 9,898693 ns | 1 ns |

El peor error CNS aparece a 89 ms, coordenada 58.613. La diferencia absoluta
máxima CNS es 1,383115e-6. Relojes discretos, estructura y cantidades de eventos,
contadores y RNG coinciden; cuerpo, sensores, estados axonales y los 44 campos
PN comprobados pasan sus límites. Se compararon 7.788 registros de eventos,
**incluidos predictores**, no 7.788 espigas físicas distintas. El cuerpo mantiene
prácticamente la misma trayectoria, pero eso no sustituye comprobar el estado
neuronal ni permite declarar navegación o vuelo.

La comparación inicial de 100 ms alcanzaba 3,510266 de error CNS. El único
refinamiento probado lo redujo a 3,248455. La referencia cambia 0,524255 al
refinarse y la candidata 0,461080, en la misma escala. Por tanto, **no basta el
refinamiento probado y no quedó demostrado que la referencia inicial explique
por sí sola la discrepancia**. No se identificó aún su causa exacta. Estos datos
tampoco demuestran que RK3(2) sea matemáticamente incorrecto o inútil: falta
aislar la diferencia temporal/numérica entre ambos métodos en este modelo.

## Qué se concretó y qué queda abierto

El núcleo, su compilador, el adaptador y el ejecutor real están implementados.
La corrida refinada gestionó 19.259 pasos aceptados y 75 rechazados dentro del
CNS, con 77.336 evaluaciones del RHS. Los rechazos también se ejercieron en el
organismo real. El núcleo recompilado pasó, en Python normal y con `-O`, los
controles adicionales de evento, rechazo y restauración exacta después de un
fallo parcial. Esos controles son sintéticos y complementan la evidencia real.

En la candidata nominal de 100 ms, el avance CNS residente ocupó 146,665 s de
287,895 s totales. Otros 141,230 s pertenecen al resto del avance. Este reparto
demuestra que mejorar sólo ese núcleo no alcanza por sí solo la meta global.
No se ejecutaron segundos de simulación ni se modificó el motor estable.

Quedan pendientes el origen de la discrepancia de 100 ms, la integración de
todos los propietarios en una infraestructura común y la meta de velocidad.
La siguiente decisión debe basarse en esa discrepancia localizada, no en
añadir kernels o algoritmos sin medirlos. La implementación queda disponible
como candidata experimental, no activada por defecto en la sesión estable.

Se ejecutaron seis corridas principales: 1.753,798 s sumados en sus recibos de
proceso y 3.143.906.493 bytes de evidencia original. Se agotó el número de
corridas previsto; compilación, empaquetado y comprobación offline se registran
aparte. Las copias de distribución contienen la misma evidencia, no nuevas
corridas. No hubo agentes ni consultas externas en esta implementación.

Una incidencia del lanzador queda registrada: la corrida candidata refinada
devolvió código 143 después de escribir COMPLETE, las 100 muestras de avance,
el estado PN y la trayectoria. Ambos NPZ pasaron la comprobación CRC y las
comparaciones leyeron íntegramente esos datos. La causa del código de salida no
se diagnosticó; no se certifica cierre limpio de ese proceso. Las demás cinco
corridas devolvieron 0. No se confundió integridad de datos con terminación
limpia.

## Reproducción y archivos

`README.md` contiene compilación y comandos de ejecución. `run_real_initial.py`
conserva la fuente exacta de las primeras cuatro corridas; `run_real.py` añade
el factor de precisión y el guardado compacto de las dos últimas. El núcleo
CUDA y el adaptador no cambiaron entre las seis corridas.

El ZIP de código contiene el núcleo, fuentes y comandos. El paquete de evidencia
contiene todas las trayectorias, eventos y estados finales PN necesarios para
recalcular las tres comparaciones. Sus snapshots PN compactos no son checkpoints
de reinicio del organismo. Ejecutar nuevamente el organismo requiere el modelo
y checkpoint locales conservados; el paquete permite verificar los resultados
guardados y compilar/probar el núcleo por separado.

Desde el paquete extraído, `python verify_saved.py` y
`python -O verify_saved.py` comprueban hashes y recalculan también los resultados
negativos. Un mensaje de evidencia reproducida no convierte esos fallos en PASS
del motor.
