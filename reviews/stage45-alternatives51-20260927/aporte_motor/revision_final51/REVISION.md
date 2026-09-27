# Revisión independiente del cierre51

**Diez brazos completos; etapas4/5 abiertas.** Recalculado directamente desde `neural_and_inputs.npz` y `traces.npz`, sin leer ni ejecutar `analyze_pilot.py` ni usar sus resultados. La revisión **no es ciega**: conocía el diseño, los umbrales y los negativos anteriores. No hubo nueva simulación CNS, GPU o cuerpo.

## Contrastes registrados

Ventana fija: muestras51–90 inclusivas. Señal neuronal: `q(bodyId10118)−q(bodyId10065)`, DNb05 izquierda menos derecha. Giro: `neural_yaw_unapplied_rad_s`, convertido a grados/s. Para aire se calcula la mitad de la diferencia entre las medias de los campos izquierdo y derecho; para conductancias, `(Golor−Gsinolor)−(Iolor−Isinolor)`.

| Contraste | Diferencia q | Giro sin aplicar, °/s | Puertas de magnitud |
|---|---:|---:|---|
| Aire, sin olor |−2,87967662e−4|−0,1044721255|Ambas superadas|
| Aire, con olor |−2,87307854e−4|−0,1150902060|Ambas superadas|
| Conductancias G frente al control I, respuesta al olor |5,41234047e−8|5,92244123e−5|Ambas incumplidas|
| Umbrales previos absolutos |1,6e−5|0,02|—|

El efecto del aire aparece también sin olor. La diferencia de esos dos contrastes es6,59808332e−7q y−0,0106180805°/s. Ambas magnitudes son inferiores a las escalas anteriores, pero el contrato no estableció una puerta de promoción independiente para esta interacción: no se crea aquí una retrospectiva.

G e I cambian la respuesta al olor respecto del padre de modo casi idéntico. La comparación específica G−I no respalda una ventaja de conductancias en esta prueba. Esto conserva el negativo del experimento ejecutado; no excluye toda ley neuronal por conductancias.

## Movimiento y dosis: qué permite afirmar el positivo del aire

**Giro aplicado exactamente cero en los diez brazos.** En los seis brazos de aire/padre el mando de avance también es cero. Sus cuerpos permanecen iguales entre condiciones examinadas; existe sólo la misma deriva mecánica diminuta registrada. Por tanto, una diferencia en el lector de giro no es navegación realizada.

G e I sí generaron avance, incluso sin olor:

| Ley | Máximo de avance sin olor, mm/s | Con olor, mm/s |
|---|---:|---:|
| G |0,0647891352|0,0647977376|
| I |0,1514752691|0,1514910791|

Ese avance basal no acredita iniciación por olor. Los cuerpos de G/I **no son idénticos** a los del padre ni entre sí. El máximo de velocidad corporal registrado ronda0,0539mm/s en G y0,1155mm/s en I.

Los sentidos opuestos del campo tampoco producen igual suma de entrada JO. En la ventana de análisis las sumas medias son5169,3852625 y5931,5382056unidades internas, con y sin olor: **14,7436% más en el segundo sentido**. Hay156 frente a179 receptores activados con la ley actual y esta geometría. La misma ley por receptor no iguala la dosis total de poblaciones desiguales. Dirección, conjunto de receptores y suma de entrada cambian juntos: esta prueba no separa una codificación específica de dirección de sensibilidad a esas otras diferencias. No se normalizan las dosis después de observar el resultado.

## Error real de unidades en el adaptador

`pilot_owners.py` pasa `body.data.qvel[:3]` directamente a `body_velocity_world_mm_s`. La traslación nativa está en centímetros, de modo que falta multiplicar por10. Fuente ejecutada: `/home/daroch/AXIOMA_FLYWIRE/matrix/src/matrix_olfactory_diagnostic.py`, líneas210–211 y290–291; copia qpos/qvel nativos, declara centímetros y convierte `position_mm=10*qpos[:3]`. También `flybody_cns_body.py:36` convierte la posición por10. Sus hashes actuales coinciden con `EXECUTED_SOURCES.json` de esta campaña.

Comprobaciones geométricas ejecutadas para los diez brazos: posición en mm exactamente igual a10×qpos; velocidad guardada en la entrada aérea igual a qvel del paso anterior; rotación consistente con su cuaternión; campo del mundo constante. La reconstrucción CPU de la entrada realmente usada coincide hasta el redondeo de las operaciones (máximo≈7,11e−15unidades). La primera revisión limitada a cuatro archivos no detectó este fallo: apareció al contrastarlos con las unidades del cuerpo y las trazas.

Recalculé la entrada con la velocidad convertida correctamente, **manteniendo la trayectoria grabada**, en los diez brazos:

| Brazos | Máximo cambio por receptor JO, unidades | Máximo cambio de suma instantánea, unidades |
|---|---:|---:|
| Padres aire0, ambos olores |1,46600121e−7|1,93436331e−5|
| Aire izquierdo, ambos olores |5,03052107e−8|6,63767878e−6|
| Aire derecho, ambos olores |5,03052036e−8|4,87511170e−6|
| G sin olor |0,0260128518|6,9133853906|
| G con olor |0,0260161635|6,9142671109|
| I sin olor |0,0541790844|14,4616525304|
| I con olor |0,0541844543|14,4630820959|

El error importa también con campo cero, porque el cuerpo en movimiento produce aire relativo. **No hay cota de sensibilidad del CNS:** estas diferencias de entrada no permiten afirmar cuánto cambiarían q, mando o trayectoria al corregirlo. No se ejecutó el organismo corregido. La evidencia original se conserva y sus contrastes describen la implementación realmente ejecutada.

## Archivos y coste

- `audit.py`: cálculo propio, lectura de arrays y verificaciones de identidad, ventanas, relojes, finitud y geometría.
- `RESULTADO.json`: números completos de cada brazo, contrastes, comprobaciones geométricas y hashes de entradas/fuentes de unidades.
- `ENTRADAS_CORREGIDAS_NO_SIMULADAS.npz`: entradas JO contrafactuales para los diez brazos sobre la cinemática grabada. No contiene una respuesta cerebral corregida.

Dos ejecuciones del análisis (la segunda añade geometría y exporta los contrafactuales):0,553531sCPU de proceso en conjunto. La inspección inicial de esquema registró0,099860sCPU; otras lecturas auxiliares no están incluidas en esa suma instrumental. Sin nuevos pasos ni cambios de raíz/core/fuentes congeladas. Revisión iniciada19:54:19UTC, bajo30sCPU y600spared.

**Dictamen:** el aire produce una señal que merece conservarse, con las limitaciones de dosis y unidades indicadas. La alternativa G no supera su control I en el contraste registrado. No se demuestra persecución del olor ni recuperación de una perturbación. Este cierre no prescribe más experimentos ni modifica los umbrales.
