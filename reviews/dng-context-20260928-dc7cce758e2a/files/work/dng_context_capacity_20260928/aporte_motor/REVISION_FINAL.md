# Dictamen independiente: capacidad directa de grupos hacia DNg100

28-09-2026 · Motor C++/CUDA.

**La tabla se reproduce y no encontré un fallo que cambie su conclusión.** C1, C1_external y C2 no pueden excluirse por falta de capacidad directa bajo el dominio declarado; C3 carece de aristas directas desde sus 97 MBON anotadas. Ninguno de estos resultados demuestra reclutamiento realizable, inicio de movimiento ni superación de etapas 4/5.

Reconstruí desde los diez **NPZ originales sin empaquetar**, cotejados con sus identidades publicadas, y desde las anotaciones canónicas Parquet. No ejecuté el adaptador nuevo, su verificador ni su codec. Reimplementé productos y reducción del consumidor para las dos DNg100, sin repetir la auditoría de las seis sumas como resultado científico. Se compararon 80 filas de ventanas, 640 puntos alineados, 320 diferencias entre historias y las ocho filas agregadas. [Datos de la revisión y hashes completos](RECONSTRUCCION_FINAL.json).

Los netos, márgenes, cotas operacionales e incrementos FP32 coinciden exactamente. La contabilidad FP64, reconstruida agrupando sumandos de otra manera, difiere como máximo 3,64×10⁻¹² unidades internas; no altera ninguna decisión. Conteos, máscaras, dominio, identidades, selecciones y banderas de exclusión concuerdan. La selección congelada incluye C1_external y conserva C1; su carácter previo al cálculo consta en el recibo del autor, no en una prueba temporal independiente realizada por mí.

| Grupo | Destino | Aristas efectivas | Intervalo del margen máximo permitido en las muestras |
|---|---:|---:|---:|
| C1 descendentes | 10045 | 214 | +12.223,09 a +12.323,64 |
| C1 descendentes | 10056 | 205 | +11.781,47 a +11.889,21 |
| C1_external, sin las dos DNg100 | 10045 | 213 | +11.541,76 a +11.642,31 |
| C1_external, sin las dos DNg100 | 10056 | 204 | +11.060,63 a +11.168,36 |
| C2 ascendentes/sensoriales ascendentes | 10045 | 155 | +3.371,14 a +3.526,61 |
| C2 ascendentes/sensoriales ascendentes | 10056 | 170 | +2.836,67 a +2.997,76 |
| C3 MBON anotadas | 10045 | 0 | −1.489,57 a −1.331,92 |
| C3 MBON anotadas | 10056 | 0 | −1.399,47 a −1.234,83 |

Son unidades internas de entrada del modelo y rangos entre evaluaciones, no incertidumbre estadística. Quitar la entrada desde la otra DNg100 reduce la cota de C1, pero no cambia el resultado: la capacidad algebraica favorable no depende únicamente de esa conexión recíproca. Los aportes observados de C1_external siguen siendo negativos y coinciden con C1 en estas capturas.

**Contraste temporal reconstruido.** En el RHS0 con reloj absoluto 47.486.000.000 ns, primera época comprometida de la ventana 1 ms, `profile−sham` del aporte C1_external es +2,597341 para 10045 y +2,893385 para 10056. En 47.535.000.000 ns, comienzo de la ventana 50 ms, es +2,292920 y +2,699206. Los cambios del margen total respectivos son +6,949097/+10,634277 y +3,565308/+7,237549. El grupo contribuye a una diferencia de entrada entre historias, pero no explica por sí solo todo ese cambio. No se emparejaron índices de intentos adaptativos; las 40 marcas absolutas por grupo/destino tienen correspondencia entre historias.

## Tres límites del dictamen

1. **Capacidad no es reclutamiento.** La cota permite saturar las transmisiones de peso positivo y silenciar las negativas de todo el grupo, con el resto congelado. Una cota positiva no identifica un estímulo que lo consiga. La etiqueta mecánica `NEGATIVE` significa que no se sostiene la exclusión de todos los grupos primarios mediante esa cota; no significa fracaso del cálculo ni éxito locomotor. No justifica escoger células por su aporte y estimularlas para producir un PASS.

2. **El cero MBON está limitado por la selección y la cobertura.** Hay cero aristas desde las 97 células con `class=MBON`, incluso antes de aplicar las exclusiones del consumidor. Sin embargo, falta `class` en 1.121 de las 1.123 aferencias de 10045 y en 1.159 de las 1.166 de 10056. Las etiquetas de `superclass` sí están presentes. El cero del subconjunto conocido es verificable; no certifica ausencia de cualquier MBON no anotada ni de influencia por rutas indirectas. Los valores faltantes no deben reinterpretarse como clase conocida no MBON.

3. **La comparación pertinente conserva el aporte inicial y el mismo instante.** Ratifico rechazar `U/deficit`: U es la contribución máxima, no el incremento respecto de la contribución actual. La tabla calcula correctamente la fila completa modificada. Para presentar una capacidad disponible frente al déficit se debe usar el incremento operacional respecto del neto original y el déficit de esa misma evaluación; no dividir máximos o mínimos de instantes distintos. Aun esa razón sólo describiría esta caja algebraica. No transforma las diferencias entre historias expuestas en una intervención causal sobre el grupo.

**Decisión:** cerrar esta ronda como localización de capacidad y cobertura. La cota no respalda descartar todas las vías clásicas por capacidad insuficiente. Cualquier paso posterior debe identificar una forma de reclutamiento sustentada por evidencia independiente; esta tabla no selecciona por sí misma otra simulación ni una modificación de ley. No inicié trabajo adicional.

## Identidades y coste

Resultado revisado: `runs/workbench/units/595415c148f54b9c9df931f8d2e967860adcf657b8801b0681e8d31845ad0656/attempts/79b4acab55874440b18f1723035804f4`.

| Archivo | SHA-256 |
|---|---|
| CONTRACT.json | `73ed2638a78c348ff11b62231c2dfad6274849b8d3d317f4462a02f31145a8ec` |
| FROZEN.json | `5694803f6ba70dcad078de854f2747984acb5bedfd1428a8c9e3a90e324053d1` |
| src/dng_context_capacity.py | `2d3b4d80d676f0caa706909868b340f35d0a0648dc5cc5719f83370d8c3e46f2` |
| SELECTORS.json | `e23deec2435d593dbede0591fc0cdb50fd7e8862c9bfb3b4fcd1c6ca24d5978d` |
| groups.csv | `441eea8b58ea3fdb4d1654a489dc3a7a9df74bd4ed5680c1c286fb2bc9a7bcb6` |
| windows.csv | `f135e845683e15f36c3631c7fb4fa25038666c6a1cba3570e36a2e14cdfdce9c` |
| aligned.csv | `f0ae277dac44d936a01ede16575c77600b63c3d1b339046252bc1ec7666a2d00` |
| differences.csv | `baf1d5109fda0b0d324d20d81e5bbee3a2e7a3505881fad420fc6aaf6145fde2` |
| assessment.json | `84cc19b728297c50f67a8cd1b71f3160e463c5c13659b11f6d78c4583e4bdcb3` |

Esta reconstrucción consumió **2,1028504 s CPU** instrumentados; sumada a la revisión de diseño, **3,3713267 s CPU**, dentro de los 60 s compartidos. Las lecturas breves y la redacción no se cronometraron individualmente. Cero nuevos CNS, cero GPU, sin subagentes y sin modificar fuentes ni resultados del autor.
