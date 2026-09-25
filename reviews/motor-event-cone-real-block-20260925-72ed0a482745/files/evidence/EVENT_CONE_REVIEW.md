# Zona estructural de influencia de los siete eventos reales

Se usó el CSR completo no nulo del preparado idéntico al replay MRI (SHA de pesos de preparación `6d8d339e29e5229ad0089d5512fe4e31b4d2bddf1a61a79611ed23b9e5240cfd`) y el bloque aceptado sham de 125 µs, con los mismos siete eventos y relojes verificados entre corridas. Se midió **alcanzabilidad**, no corrientes ni causalidad dinámica. El grafo base tiene 24.559.135 aristas con peso no nulo de 25.582.938 almacenadas.

| Profundidad desde siete fuentes activas | Neuronas nuevas | Neuronas acumuladas | Aristas activas salientes de las acumuladas |
|---|---:|---:|---:|
| 1 salto | 1.263 | 1.270 (0,76 %) | 281.419 (1,15 %) |
| 2 saltos | 13.328 | 14.598 (8,76 %) | 3.669.879 (14,94 %) |
| 3 saltos | 88.188 | 102.786 (61,66 %) | 18.897.492 (76,95 %) |
| 4 saltos | 60.311 | 163.097 (97,84 %) | 24.522.782 (99,85 %) |

La coordenada de mayor error del MRI congelado, fila 29.460 `hDeltaA`, no recibe aristas directas de esas siete fuentes; está a **dos saltos** por las filas 12.940 `FB4O_L` y 131.606 `FB2B_b_R`. Tres rutas base no nulas observadas son 57.513→12.940→29.460 (`+0,03`, `−0,06`) y 51.040/60.830→131.606→29.460 (`+0,03`, `−0,03`). Estas rutas no prueban que sus flujos hayan causado el error: las conductancias, cancelaciones, propietarios especializados y tiempos importan.

La implicación algorítmica es acotada. Una región de un salto es barata pero puede omitir efectos que aparecen en el segundo; integrar sin selección toda la región de tres saltos trataría el 77 % de las aristas activas y difícilmente conservaría la ganancia por localidad. Un `F_fast` genérico prometedor podría incluir eventos y vecinos de primer salto, y seleccionar parte del segundo mediante cotas de influencia/defecto con expansión y retroceso versionados. Debe demostrar error y coste total sobre el operador efectivo, incluidos PN/KC/visión, antes de reemplazar el motor. Las rutas B/Krylov y C/QSS permanecen controles rivales.

El análisis prospectivo tomó 0,930 s bajo el presupuesto de 60 s y 4 GiB. Un verificador independiente por posiciones de aristas recomputó las cuatro capas, detectó corrupción de fuentes y pasó normal/`-O`. Su alcance sigue siendo el CSR base; las aristas efectivas de propietarios especializados pueden ampliar o alterar la zona.
