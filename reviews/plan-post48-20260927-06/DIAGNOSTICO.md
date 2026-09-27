# Diagnóstico de la ley consumida en48

Cero pasos neuronales nuevos. Extracción de ocho bloques:901–1000 y2901–3000ms en cuatro brazos. Verificación exacta de margen, escritor del objetivo/tasa y derivada; resultados numéricos generados desde datos extraídos. No es atribución causal por fuente.

| Brazo, ventana | ID | Neto medio | Umbral medio | Máximo neto+drive con umbral cero | Objetivo final máximo |
|---|---:|---:|---:|---:|---:|
| sham_00901_01000ms | 10045 | -805.702097 | 663.258545 | -678.277588 | 0 |
| sham_00901_01000ms | 10056 | -815.855614 | 559.186096 | -691.002747 | 0 |
| sham_02901_03000ms | 10045 | -1053.275276 | 663.258545 | -826.333618 | 0 |
| sham_02901_03000ms | 10056 | -1048.363637 | 559.186096 | -840.300781 | 0 |
| dm1_00901_01000ms | 10045 | -805.702097 | 663.258545 | -678.277588 | 0 |
| dm1_00901_01000ms | 10056 | -815.855614 | 559.186096 | -691.002747 | 0 |
| dm1_02901_03000ms | 10045 | -1026.771262 | 663.258545 | -797.787720 | 0 |
| dm1_02901_03000ms | 10056 | -1018.702022 | 559.186096 | -809.031372 | 0 |
| profile_00901_01000ms | 10045 | -805.702097 | 663.258545 | -678.277588 | 0 |
| profile_00901_01000ms | 10056 | -815.855614 | 559.186096 | -691.002747 | 0 |
| profile_02901_03000ms | 10045 | -1046.718049 | 663.258545 | -819.379150 | 0 |
| profile_02901_03000ms | 10056 | -1038.020253 | 559.186096 | -829.661804 | 0 |
| permuted_00901_01000ms | 10045 | -805.702097 | 663.258545 | -678.277588 | 0 |
| permuted_00901_01000ms | 10056 | -815.855614 | 559.186096 | -691.002747 | 0 |
| permuted_02901_03000ms | 10045 | -1041.543332 | 663.258545 | -814.870605 | 0 |
| permuted_02901_03000ms | 10056 | -1033.189672 | 559.186096 | -825.189758 | 0 |

El objetivo rectificado es exactamente cero en todas las muestras seleccionadas. El estado conserva residuos subnormales; ver DIAGNOSTIC_NOTES.md. Con las entradas guardadas, cualquier umbral no negativo mantiene la rectificación a cero porque neto+drive<0. Esto descarta bajar sólo un umbral positivo a cero como rescate sobre estas entradas, sin simular una trayectoria intervenida ni identificar la causa biológica.

La discrepancia máxima entre neto y suma de auxiliares positivo/negativo es 0.0020751953125 unidades internas. Se informa sin renombrarla como identidad exacta: las reducciones FP32 usan otro orden. No alcanza para cambiar el signo en estas ventanas.
