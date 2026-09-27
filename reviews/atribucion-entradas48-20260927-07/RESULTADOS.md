# Atribución de entradas — campaña 48

Arithmetic of six direct input rows at four final checkpoints. Not a recurrent causal attribution or historical whole-RHS replay.

Cálculo CPU desde cuatro estados finales de 3000 ms. No se avanzó el cerebro. Las etapas 4/5 continúan abiertas.

| Condición | DNg100 L: entrada + | Entrada − | Saldo | DNg100 R: saldo |
|---|---:|---:|---:|---:|
| sham | 6870.049 | -7696.363 | -826.314 | -840.282 |
| dm1 | 6883.740 | -7681.494 | -797.754 | -808.999 |
| profile | 6873.309 | -7692.673 | -819.364 | -829.648 |
| permuted | 6878.882 | -7693.742 | -814.860 | -825.180 |

Unidades internas del modelo. La suma no es una medición de corriente biológica. Los pesos guardados de estas filas son idénticos entre brazos; sus diferencias provienen algebraicamente de la transmisión presináptica. Esto no identifica qué mecanismo generó ese estado.

| Destino | q izquierdo sham → profile | q derecho sham → profile |
|---|---:|---:|
| DNg100 | 1.92686e-322 → 1.92686e-322 | 2.81617e-322 → 2.81617e-322 |
| DNb05 | 0.721735 → 0.722038 | 0.745411 → 0.746625 |
| DNa02 | 4.82648e-108 → 4.82648e-108 | 2.61855e-322 → 2.61855e-322 |

Hay actividad de DNb05, sin evidencia de orientación útil en este contraste. DNa02 sigue siendo un observador y no sustituye el lector.

Las seis filas suman 7227 aristas. Aristas entrantes desde ORN prescritas, en orden DNg100 L/R, DNb05 L/R y DNa02 L/R: [0, 0, 2, 16, 0, 0]. Ninguna recibe APL directamente; se conservaron los controles de rutas especializadas y salida PN general desactivada.

Los mayores aportes inhibidores por tipo hacia DNg100 incluyen GNG127 y GNG031. Ya aparecían en el preparado del 26-09; no son un mecanismo nuevo descubierto aquí. Su tamaño no autoriza a retirarlos: una lesión por ranking confundiría aporte actual con causalidad y podría compensar una calibración incorrecta.

Control algebraico adicional: se permite combinar independientemente los mínimos/máximos de cada entrada observados en los cuatro estados finales, conservando W/caps. Es una caja artificial, más permisiva que elegir un brazo real; no es una trayectoria ni un rango fisiológico. Para DNg100, los máximos márgenes con drive=0 son [-1417.518403, -1324.368102]. Resultado por fila: [True, True] (True significa que toda esta caja queda bajo umbral). No excluye estados fuera de estos rangos ni descarta el proyecto.

En RESULTADOS.json están todas las agrupaciones, diferencias, sumas FP32 y comparación separada con el último RHS registrado. No se declara reproducción del RHS completo ni continuación cualificada del checkpoint.

Verificación: `python3 verify_capsule.py` y `python3 -O verify_capsule.py`. Recalcular: `python3 analyze_endpoint.py`. La extracción completa usa `extract_endpoint.py` y requiere los originales fijados por hashes en EXTRACTION.json; esos estados completos no forman parte de esta cápsula.
