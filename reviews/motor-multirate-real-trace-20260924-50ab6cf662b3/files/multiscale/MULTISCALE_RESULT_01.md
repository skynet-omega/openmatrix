# Operador por bloques: primer discriminador sobre el conectoma real

**Resultado de la prueba fijada antes de cargar el organismo:** negativo para la
partición de 16×16 intervalos de índice y su aproximación de rango 8/32. Una
carga de MaleCNS, cero milisegundos nuevos de simulación, 13,10 s de pared;
ningún benchmark de GPU. El recibo completo es
[`multiscale_probe_01/RESULT.json`](multiscale_probe_01/RESULT.json), con hashes
del CSR, pesos efectivos, máscara visual, capacidades, escala, estado inicial,
snapshot final, fuente y plan.

| Bloque de salida / operador | Aristas efectivas | Rango | Energía de operador no capturada | Bytes de factores / CSR | Error máximo observado en entrada, estado inicial | Cota analítica por fila para ese estado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No visual, W·caps | 988.258 | 8 | 86,52 % | 0,094 | 17.907,97 | 5,21×10⁶ |
| No visual, W·caps | 988.258 | 32 | 75,10 % | 0,375 | 12.418,28 | 4,86×10⁶ |
| Visual, excitación | 76.091 | 8 | 54,23 % | 0,956 | 1,3682 | 194,27 |
| Visual, excitación | 76.091 | 32 | 25,05 % | 3,822 | 0,7030 | 132,02 |
| Visual, inhibición | 113.865 | 8 | 46,62 % | 0,642 | 1,6880 | 291,15 |
| Visual, inhibición | 113.865 | 32 | 25,42 % | 2,567 | 1,1592 | 215,00 |

El segundo estado real, tras 1 ms sham, dio prácticamente las mismas cotas y
errores; los valores íntegros se registraron por estado. El umbral predeclarado
para un cribado positivo era **≤50 % de bytes CSR y cota por fila ≤10⁻⁶** en
ambos estados. Ninguna combinación lo cumplió. Los dos bloques elegidos sólo
contienen el 5,10 % de las aristas del grafo completo; un resultado positivo
aquí tampoco habría demostrado rendimiento global.

La cota procede de `||(A−QQᵀA)x||₂ ≤ ||A−QQᵀA||_F ||x||₂`, por lo que
también cubre el valor absoluto de cada fila **en aritmética exacta**. El código
calcula la norma de Frobenius de la proyección a partir de la identidad
pitagórica y añade una pequeña holgura FP64; esto **no es una prueba de
intervalos de redondeo**. El fracaso supera el umbral por muchos órdenes de
magnitud y no depende de interpretar esa holgura como un certificado formal.
La corriente no visual se forma con `W_ij*caps_j`; las filas visuales separan
conductancias positivas y negativas para preservar las dos entradas de su ley
no lineal. Los factores se deben recalcular si cambian pesos,
`caps`, máscara visual o escala. No se ensayaron aprendizaje/plasticidad,
propagación recurrente, eventos, cuerpo ni tiempos de núcleo.

Una comprobación separada de la construcción de los dos bloques, con una nueva
carga y cero simulación (11,39 s), confirmó exactamente todos los punteros de
fila, índices de columna y valores originales tras el corte CSR, incluidos los
1.303.644 elementos seleccionados. Así queda justificada la alineación con
`weights64` usada en el cribado. Recibo:
[`multiscale_verify_mapping_01/RESULT.json`](multiscale_verify_mapping_01/RESULT.json).

**Decisión:** retirar esta variante concreta de bajo rango como atajo de
velocidad próximo. No se ha medido una partición anatómica o aprendida, un
esquema jerárquico, una representación bajo rango más residuo disperso ni un
compresor adaptado a estados; estos siguen siendo hipótesis distintas que
necesitan una puerta de cobertura de todo el grafo, coste de actualización y
error acotado por estado/evento. La baja dimensión global o una coincidencia
conductual puntual no bastarían para afirmar compresibilidad por bloques.
