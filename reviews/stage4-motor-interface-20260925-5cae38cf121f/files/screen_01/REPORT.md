# Puerto motor: criba de una hipótesis de normalización

**DESCARTADO** en el alcance de este filtro. Parámetros calculados sólo con los primeros 200 ms del sham; ninguna búsqueda de ganancia ni aprendizaje con etiquetas olfativas. El sham evaluado usa los 200 ms siguientes. Las otras vidas ya estaban expuestas: no son confirmación ciega.

| Registro | Lector actual, integral ° | Puerto calibrado, integral ° | Reset puntual diagnóstico, integral ° |
|---|---:|---:|---:|
| sham | +0.009566 | +0.095724 | -0.036363 |
| uniform | +0.024753 | -0.381064 | -0.067493 |
| odor_left | +0.096491 | +1.384462 | +0.004242 |
| odor_right | -0.056393 | -1.497982 | -0.148373 |
| spatial_plus | +0.048194 | +0.611923 | -0.044114 |
| spatial_minus | -0.009552 | -0.807817 | -0.101754 |
| long_identity | -0.022796 | -1.663585 | -0.253216 |
| long_equal | +0.069465 | -0.227455 | -0.161127 |

Criterios: `{"heldout_rest_nonregression": false, "static_left_direction": true, "static_right_direction": true, "spatial_plus_direction": true, "spatial_minus_direction": true, "static_bilateral_separation": true, "spatial_bilateral_separation": true}`.

Son mandos recalculados con estados neuronales congelados; no movimientos corporales predichos ni resultados en lazo cerrado. Alterar el lector cambiaría la propiocepción de una nueva vida. El reset de baseline se muestra como sensibilidad y no se promociona. La normalización de q no lo convierte en frecuencia de disparo o calcio.

Etapas 4 y 5 abiertas. Cero organismos nuevos, cero cambios del motor vigente, sin retocar parámetros tras observar el resultado.
