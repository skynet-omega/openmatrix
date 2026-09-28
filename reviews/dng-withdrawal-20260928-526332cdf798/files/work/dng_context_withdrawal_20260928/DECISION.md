# Decisión después de las retiradas directas

28-09-2026. Mantener etapas 4/5 abiertas. El resultado nuevo descarta que la recepción negativa descendente o ascendente retirada por separado sea la única barrera directa en las evaluaciones guardadas. La prioridad científica es identificar la escala de entrada y el umbral de DNg100; no seguir modificando grupos o ganancias hasta encontrar movimiento.

## Tres alternativas causales conservadas

| Alternativa | Evidencia útil y limitación | Discriminador y falsador | Estado |
|---|---|---|---|
| B1: signo funcional y receptor | Anotaciones MaleCNS locales; procedencia explícita ACh positiva, GABA/Glu/histamina negativas, otras cero. Presinapsis y receptor no son equivalentes. | Cotejar todas las entradas de las dos DN con la hipótesis usada; una contradicción obliga a reparar procedencia. Para cambiar el signo biológico hace falta evidencia independiente del receptor o intervención identificada. | Cotejo ejecutado: 0 contradicciones/2.289 aristas. Consistencia interna; signo fisiológico aún no identificado. No repetir descarga de anotaciones. |
| B2: liberación y filtro de transmisión | Filtro compartido de 5 ms prestado de un modelo; datos PN→LHN ya analizados localmente miden otro circuito. | Entrada y respuesta postsináptica emparejadas e identificadas, con reloj y unidades. Una ley que no prediga condiciones reservadas se descarta. Sin esa identidad no se traslada una cinética PN→LHN a DNg100. | No se identificó en los recursos consultados un registro específico de transferencia hacia DNg100. Es un hueco de esta búsqueda, no prueba de inexistencia universal. |
| B3: ley receptora/intrínseca y escala anatómica | theta/gain genéricos escalados por volumen/mediana CNS; el propio recibo declara transferencia no validada. Hay actividad de calcio BDN2 y conducta, sin conversión a corriente o target del modelo. | Auditar primero correspondencia de magnitudes y preparado; una ley alternativa requiere observables independientes que distingan sus predicciones. Calcio puede refutar actividad/relación temporal compatibles con el protocolo, pero no fijar theta o gain por sí solo. | Procedencia revisada; prioridad siguiente. No se ha demostrado un bug neuronal ni una ley sustituta válida. |

La pregunta inmediata para B3 es qué observación independiente restringe `net + drive − theta`, no qué gain produce avance. Una alternativa de ingeniería podría conservarse como provisional si mejora controles funcionales, con presupuesto y contrato nuevos; eso no la convierte en fisiología validada. No se presupone que el proyecto sea imposible por la identificación incompleta.

## Fuentes recuperadas que cambian la decisión

- `src/anatomical_rate_brain.py`, `src/anatomical_morphometry.py`, `src/hybrid_visual_brain.py`, `src/synaptic_visual_brain.py` y recibo49: definen la procedencia concreta de signos, escala, theta, gain y filtro. Los parámetros consumidos de las dos DN están en `PARAMETER_PROVENANCE_CHECK.json`; el extracto del recibo conserva fuente y hash. Un volumen real no valida automáticamente su transformación en excitabilidad.
- `work/stage3_brief_orn_control_20260917/README.md` y `work/stage3_synapse_comparison_20260917/README.md`: ya existe análisis de 10 pares PN→LHN y 3.646 ventanas válidas, con dependencia del destino. Reutilizar ese conocimiento evita repetirlo o presentar sus parámetros como una calibración de DNg100.
- [Sapkal et al., Nature 2024](https://www.nature.com/articles/s41586-024-07854-7): actividad BDN2 registrada con calcio GCaMP6f/tdTomato a 6 Hz en hembras de 4–7 días sobre bola; relación con avance. Permite contrastar actividad y comportamiento en ese preparado, no inferir directamente corriente, voltaje, una curva F–I ni el umbral interno de nuestro modelo. [Datos asociados](https://doi.org/10.17617/3.OIX8RZ) y [código del estudio](https://github.com/bidaye-lab/Sapkal_et_al_2024). Se consultó la fuente primaria, sin descargar toda la colección.
- El archivo local `hook_flexion_03_bdn2.parquet` mide aferentes propioceptivos durante activación de DNg100; no es una medición de recepción cerebral de DNg100. No sustituye el dato B2.

## Siguiente unidad acotada y condición de parada

Antes de otro prototipo completo, recuperar sólo el subconjunto BDN2 del estudio citado y verificar identidad, unidades, sincronización y protocolo frente a las preparaciones actuales. Primer límite: 20 minutos de inspección de metadatos, 25 MiB de transferencia y 120 s CPU; ningún CNS/GPU. Si no se puede aislar un subconjunto pertinente dentro de ese límite, registrar el hueco y detener la adquisición, sin descargar por precaución.

El producto es una restricción medible y una condición reservada que pueda refutar B3, o una declaración precisa de no identificabilidad. No ajustar contra la misma traza evaluada. No inferir q ni theta a partir de fluorescencia sin un modelo de observación respaldado. Una nueva simulación exige una predicción distinta del mero clamp directo, duración/budget y controles fijados antes de mirar el resultado.

Se conserva C1/C2 como contextos indirectos posibles, pero su retirada directa aislada ya fue contrastada. C3 anotada directa sigue sin aristas en estas dos DN; la falta de anotación de clase impide convertirlo en ausencia general de rutas MBON. Patas y músculos continúan posteriores al mando y la marcha estables por separado.

## Eficiencia del laboratorio

La revisión paralela no justifica reconstruir el sistema: recuperar contexto ~0,95 s, búsqueda ~0,04 s, reutilización y registro ~0,30 s. Se incorporan dos reparaciones causales y sus regresiones. La invalidación demasiado amplia de caché queda documentada; corregirla requerirá dependencias por componente, no retirar comprobaciones de identidad. No bloquea esta decisión ni autoriza otra ronda de microoptimización.
