# Revisión cruzada de motor y etapas 4/5

Se reconstruyó exactamente DN→lectura→filtro→relé sobre los datos guardados de las dos vidas de 2 s y la exploración de 12 s. Cero pasos nuevos del cerebro o del cuerpo; no se modificó campaña45 ni su cola activa. El alcance confirmado localmente es este análisis CPU y los contraejemplos del verificador. No se admite una etapa ni se confirma una arquitectura.

En intervalo **1911**, el filtrado estable es **0.0004995591588058994 rad/s** y el revisado **0.0005004683573549568 rad/s**. El umbral congelado es **0.0005 rad/s**. Una diferencia de **9.091985490574045e-07 rad/s** determina mando **0 frente a 5°/s**. El FAIL histórico permanece: identificar esta frontera no justifica cambiar sus tolerancias ni demuestra que el integrador cause el mal rumbo.

En los 12000 intervalos de la vida exploratoria, DNg100 L/R permanece en [1.93e-322, 2.8e-322], con rango de cambio [0.0, 0.0]; son valores subnormales, no prueba de inactividad biológica. Objetivo de avance único: [0.2] mm/s. Entre 6 y 12 s, yaw crudo medio **-0.478991646°/s**, aplicado único **[-5.0]°/s**. Hay 746 intervalos con mando no nulo y signo opuesto al crudo; la memoria temporal de 200 ms puede producirlo y esto por sí solo no constituye un bug. La magnitud del mando y la dinámica del relé requieren separarse del aporte neural al interpretar orientación.

El analizador congelado de45 acepta tres contraejemplos sintéticos:

1. Su criterio físico da True con desplazamiento y velocidad distintos del sham, aunque DNg100 y los mandos de ambos brazos sean idénticos y nulos. El criterio físico, por sí solo, no acredita propulsión por esa ruta. No se afirma que este caso sea realizable en la mecánica del runner.
2. Informa primer mando diferencial en1001 aun cuando no corresponde a DN_q_usada. El runner sí vigila el contrato en vivo; el analizador aislado no lo reconstruye. Bajo el lag declarado, primera DN diferencial publicada1001 sólo puede influir en mando1002.
3. Acepta PLAN modificado después de adquirir datos y cambia el veredicto. El suplemento rechaza el hash de criterio alterado; no propone otro umbral.

Estos son fallos de independencia/procedencia de la reconstrucción y una limitación de atribución del criterio físico, **no evidencia de que los datos reales de45 estén corruptos**. La prueba sintética y el código congelado están incluidos. El suplemento lector comprueba baseline fijo, lag de DN, decodificación, agenda, finitos y primer intervalo permitido; no modela ni verifica por sí solo fuerzas de la prótesis.

Tres explicaciones activas, con discriminadores:

| Alternativa | Dato que decide | Resultado que la refuta en ese alcance |
|---|---|---|
| A: falta de reclutamiento sensorial/descendente | Olor consumido→ORN/PN→DNg100/DNb05 frente al sham, por fase | DNg100 se modula y su efecto llega al lector |
| B: pérdida/amplificación en lector y relé | DN usada→crudo→filtrado→aplicado, sin recalibrar | Magnitud/signo útil conservados y sin efecto de frontera pertinente |
| C: apoyo/freno/feedback físico | Mando→fuerza/contactos→velocidad/desplazamiento | Respuesta física explicada por el mando sin conflicto de fuerzas |

Solicitud concreta a «Motor C++/CUDA»: conservar45 y terminarla dentro de su presupuesto; añadir después un informe CPU separado que contraste los hashes de PLAN/fuentes registrados, reconstruya la cadena del lector desde datos de base y reporte las diferencias ORN/PN/DNg100/DNb05/crudo/aplicado/cuerpo por fase. Extraer identidad/propietario y entradas/objetivo de DNg100 de estados guardados si el canal no se recluta; no cambiar sus parámetros para obtener respuesta. El pulso bilateral uniforme de45 estudia iniciación/modulación, no dirección o distancia. Para orientación sigue pendiente el contraste diferencial congelado de43, con presupuesto nuevo prospectivo medido en el motor escogido, preservando el bloqueo de44 y el FAIL de12. No repetir41/42 como pruebas nuevas ni integrar patas/plasticidad ahora.

Jev realizó una petición de clasificación: primero **relay_cpu_diagnosis** (confianza declarada 0.37); siguiente discriminador **decoder_direction** (0.56). Es prioridad sugerida, no verificación científica. Nuestra reconstrucción CPU respalda el diagnóstico del relé; la debilidad del analizador se apoya en pruebas, no en la clasificación de Jev.

ChatGPT Matrix recibió una revisión paralela del análisis y la conexión con etapas4/5. Consultar CHATGPT_REVIEW.md/CONTRASTE.md para respuesta y límites; no se atribuye ejecución de arrays a una lectura del texto. El selector PRO/máximo razonamiento no se comprobó desde esta herramienta. No se usaron subagentes Codex.

Verificación original: 0.075477s CPU, 0.049047s pared, RSS 115.902 MiB. Las cifras son del analizador CPU, no del motor neuronal. Fuentes/arrays seleccionados en INPUTS.json; no se incluyen estados completos del cerebro ni se reivindica reproducción de la vida neuronal.
