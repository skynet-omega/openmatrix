# Aporte del motor a51: aire antenal y contexto propioceptivo

27-09-2026. Entrega CPU acotada; cero pasos de cerebro/cuerpo y cero llamadas GPU. **Las etapas4/5 siguen abiertas.** Se entrega una interfaz ejecutada, no una nueva simulación del organismo. Archivos nuevos sólo en esta carpeta; matriz, núcleo y campañas anteriores permanecen intactos.

## Anatomía realmente comprobada

`anatomy_probe.py` leyó `male_v10/counts_pre_post.npz`, los IDs canónicos y las anotaciones. Verificó la orientación pre→post y las sumas salientes contra `node_connectivity.parquet`, sin umbral de poda. Hay166.700 neuronas y25.582.938 aristas. Son conteos anatómicos; no ganancias, actividad ni conexiones funcionalmente excitadoras demostradas.

| Fuentes JO | Izquierda | Derecha |
|---|---:|---:|
| JO-C |46|22|
| JO-E |157|110|
| Total |203|132|

**Usar `rootSide`: `somaSide` está vacío en estas335 fuentes.** Todas entran por el nervio antenal `AN`. En total alcanzan2.283 células mediante36.023 aristas y186.266 sinapsis. El buscador amplio AMMC/WED/APN/wind identifica476 destinos candidatos,57.359 sinapsis y2.515 aristas adicionales desde esos candidatos hacia WEDPN. Esa etiqueta incluye tipos heterogéneos: no son476 relés de viento confirmados.

| Destino anotado | bodyId | Sinapsis JO-C/E | Fracción de su entrada anatómica |
|---|---:|---:|---:|
| AMMC012 izquierda |11960|2037|42,821%|
| AMMC012 derecha |11702|1098|30,483%|
| AMMC013 izquierda |523640|1614|20,909%|
| WED203 izquierda |10371|1457|13,933%|
| DNb05 izquierda |10118|119|0,707%|
| DNb05 derecha |10065|30|0,170%|

No aparece un alias explícito APN2/APN3 en los campos inspeccionados. Sí hay correspondencias aPN1 y AMMC-A1. **No renombrar AMMC012/013 como APN2/3 por tener muchas sinapsis.** Las rutas anatómicas permiten probar entrada en las JO identificadas conservando la red, sin afirmar esa equivalencia entre nomenclaturas. Los CSV conservan todos los destinos, lados y tipos; `ANATOMY.json` incluye hashes de entrada y los alias encontrados.

## Única interfaz de aire entregada

`air_interface.py`: hipótesis de ingeniería explícita y sin estado oculto. Recibe velocidad del aire y del cuerpo en coordenadas del mundo, más la rotación cuerpo→mundo. Calcula `u = R.T @ (aire − cuerpo)` y proyecta en los ejes supuestos `(1,+1,0)/sqrt(2)` y `(1,−1,0)/sqrt(2)`. Ejes corporales: delante, izquierda, arriba. Proyección positiva activa C; negativa activa E. Se aplica `drive = 80*s/(s+100)` para `s=max(proyección firmada,0)` en mm/s.

Los100mm/s,80unidades internas y ejes a45° **son parámetros de ingeniería, no estimaciones fisiológicas**. La literatura apoya una distinción cualitativa C/E ante deflexiones opuestas; no calibra esta transformación velocidad→deflexión→entrada. No representa mecánica antenal, adaptación, componente vertical ni velocidad local debida a rotación y brazo de palanca. Tampoco demuestra que todos los subtipos C/E tengan la misma respuesta.

La contribución es aditiva en335puertos; no reemplaza actividad recurrente. Igual ley por receptor, sin compensar artificialmente los tamaños203/132. No recibe coordenadas de olor, objetivo, error de rumbo ni una compuerta externa olor×viento. Un pulso de torque corporal y un campo de aire sensorial son intervenciones diferentes. El integrador debe verificar orden de IDs, decidir la velocidad física del campo y conservar su reloj; este archivo no integra el CNS.

`test_air_interface.py` ejecutó18 comprobaciones en seis entradas, incluidas invariancia galileana, covariancia ante rotación, simetría por receptor, intercambio C/E, cero relativo, actualización aditiva que preserva otros puertos y rechazo de entradas inválidas. Resultados: `AIR_INTERFACE.json` y `air_interface_cases.npz`. **La aprobación de esta API no es evidencia de navegación ni recuperación.**

## Fujiwara: bloqueo concreto de la transferencia directa

La documentación disponible describe fases y posiciones2D de tres patas izquierdas; el receptor actual requiere seis ángulos interiores femur–tibia y seis velocidades angulares firmadas con geometría de bisagra3D. La fase de zancada no es el ángulo articular. El término AN de esos experimentos significa neurona ascendente, distinto del nervio antenal `entryNerve=AN`.

`proprioception_identifiability.py` ejecutó el método `encode` actual extraído por AST, sin importar el organismo. Dos geometrías con exactamente las mismas proyecciones2D **y longitudes de segmentos** dan ángulos90° y144,7356°. El receptor produce entradas distintas en cuatro puertos bajo las cuatro hipótesis de polaridad. Arrays y resultado en `proprioception_witnesses.npz` y `PROPRIOCEPTION.json`.

Esto prueba que esos observables no identifican el input del receptor3D actual. No prueba que sea imposible construir un modelo planar declarado ni usar las medidas para otro contraste. No rellenar las patas derechas con un trípode inferido presentado como observación. No se descargaron ni reprodujeron los archivos crudos de775,9MB y334,9MB. Se comprobaron las tres fuentes pequeñas ya locales contra sus hashes de adquisición. El receptor actual y sus escalas/polaridades tampoco quedan fisiológicamente validados por esta prueba.

## Fuentes primarias y entrega

- [Suver2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6533146/): integración bilateral del viento; rutas candidatas JO-C/E, APN y WPN; distinción pull/push. La nomenclatura fisiológica no suministra automáticamente un bodyId en male_v10.
- [Yorozu2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2755041/): poblaciones mecanosensoriales con respuestas diferentes a desplazamientos sostenidos y sonido.
- [Matsuo2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4023023/): organización funcional por zonas de proyección de Johnston.
- [Fujiwara, depósito original](https://zenodo.org/records/6365304): documentación y código del experimento de zancadas y células HS. Se usó el esquema declarado y archivos pequeños, no un replay biológico.

Los tres ejecutables consumieron aproximadamente1,884sCPU de proceso, incluyendo sus importaciones. Pico máximo registrado743,4MB. Lecturas auxiliares, edición y herramientas no están incluidas en esa suma instrumental. Cero descargas de archivos en esta contribución; consultas bibliográficas web separadas. Preparación iniciada18:53:08UTC, tope19:08:08UTC. `ENTREGA.json` registra cierre, hashes y costes medidos.

**Decisión para Matrix Astra:** aire tiene una implementación CPU concreta para evaluar como hipótesis de ingeniería, sin equivalencia fisiológica; contexto de zancada no admite el trasplante directo propuesto con estos observables. Seleccionar como máximo los dos prototipos previstos tras contrastar las otras familias. Ninguna de estas cribas permite marcar4/5 superadas ni autoriza retocar parámetros hasta obtener orientación.
