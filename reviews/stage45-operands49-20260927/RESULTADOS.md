# Ronda49: continuación y entradas consumidas

Etapas4/5 abiertas. Se ejecutaron50ms de colaOFF por historia, sham y profile, desde los estados finales48. No se cambió la ley, el lector, los pesos ni el cuerpo; no se aplicó giro ni viento.

Se reconstruyeron exactamente 26736 sumas de las seis neuronas en las diez ventanas prefijadas. Diferencia máxima de la transferencia positiva frente al cálculo CPU: 1ULP, dentro del límite de2ULP documentado para tanhf; las sumas/márgenes y la paridad del organismo siguen siendo exactas.

| Cola ms | Historia | MargenDNgL | MargenDNgR | qDNb05L | qDNb05R | qDNa02L | qDNa02R |
|---|---|---:|---:|---:|---:|---:|---:|
|1|sham|-1483.488281|-1393.645508|0.721740445|0.745410803|4.62201492e-108|2.61854792e-322|
|1|profile|-1476.691406|-1383.170898|0.722043126|0.746623638|4.62201492e-108|2.61854792e-322|
|5|sham|-1460.264282|-1371.245728|0.721759470|0.745408456|3.88719183e-108|2.61854792e-322|
|5|profile|-1454.060669|-1361.395264|0.722063458|0.746619533|3.88719183e-108|2.61854792e-322|
|10|sham|-1433.816528|-1345.318115|0.721778708|0.745405757|3.13070222e-108|2.61854792e-322|
|10|profile|-1428.303223|-1336.206299|0.722082797|0.746612912|3.13070222e-108|2.61854792e-322|
|20|sham|-1389.741699|-1300.769287|0.721800189|0.745400534|2.03073538e-108|2.61854792e-322|
|20|profile|-1385.345947|-1292.878174|0.722081251|0.746573445|2.03073538e-108|2.61854792e-322|
|50|sham|-1335.794678|-1242.276367|0.721887160|0.745388222|5.54227488e-109|2.61854792e-322|
|50|profile|-1332.224731|-1235.022949|0.721841520|0.746163512|5.54227488e-109|2.61854792e-322|

Diferencia bilateralDNb05 (profile−sham) al primer y último ms: [-0.0009101545924298238, -0.0008209303898017994]. Es estado residual tras historias distintas, no aprendizaje probado ni señal de dirección del olor.

Propiocepción consumida idéntica bit a bit entre historias: {'angles_rad': True, 'angular_velocity_rad_s': True, 'normalized_afferent_drive': True}. Estado/traza corporal idénticos: {'qpos': True, 'qvel': True, 'position_mm': True, 'yaw_delta_deg': True, 'contact_force_N': True}. Velocidad articular máxima: 9.63145608e-06°/s. Este contraste no aporta una contingencia propioceptiva distinta para un replay causal.

DNg100 mantuvo objetivo0 en todas las muestras de las cinco ventanas por brazo. No hubo mando de avance adicional. DNa02 es sólo un observador, incluso si su estado cambia. No atribuir ninguna recuperación de navegación a esta colaOFF.

La comparación continua4ms frente a2+2ms conservó exactamente todos los propietarios serializados. El nuevo observador también conserva el final4ms del observador48. Es una cualificación local del apéndice49; no corrige retrospectivamente la etiqueta histórica restart_tested de48.

La revisión no autoral encontró tres huecos del verificador (identidadpresináptica, predictor/comprometido y ley positiva); se corrigieron offline y se detectaron once corrupciones deliberadas bajo-O. Se preservan los verificadores fallidos. No hubo cambios de tolerancia en decisiones científicas.

Alcance: un preparado y dos historias ya expuestas, no cohorte nueva, navegación ni equivalencia biológica. Cotejo de evidencias/falsadores en FUENTES_Y_DECISIONES.md.
