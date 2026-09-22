# PN conjunto: siguiente ronda A/B/C

La respuesta final de ChatGPT confirma lectura de README/VERDICT del commit78391c0; no ejecutó el ZIP. No contiene una nueva propuesta validada. Se conserva el cierre ABC previo y no se rescata C incremental cambiando parámetros.

Limitación medida: PN consume≈0,652s por1ms de un perfil corto del organismo, con178838coordenadas,2089nodosNa/K y5702nodosCa. G tiene537608entradas y M180397. Cero filas de M son nulas: eliminar nodos como si fueran algebraicos sin masa alteraría el modelo y se descarta. El objetivo permanece1s completo en≤300s, luego≤60s.

Tres alternativas de esta ronda, con el original como control:

- A: condensación exacta de las ecuaciones lineales pasivas **dentro de cada etapa implícita**, incluyendo G+shift*M. Se mantienen como incógnitas de Newton los nodos con canales/receptores y las uniones necesarias. El RHS y las variables eliminadas se reconstruyen: no se ponen derivadas a cero. Factores dependen de anatomía y paso; nunca se reutiliza una respuesta neural. Predicción: menos trabajo por iteración de Newton sin modificar voltajes/cargas/gates aceptados. Falsador: residuo completo no conserva el límite original o coste no baja materialmente.
- B: eliminación eléctrica paralela en GPU. Medir primero dependencias reales y coste de resolver sistemas capturados antes de portar toda la cinética. Falsador: dependencia/transferencias o falta de precisión impide ahorrar≥2×en el tramo PN; no se infiere inviabilidad de todaGPU a partir de un kernel secuencial.
- C: ejecución compilada del paso PN completo enCPU, manteniendo todas las coordenadas y etapas SDIRK. Diferencia frenteA: conserva el problema completo y elimina organización enPython; frenteB: ejecuta enCPU. Falsador: coste residual dominado por tráfico/factorización y ganancia insuficiente bajo el mismo error.

Información legal: matrices, canales, calcio, estado y etapas presampleadas de una captura real emparejada. No hay aprendizaje, etiquetas ni ayuda conductual. Ingeniería de Schur/SDIRK/compilación, sin reivindicación de novedad biológica. Referencia misma preparación, métodos y tolerancias; medir avance PN completo y luego organismo, separando arranque/exportación. No confundir un solve lineal con paso PN.

Presupuesto previo:4h de trabajo computacional,8GiB RSS porproceso,12GiB VRAM,1captura de≤1ms del organismo,≤3variantes lineales acotadas,como máximo2prototiposPN completos y8prefijos integrados de≤5ms/≤300s cadauno. Referencia base de1s:dos refinamientos(h125/62,5us),≤300s cadauno,≤120muestras completas. Pruebas/correcciones de defectos se registran aparte sin nuevos barridos de parámetros. No ajustar fisiología ni tolerancias para producirPASS.
