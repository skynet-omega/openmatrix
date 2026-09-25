# Revisión acotada del pipeline vigente

Especialista `/root/stage37_structural_review`, revisión estática y de registros existentes; sin modificar fuentes ni ejecutar organismos. Síntesis de los hallazgos contrastados localmente. No es auditoría externa independiente del proyecto.

- `46__src__matrix_olfactory_diagnostic.py`, líneas 151–156 de las fuentes ejecutadas en campaña36: captura baseline DNb05 durante preparación de candidata, antes de avanzar los 40 ms. `run_replay.py` 272–301 no lo reinicia. El donante histórico tenía un reset posterior fuera de la función extraída. La diferencia no convierte el contrato actual en un error de implementación. La criba37 registra sensibilidad al reset y sus regresiones.
- La diferencia entre contrastes DNb05 de identidad y antenas igualadas es negativa en 999 de 1000 muestras consumidas, positiva en cero. La oscilación del mando total no demuestra inversión de la contribución sensorial.
- En las 1000 muestras archivadas del último kernel aceptado de cada milisegundo, DNa02 tiene target cero. Los máximos de entrada neta fueron 284,68 frente a umbral 641,87 en la derecha y 48,81 frente a 548,00 en la izquierda. No es captura de todos los subpasos ni justificación para bajar umbrales.
- Los canales de avance son DNg100, IDs10045L/10056R; q constante subnormal (~1,93e−322 y 2,8e−322). Su contribución desaparece en la resta del baseline. El avance de 0,2 mm/s procede del término tónico. El atlas confirma identidad, no calibra sus parámetros.
- El lector consume q adimensional. `hybrid_visual_brain.py` publica q*r_max como tasas del modelo en float32. No confundir esas dos unidades ni ninguna con calcio. El controlador de rodillos convierte mm/s a m/s y grados a radianes correctamente: no se encontró un factor1000 perdido.
- Cinco observables preparados (DN actual/usado/baseline, qpos, sensores) de los brazos stage3 coincidieron con campaña36; frente a referencia27 hubo diferencias DN ≤2,89e−10. Este chequeo parcial no se presenta como igualdad de los 588 arrays.

El panel anatómico posterior de37 confirmó valores iniciales/finales publicados para22 células preseleccionadas:17 ceros en ambos finales,5 activas. Está preservado en `population_01`. No demuestra silencio continuo ni decide que un tipo activo sea el lector correcto.
