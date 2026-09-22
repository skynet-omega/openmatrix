# Ronda PN A/B/C — resultados medidos

El motor completo aún no cumple1segundo simulado en60segundos reales. El paso PN se conserva, pero ninguna ruta completa supera el umbral2x en este prefijo.

|Ruta|PasoPN completo: ganancia|Error máximo voltaje(mV)|Veredicto|
|---|---:|---:|---|
|A|0.8510x|7.11e-15|DESCARTADO_PARA_PROMOCION_PN|
|C|0.9571x|8.88e-16|DESCARTADO_PARA_PROMOCION_PN|
|B|1.2901x|7.11e-15|DESCARTADO_PARA_PROMOCION_PN|

A condensó pasivos por etapa sin omitir masa. C compiló Newton/cinética eléctricos enCPU, conservando validación y química externas. B integró sólo el solverGPU en el paso PN original, no escribió un tercer integradorPN. Todos usaron64entradas reales prescritas de1ms, no realimentación corporal duranteelreplay. A/C se descartan como reformas de velocidad; B no se promueve como motorPN completo.

La piezaB paralela reduce451niveles conservadores a20rondas, con acumulación determinista porvecino y núcleo42 enmemoria compartida. Sondeo corregido: 3.84–4.77x en dos solves reales; 0.416ms residentes. No extrapolar esa ganancia alpasoPN: observado1.29x. Clasificación de la pieza lineal: PROMETEDOR_NO_CONFIRMADO para integración residente.

Verificación: LU dispersa independiente, residuo completo reconstruido,20repeticiones GPU idénticas porcaso, NaN/pivotesnegativos/diagonal fueradelsoporte rechazados; rechazo de segundaetapa sincommit parcial A/C. Verificador ejecutado conPython-O y tres corrupciones detectadas. Revisión local corrigió unabarreraGPU ausente; medición previa preservada como no promovible. ChatGPT identificó precondición diagonal no comprobada enSchur; reparada y probada. No ejecutó códigoexternamente. Jev hizo una llamada real de clasificación(1037tokensentrada,218salida); no decide fidelidad.

Autocrítica: el plan permitióprobarC antesdeterminar elcosteB completo. Las mejoras lineales no predecían una aceleración suficiente delPN. También hubo un benchmark adicional de refactorización delnúcleoB fuera de los3sondeos iniciales; está declarado enB_IMPLEMENTATION.md, sin ocultarlo como bug ni cambiar el umbral. Las repeticiones por defectos reales se conservan separadas. No continuar esta ronda por ajustes de tolerancia. Siguiente decisión: estados/cinética/residuos residentes enGPU, con alternativas de partición y organización temporal.

Contexto separado: bloquebase166700neuronas,1s en20.606s; nueva comparación muestreada106veces da error6.824782e-05, refinamiento9.858171e-06; pasa límites1e-4/1e-5 con margen estrecho de referencia. No es cota de error verdadero. Datosbase están fuera del ZIP PN; no se atribuye su reproducción alverificadorPN.

Anteriores metodológicos: [eliminación neuronal por árboles](https://www.neuron.yale.edu/neuron/static/papers/nc97/nc3p3.htm), [Hines enGPU](https://arxiv.org/abs/1810.12742). Son antecedentes, no mediciones de estehardware ni novedadbiológica.

Reproducción desde ZIP PNextraído, conNumPy/SciPy/Numba/CuPyCUDA instalados:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I -B -O verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I -B validate.py nueva_validacion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I -B compare_backend.py --route B --out nueva_integracion
```
Las ejecuciones usan carpetas nuevas; no sobreescriben evidencia. El segundo comando ejecuta la validaciónGPU/CPU; el tercero repite elPN conentrada prescrita1ms. No reconstruyen la mosca completa.
