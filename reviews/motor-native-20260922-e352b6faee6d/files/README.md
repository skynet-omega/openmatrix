# OpenMatrix — núcleo nativo y comparación de métodos

Ronda acotada del 22-09-2026. La prioridad es ejecutar segundos completos con precisión comprobada antes de una campaña amplia de etapa 3. Resultados medidos, límites y autocrítica en `RESULTADOS.md`; reconstrucción numérica en `VERIFIED.json`. Los archivos originales y fallos se conservan.

## Arquitectura y alcance

El modelo declara poblaciones, estados, unidades, parámetros, ecuaciones, salidas y conexiones dispersas. El núcleo genera operadores y derivadas; no selecciona algoritmos por nombre anatómico. La carga grande combina una red recurrente, transductores de siete estados y química de doce especies. Es un ensayo sintético con acoplamiento continuo, no el cerebro ni la visión biológica completos.

A (`implicit.py`, `native_ops.py`, `native_codegen.py`, `native_support.cpp`, `bridge.cpp`) ejecuta callbacks C++ y grafos CUDA. Cada llamada actualiza un bloque GPU con punteros/tiempo/gamma; los kernels leen ese bloque al ejecutar. Un mismo stream ordena copia, cálculo y comprobación finita. Los estados permanecen en GPU; siguen existiendo sincronizaciones y transferencias pequeñas por callback. No se afirma que todo el control sea GPU residente. Los grafos no reutilizan respuestas neuronales anteriores.

B (`implicit_blocks.py`, `block_preconditioner.py`, `bridge_blocks.cpp`) conserva callbacks Python para aislar la alternativa matemática. Prepara Jacobianos por célula declarada, factoriza con pivote y reutiliza los factores al resolver. Las conexiones externas al bloque siguen en el residuo/JVP global; no se eliminan ecuaciones. Admite hasta 16 estados por célula en este prototipo.

C sigue sin implementar: una descomposición IMEX global declarada, justificada por rigidez y acoplamiento. No es el exponencial diagonal descartado antes. La referencia ERK8 (`reference_erk.py/cpp`) utiliza un método existente de SUNDIALS, sin Newton, y admite únicamente masa identidad. Fue incorporada como referencia de precisión, no como un tercer candidato completo ni como prueba de superioridad universal.

Se conservan dos perfiles: rápido (rtol 1e-3, atol 1e-5 por escala) y preciso (1e-7, 1e-9). Los límites externos son distintos: máximo de `abs(candidato-referencia)/(escala+abs(referencia))` <=0,01 / <=1e-5. Son métricas normalizadas en los estados inspeccionados; no porcentajes universales de error biológico ni garantías continuas. Todos los cálculos de esta ronda usan FP64.

SUNDIALS resuelve `M dx/dt=F(t,x)` con masa dispersa constante; el JVP incluye las conexiones. El implícito exige diagonal positiva y dominancia estricta por filas. No hay DAE, eventos de espiga, retardos, ruido, plasticidad de eventos o cuerpo acoplado en esta versión. Las sesiones permiten inspecciones y cambios por épocas. Un reinicio conserva el estado físico y comienza otra historia adaptativa; no es una serialización completa del integrador. Los ensayos son deterministas; no entrenan organismos ni producen checkpoints de aprendizaje.

## Reproducción desde extracción nueva

Se requiere Linux/WSL, CUDA 12 compatible con la GPU, CMake, GCC/G++12, Python con NumPy/SciPy/CuPy. `ENVIRONMENT.json` identifica el entorno utilizado. SUNDIALS 7.6.0 y su licencia se incluyen como fuentes de compilación, sin descarga ni instalación global. El generador actual usa compute89; otra GPU requiere una revisión explícita del destino de compilación y una nueva validación.

Desde el directorio recién extraído:

```bash
python -B -O verify_package.py
python -B build_dependency.py
python -B build_extras.py
python -B -O reproduce.py
```

La comprobación corta reconstruye resultados guardados, prueba contratos, operadores, bloques, sesiones y un caso de masa/clamp. Escribe resultados nuevos en `reproductions/`. Se ejecuta también con `-O`: las comprobaciones científicas no dependen de `assert`.

Para una campaña completa nueva, máximo 20 intentos y 3.600 segundos de integración/procesos:

```bash
python -B -O reproduce.py --full
```

La campaña completa usa otro directorio y conserva fallos/límites de tiempo. No se repitió entera para preparar esta entrega. Sus resultados nuevos no heredan el veredicto original. `campaign.py` y `finish_campaign.py` preservan los comandos históricos; el primero dejó continuar una referencia incompleta y produjo un error de contexto al comparar. `reproduce.py` separa esas corridas y usa la referencia completa antes de comparar.

`PLAN.json`, `FREEZE.json`, `BLOCKS_PLAN.json`, `REFERENCE_PLAN.json` registran criterios, fuentes y reasignación del presupuesto. `PARENT_DIFF.txt` y `parent/` preservan el control anterior. Los reinicios de pruebas marcados como corruptos son negativos deliberados, no puntos válidos para reanudar una simulación.

ChatGPT revisó el contrato de punteros/stream; Jev clasificó una tarea acotada. Sus respuestas no equivalen a ejecutar este código. No se usaron subagentes Codex.

Referencias de implementación: [SUNDIALS ARKODE](https://sundials.readthedocs.io/en/latest/arkode/Usage/User_callable.html), [grafos CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html). Las cabeceras incluidas de la versión fijada determinan la API efectivamente compilada.
