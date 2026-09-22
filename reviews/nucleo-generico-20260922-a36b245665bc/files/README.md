# OpenMatrix: primer núcleo declarativo

Ronda terminada: [resultados reconstruidos](RESULTADOS.md). Objetivo: separar modelos científicos, compilación y ejecución,
sin seleccionar integradores por nombre anatómico. No sustituye aún el organismo.

El contrato prospectivo está en PLAN.json. A es acoplamiento global; B conserva
multirrate con corrección de ondas y C acciones exponenciales como rivales para
rigidez. Esta ronda implementa la interfaz común y la primera ejecución de A,
con referencia SciPy y backend GPU explícito residente. No afirma haber resuelto
el acelerador implícito general que falta para masas y canales rígidos.

Los perfiles `fast` y `precise` conservan ecuaciones, conectividad y FP64. Cambian
el control local de paso. El límite global muestreado se evalúa aparte, con
escalas declaradas: 0,01 y 0,00001, respectivamente. No es un porcentaje universal
de error biológico ni una relajación de contratos históricos de etapa 3.

Dominio inicial: ecuaciones diferenciales deterministas suaves, puertos graduados,
masa diagonal positiva en GPU y masa constante invertible en la referencia CPU.
Cambios externos programados se ejecutan en fronteras temporales exactas. Ruido,
DAE, retardos y espigas con reinicio requieren implementaciones adicionales;
deben rechazarse explícitamente. `precise` significa precisión numérica, no
cobertura completa del cerebro.

La intervención de clamp fija un estado y anula su derivada; cambiar pesos
modifica conexiones. La eliminación retira los estados y las conexiones indicadas
por el descriptor completo. La recompilación valida el modelo nuevo antes de
cambiar el estado vivo. El escáner devuelve estados con identidad, unidades y tiempo.

Referencias de ingeniería: [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[Arbor: compilación y matrices](https://docs.arbor-sim.org/en/latest/dev/matrix_solver.html),
[GeNN: modelos personalizados](https://genn-team.github.io/genn/documentation/5/custom_models.html).
Se reutilizan bibliotecas numéricas existentes; no se reivindica una nueva familia
de integradores ni validación biológica.

Segunda implementación prospectiva de la ronda: C, punto medio exponencial con
separación diagonal local. El compilador deriva automáticamente D=∂f_i/∂x_i
manteniendo puertos fijos; no escribe una fórmula especial para cada célula.
Se calcula z=x+h/2·φ1(hD(x)/2)f(x), y después
x_nuevo=x+h·φ1(hD(z))[f(z)+D(z)(x−z)]. El término de corrección es necesario;
exp(hD)f(z) por sí solo no sería punto medio de orden dos. Se usa duplicación de
paso (divisor3) y una referencia independiente. Es exacto para una ecuación afín
escalar constante, pero no resuelve exactamente la rigidez fuera de la diagonal.
Esto pone a prueba una parte concreta de C, no implementa Krylov/racional general.
A/B/C conservan su papel: A acoplamiento global, B multirrate aún no implementado,
C separación exponencial. No se selecciona C antes de medir.

Uso inicial (desde esta carpeta, con Python NumPy/SciPy/CuPy-CUDA):

```bash
export OPENBLAS_NUM_THREADS=1
python -B -O cli.py --model examples/mixed.json --mode fast --until 0.1 --edits examples/schedule.json --scan cells:v --out demo_nueva
python -B -O cli.py --resume demo_nueva/final_checkpoint --until 0.2 --scan cells:v --out continuacion_nueva
```

El ejemplo fija dos voltajes a los20ms, conserva un checkpoint anterior a la
intervención y reanuda todos los estados y el controlador. Modificar o eliminar
material es una intervención externa que puede inyectar/retirar cantidades;
no se reivindica conservación a través de esa cirugía. El núcleo conserva estados
por identidad, no inventa un balance químico al retirar una especie. Las unidades
son verificadas dimensionalmente, pero todavía no hay conversión automática de
mV aV. La compilación de puertos admite salidas dependientes de estados/parámetros
pero rechaza bucles algebraicos instantáneos entre salidas y entradas.

El backendCPU de referencia admite masas constantes no diagonales hasta4096estados.
Ese tope es de esta referencia, no un argumento sobre el límite físico del motor.
El backendGPU todavía las rechaza. Faltan un integrador implícito disperso general,
colas de espigas/retardos, ruido, adaptación estructural endógena y el cuerpo.

Siguiente ruta matemática: A implícito con residuo M·Y−b−hγf(Y), derivadas que
incluyan todas las conexiones y Newton–GMRES con precondicionamiento estructural.
La documentación confirma la pareja ESDIRK3(2)5L[2]SA y la interfaz con Ginkgo:
[tablas de ARKODE](https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html),
[solvers Ginkgo–SUNDIALS](https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_links.html).
No se ha instalado ni integrado ese backend. Se mantiene B como alternativa
multirrate condicionada al acoplamiento; C diagonal ya tiene un negativo medido.
