# Motor para cerebros completos: corrección de alcance

La prioridad del usuario es un motor que admita nuevas áreas y el reemplazo de prótesis sin reescribir su núcleo. Los experimentos PN son pruebas de costes y métodos; no constituyen ese motor. No se promueve el prototipo PN residente como arquitectura general.

La limitación quedó demostrada, no sólo planteada: el solver actual acepta una cadena de128 coordenadas, pero rechaza una malla16×16 por su núcleo de252 coordenadas, y falla al construir un grafo completamente conectado de96. Sus canales NaT/NaP/K están escritos directamente; el kernel deCa admite como máximo4gates. Son decisiones de un prototipo local que no deben convertirse en requisitos biológicos. [Datos](INVENTORY.json).

La captura de166700neuronas contiene105269entradas con la bandera lógica `visual`. Esta cifra describe el modelo descargado, no una medición anatómica nueva ni cobertura biofísica de la visión. Contar esas neuronas dentro del bloque base no significa haber simulado todos sus mecanismos visuales.

## Representación matemática común

Propuesta inicial a contrastar, no motor implementado:

\[
M(x,\theta)\dot x=F(x,s,u,\theta),\qquad
\dot s=H(x,s,\theta),\qquad
x(t_e^+)=R_e(x(t_e^-),s(t_e^-)).
\]

`x` reúne voltajes, compuertas, concentraciones, recursos y estados de transducción. `s` incluye estados sinápticos y memorias físicas de filtros; no es una memoria cognitiva externa. La conectividad y los retardos relacionan puertos declarados. `u` procede del entorno y del cuerpo a tiempo causal. Cada variable declara unidad, dominio y tolerancia; cada mecanismo declara lecturas, escrituras, ecuaciones, eventos e invariantes.

El modelo describe **qué ecuaciones existen**. Un compilador construye sus operaciones y derivadas numéricas. El motor administra memoria, dependencias, integración, eventos, errores y confirmación atómica del estado. El nombre de un área puede aparecer en informes, pero no seleccionar una rama especial del integrador. Una especialización por estructura matemática comprobada es admisible; una excepción escrita para PN o retina no demuestra extensibilidad.

La primera familia soportada debe declararse con precisión: ODE con masa constante definida positiva y eventos explícitos. Masa singular, dependiente del estado, ruido y plasticidad estructural requieren capacidades y pruebas específicas antes de anunciar soporte. Escribir una ecuación general no implementa esas capacidades. Un problema fuera de la familia debe rechazarse con una razón clara, sin aproximarlo silenciosamente.

El control de error debe incluir todos los estados relevantes y la interfaz con el cuerpo, con escalas por unidad. Además del error temporal, hay que controlar residuo lineal/no lineal, conservación, límites de probabilidad, tiempos de eventos y error del intercambio entre bloques. El mismo presupuesto se usa en las tres alternativas.

## Tres arquitecturas rivales

|Ruta|Cambio sustancial|Ventaja que debe demostrar|Falsador|
|---|---|---|---|
|A — compilación global residente|Compilar la descripción completa a operaciones GPU y un integrador implícito/IMEX común, con operadores dispersos y derivadas generadas.|Añadir mecanismos sin tocar el planificador; pocos intercambios conCPU; error común sobre la red.|Un mecanismo nuevo exige modificar el núcleo, o la rigidez/coste del sistema global impide alcanzar el presupuesto.|
|B — integración por particiones y formas de onda|Particionar por dependencias y escalas matemáticas; intercambiar trayectorias y corregirlas hasta que el error del acoplamiento esté dentro del presupuesto.|Avanzar componentes lentos con menos trabajo sin congelar la comunicación continua.|La realimentación fuerte exige tantas correcciones o pasos pequeños que elimina el ahorro; el residuo local oculta error global.|
|C — evolución lineal por funciones de matrices|Separar una parte lineal verificada y avanzar su acción mediante métodos exponenciales/racionales; tratar la parte no lineal y los eventos con un método común.|Reducir el número de pasos impuestos por rigidez pasiva sin eliminar modos físicos.|El cambio frecuente del operador, la dimensión de Krylov o los eventos cuestan más que la referencia; no se controla el error conjunto.|

Son alternativas numéricas sobre **la misma descripción del modelo**, no tres biologías ni una mezcla por intuición. Pueden necesitar familias de kernels y solvers: lo unificado es el contrato de ecuaciones, estado y tiempo. Extensibilidad no garantiza una velocidad constante al añadir cualquier cantidad de detalle.

## Contrastes ya investigados

Arbor separa recetas, células, mecanismos y ejecución; su compilador obtiene la derivada del corriente respecto al voltaje y su solverGPU aprovecha la estructura de ramas. Esto ofrece un referente para generación y disposición de memoria, pero su método documentado no es automáticamente equivalente a nuestroSDIRK ni a una masa generalizada. [Documentación de matrices](https://docs.arbor-sim.org/en/latest/dev/matrix_solver.html), [métodos](https://docs.arbor-sim.org/en/latest/dev/numerics.html).

GeNN genera código para modelos neuronales y sinápticos personalizables. Es un referente real de extensibilidad, no prueba de que nuestras ecuaciones rígidas y acoplamientos graduales vayan a ser eficientes sin trabajo adicional. [Modelos personalizados](https://genn-team.github.io/genn/documentation/5/custom_models.html).

CoreNEURON separa una ejecución optimizada de modelos NEURON y emplea disposición por variables y soporteGPU, con restricciones de compatibilidad explícitas. Debe ser un comparador de ingeniería, sin copiar sus cifras de otras máquinas. [Descripción](https://www.neuronsimulator.org/en/latest/coreneuron/index.html), [compatibilidad](https://www.neuronsimulator.org/en/latest/coreneuron/compatibility.html).

Para B, existe una aplicación neuronal de relajación por formas de onda que incorpora conexiones eléctricas continuas junto con espigas. Su utilidad concreta es exigir control del intercambio, en lugar de asumir que bloques de50–100ms son causales por estar enGPU. [Hahne et al.,2015](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2015.00022/full). Los métodos multirrate acoplados también muestran que separar componentes rígidos fuertemente acoplados tiene condiciones de estabilidad; no basta asignar un paso distinto a cada área. [Sandu,2019](https://arxiv.org/abs/1808.02759), [métodos implícitos](https://arxiv.org/abs/1910.14079).

Para C, los métodos exponenciales tienen teoría de error y requisitos propios; separar una parte lineal no convierte automáticamente el paso en exacto. Deben contabilizarse las acciones de funciones de matrices y las aproximaciones no lineales. [Hochbruck y Ostermann,2010](https://na.math.kit.edu/download/papers/acta-final.pdf).

## Prueba decisiva de extensibilidad

Congelar compilador, planificador, interfaz y criterios antes de añadir un mecanismo no usado durante su diseño. Añadirlo únicamente mediante una descripción nueva y parámetros; cambiar su etiqueta entre «visual», «olfativa» y una etiqueta neutra no debe alterar las ecuaciones generadas ni su trayectoria. Cambiar las ecuaciones sí debe cambiar el programa generado y su identidad.

El banco debe mezclar transmisión gradual, espigas, retardos, canales rígidos, química y realimentación, con grafos de ramas, ciclos y conectividad irregular. La selección se hace con carga mixta, no con el área que favorezca a cada candidata. Comparar el mismo modelo y precisión con una referencia independiente pequeña y con el modelo completo al escalar. Medir arranque aparte, pero incluir integración, comunicación, eventos, verificación operativa y cuerpo en el tiempo relevante.

Escalar neuronas, conexiones, compartimentos y estados por neurona de forma separada. Registrar memoria por variable y conexión, transferencias por segundo simulado, resoluciones y rechazos. Un índice de105269células visuales no predice esos costes. La prueba de1segundo por60segundos reales sólo se aprueba con la carga completa declarada, nunca sumando aceleraciones de piezas.

## Decisión actual

No elegir aún A/B/C por plausibilidad. Conservar la captura y los negativos como controles; usar las herramientas neuronales existentes como referencias y posibles componentes, no afirmar que no existen. Primero fijar el contrato común y recibir la revisión matemática solicitada aChatGPT. El mensaje de corrección encontró la conversación ocupada: no se da por entregado hasta tener recibo.

La campaña PN residente llegó provisionalmente a2.665x en1ms prescrito, sin integración corporal ni prueba de extensión. Se suspende su promoción por la corrección de alcance del usuario, preservando fuentes, datos y fallos. La etapa3 y la meta de rendimiento completo permanecen abiertas.
