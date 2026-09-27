# Decisión posterior a la atribución de 48

**Etapas 4/5 abiertas. La consulta acotada terminó; no apareció un bug en las seis sumas examinadas.** Se conservan el motor, el cuerpo y los lectores. La campaña 48 sigue siendo negativa para iniciar avance; no probó recuperación del viento. Esta atribución comprueba una operación del preparado, no su equivalencia fisiológica ni la viabilidad global del objetivo.

## Qué cambia con esta evidencia

Se extrajeron 7227 aristas entrantes a seis destinos prefijados, conservando identidad anatómica, pesos efectivos y transmisión filtrada. Los ocho saldos finales DNg100 se reproducen bit a bit con el orden FP32 del motor. Los pesos son iguales entre brazos. El efecto final en su entrada depende de señales presinápticas distintas, sobre un balance positivo/negativo grande que conserva saldo negativo. Los tipos de mayor aporte ya se conocían en el preparado del 26-09: este trabajo sigue su estado durante otra condición; no descubre una nueva causa.

El control de rangos observados permite combinar independientemente todas las variaciones presinápticas de estos cuatro endpoints. Aun con esa libertad artificial, DNg100 conserva margen negativo. Esto acota la hipótesis de que bastaría recombinar **estos valores** con la misma lectura. No limita otros estados, olores o leyes fisiológicas y no selecciona un valor de ganancia/umbral.

DNb05 recibe algunas ORN prescritas directamente: 2 aristas izquierdas y 16 derechas, todas VA6. En profile, su aporte directo al cambio de entrada es aproximadamente 0,074/0,860 frente a cambios totales de 9,943/53,368. La mayoría de ese contraste de frontera corresponde algebraicamente a las demás fuentes. No se interpreta la diferencia como mediación causal identificada.

Usar liberación instantánea en lugar de transmisión filtrada desplaza el saldo DNg100 unos 29–31 puntos internos, magnitud comparable con algunas diferencias entre brazos. La publicación no debe usarse como sustituto de la señal consumida. No se encontró que el motor cometa esa sustitución.

## Alternativas vigentes y discriminadores

| Alternativa | Información y operación que se examinarán | Discriminador y falsador |
|---|---|---|
| A. Transformación de señal y balance en la red | Señales firmadas realmente consumidas y sus cambios antes de la rectificación, con historia común | Una ruta candidata debe predecir un cambio finito local y un signo downstream. Un aporte grande por sí solo no la selecciona. Si el cambio local ocurre y el efecto previsto no, se descarta esa hipótesis. |
| B. Ley o estado local de un puerto/neuronas | Entrada, estado y objetivo después de todos los propietarios; evidencia fisiológica independiente pertinente | Una discrepancia concreta entrada→respuesta debe sobrevivir controles de estado/interfaz. Una señal ya pequeña antes del nodo no acredita que su ley la haya extinguido. No se recalibra por obtener movimiento. |
| C. Error de identidad, escala o implementación | Misma evaluación, mismos operandos, orden numérico y objetivo final | Sólo una contradicción con esos controles justifica reparación. Los ocho saldos DNg100 actuales no presentan esa contradicción. No se extiende el resultado al resto del cerebro. |

## Próximo paso seleccionado

Preparar una **captura breve de operandos consumidos**, antes de otra intervención neuronal. Se mantienen DNg100 y DNb05; DNa02 permanece como observador. La primera captura debe registrar las entradas de estas filas en la misma llamada del coeficiente, con pesos transitorios, transmisión, caps, máscara, drive, theta/gain, objetivo final, contexto predictor/comprometido, aceptación y límites de eventos. Debe permitir reconstruir esa llamada sin volver a evaluar el cerebro ni usar publicaciones de otro instante. No se le llamará experimento causal mientras sólo observe.

La captura se justificará por localizar **cambios de contribución** y una predicción local finita, no por volver a demostrar target=0 ni elegir el mayor inhibidor. Se conservarán los antecedentes negativos P9/AOTU/normalización/preparación y el contraste lateral46. No se repite una campaña larga ni se añade otra descarga genérica de datos.

Antes de adquirir datos nuevos se requieren dos comprobaciones de ingeniería concretas: (1) un testigo positivo/negativo contra el observador real y sus rutas de captura, fuera de la dinámica; un tanh de NumPy aislado no certificaría el observador CUDA; (2) una continuación realmente cualificada o reconstrucción desde la preparación ya válida. Los final_state48 dicen restart_tested:false. No inventar un checkpoint de 1000 ms ni presupuestar sólo la ventana omitiendo su prefijo. Esta ronda presupuestó cero pasos nuevos; el piloto siguiente debe fijar coste total, condición, ventana y efecto material antes de ejecutarse. Como máximo dos prototipos completos; ninguno se abrió aquí.

La iniciación autónoma y la orientación con avance común explícitamente asistido siguen siendo preguntas separadas. No se exige resolver DNg100 como puerta universal para estudiar orientación; tampoco se propone repetir46 sin una predicción distinta. VNC y patas conservan su prioridad posterior. La etapa5 requiere viento y recuperación del error, ausentes de48.

## Contraste con los tres asesores

- **Motor C++/CUDA** comprobó en archivos locales la identidad, rutas especiales y procedencia de caps/CSR. Su verificación no recalculó nuestras sumas ni reprodujo el cerebro. Es una revisión no autoral acotada de esquema, no una auditoría global.
- **ChatGPT ASTRA_V2** propuso descomposición simétrica de cambios de señal/pesos y captura breve. Se incorporan ambas, corrigiendo su extensión de la coincidencia a seis RHS y nuestra comunicación inicial sobre ORN directas. Sólo se contrastaron dos saldos DNg100 por brazo. Ninguna diferencia entre instantes distintos basta para diagnosticar bug. No convertir su preferencia de secuencia en una puerta obligatoria de orientación.
- **ChatGPT_Motor_V2** corrigió su propuesta de Jacobiano: con margen negativo, la derivada del objetivo rectificado es cero y no refuta una perturbación finita. Se conserva el discriminador anterior a la rectificación y el testigo de captura; no se añade un buscador operativo al organismo.

Ambos ChatGPT revisaron cifras y argumentos enviados; no ejecutaron los NPZ de esta consulta. El modo PRO no está verificado. Se conservaron respuestas originales, errores y correcciones; una coincidencia entre asesores no convierte la inferencia en evidencia biológica.

**Motivo de parada:** atribución y control de rangos completados, mecanismo causal aún no identificado. Se entrega un resultado reproducible y una próxima captura con finalidad precisa, sin escoger una lesión o recalibración por conveniencia del PASS.
