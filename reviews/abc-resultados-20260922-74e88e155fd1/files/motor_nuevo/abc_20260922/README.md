# MOTOR ABC: discriminador recurrente base

**PROMETEDOR_NO_CONFIRMADO. Se ejecutó 1 segundo continuo del bloque neuronal base en 20.606 segundos reales.** Incluye muestreo y comprobaciones durante avance; excluye construcción/JIT y compresión posterior. No contiene PN/KC/APL/retina especializada ni cuerpo. No se validó contra referencia refinada durante ese segundo completo; el prefijo de 5 ms coincide exactamente con el ensayo corto. Etapa 3 abierta.

## Comparación congelada

166.700 neuronas, 25.582.938 conexiones almacenadas, dos condiciones expuestas (tónica y pulso visual), 5 ms por condición. Referencia RK4 refinada de 6,25 a 3,125 µs. Error de referencia máximo: 2.05886e-08. Límite de candidatas: 1e-4 en q/s en muestras cada 0,5 ms. Es concordancia numérica muestreada, no cota de error continuo.

| Variante | Máximo error q/s, dos condiciones | Pared 5 ms tónica / pulso (s) | Ambas pasan |
|---|---:|---:|---|
| A125 | 4.55651e-05 | 0.5304 / 0.5299 | True |
| A250 | 0.00012491 | 0.2658 / 0.2686 | False |
| B125 | 7.76968e-06 | 0.2116 / 0.2114 | True |
| B250 | 2.56865e-05 | 0.1088 / 0.1076 | True |
| C125q7 | 7.76969e-06 | 0.5678 / 0.8434 | True |
| C125q5 | 1.49004e-05 | 0.2549 / 0.4922 | True |

A125 conserva precisión; A250 falla sin rescate. La iteración implícita resuelve su ecuación, pero eso no garantiza exactitud temporal. B250 es el más rápido de los elegibles. C reduce aristas actualizadas, pero el coste de escrituras atómicas hace que pierda ante B. No se descarta toda integración implícita ni toda propagación incremental: la selección sólo concierne este bloque y estas condiciones.

## Autocrítica y próxima decisión

El trabajo anterior medía piezas con entradas prescritas: eso era insuficiente para decidir velocidad de una red en evolución. Aquí se ejecutó recurrencia y un segundo real de tiempo neuronal. Sigue siendo incorrecto compararlo como si sustituyera la mosca heterogénea: usar sólo el bloque base elimina precisamente varios componentes rígidos costosos. Tampoco basta mantener conteos de eventos: la futura integración debe conservar tiempos y feedback.

Mantener tres vías: A, acoplamiento implícito conjunto de PN para reducir iteraciones globales; B, avanzar PN/KC/eventos con estado residente y frontera corporal de 1 ms; C, bloques y propagación incremental de señales donde la actividad justifique el coste de colas. El resultado actual prioriza B para recorridos densos; no decide el solver PN. No añadir variantes a esta ronda.

## Verificación y presupuesto

Se ejecutaron 16 prefijos: cuatro referencias y doce candidatos, más un segundo base (1/1 autorizado). Cero corridas nuevas del organismo completo y cero modificaciones al motor productivo. Independencia local: NumPy para RHS y Radau sobre 16 células acopladas sintéticas, doce comparaciones; prueba de invalidación A–B–A y actualización CSC sin umbral. Verificador reconstruye decisiones desde arrays y contrato congelado; pruebas de corrupción de criterio, contexto, alcance y bandera de finitud.

Pico RSS del proceso largo: 1.429 GiB. Pool CuPy: 0.321 GiB (no es VRAM total del proceso). Tiempo de avance registrado: 20.388 s. Precisión FP64.

Jev realizó una llamada real con cuatro tareas; clasificó razonamiento, implementación y revisión externa. No tomó decisiones científicas. ChatGPT recibió fuentes públicas fijadas al commit para revisión. PRO/máximo confirmados por el usuario, selector no observable. Sin subagentes Codex.

## Reproducción desde el ZIP extraído

Python 3.10, NumPy, SciPy 1.15.3 y CuPy-CUDA; GPU NVIDIA con FP64 y CUDA Graphs. Versiones ejecutadas adicionales en runs_01/freeze.json. Los siguientes comandos no importan fuentes del árbol de trabajo original. El corto verifica arrays y pruebas independientes; el completo repite las 16 corridas base en un directorio nuevo. La ejecución larga original se verifica por hashes/prefijo; no se repite automáticamente para preparar entregas.

```bash
export OPENBLAS_NUM_THREADS=1
python -I -B -O verify_package.py
python -I -B -O motor_nuevo/abc_20260922/test_numerics.py
python -I -B -O motor_nuevo/abc_20260922/test_verifier.py
python -I -B -O motor_nuevo/abc_20260922/report.py
python -I -B motor_nuevo/abc_20260922/recurrence.py data/connections.npz replay_base
python -I -B -O motor_nuevo/abc_20260922/verify.py replay_base replay_result.json
```

Parada: hito completo y variantes de esta ronda agotadas. Próxima campaña: subsistema PN y acoplamiento heterogéneo, con A/B/C, contrato nuevo y tolerancias conservadas.

Antecedentes matemáticos: [SciPy, integración y Radau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html); [QSS Solver original](https://github.com/CIFASIS/qss-solver). C es actualización incremental con reloj RK4, no una implementación de QSS/LIQSS. La publicación emplea ZIP dividido para respetar [límites de archivos de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Revisión externa resuelta con pruebas

ChatGPT declaró leer siete archivos del commit de diseño, sin ejecutar corridas. Sus observaciones motivaron una repetición instrumentada de los16prefijos: A, B y referencias no activaron guardas y conservaron exactamente sus trayectorias. C sí activó la guarda de transmisión intermedia negativa en 4 corridas. La clasificación por signo fijo del peso no coincide en general con el operador original durante esas etapas: **C queda descartada en esta ronda**, además de perder en coste. No se cambiaron sus parámetros ni se rescata el código congelado. El PASS q/s muestreado original se conserva como resultado numérico, no como prueba de equivalencia del operador.

Se reprodujo deliberadamente el NaN oculto por fmax y se verificó una bandera persistente previa a esas operaciones. SafeEngine es la entrada prospectiva que rechaza esas banderas y el dominio no soportado deC. La comparación de buffers CSC y captura/sin captura pasó en dominio no negativo. El primer intento instrumentado terminó al detectarC; fallo y fuente exacta conservados en guarded_01.

La repetición instrumentada del segundo completo tomó 21.513s, con guardas intermedias sin activación y archivo de trayectoria completa idéntico por SHA256. No aporta una referencia de exactitud a1s: comprueba integridad y reproducibilidad. Presupuesto adicional de verificación por defecto real:16prefijos completos, un intento instrumentado parcial y un segundo repetido; cero candidatas nuevas. Estos costes se separan de las mediciones primarias.

Todos A/B/C usan GPU y grafos en este discriminador; su comparación identifica coste del método/propagación sobre esta plataforma, no una ablación causal de residenciaCPU/GPU. Orden fijo y una corrida por condición limitan inferencias sobre diferencias pequeñas. Preparación, construcciónCSC, captura y compilación quedan fuera de los tiempos de avance.
