# Decisión de arquitectura a partir del bloque real

24-09-2026. El padre y sus parámetros no cambian. Esta ronda tiene una captura completa consumida; no se ha ejecutado un integrador candidato. La primitiva dispersa es infraestructura de un operador, no un tercer prototipo de organismo.

## Limitación observada y operación propuesta

El bloque aceptado de125µs tiene60 evaluaciones globales, siete eventos internos y ocho intervalos. Cualquier alternativa que haga al menos una evaluación global por intervalo tiene techo7,5× antes de auditorías: no pasa la puerta10×. Esta exclusión es condicional y no descarta multirritmo. Los4.062 puertos proyectados originan885.587 de25.582.938 aristas (3,4616%). Sus4.060 valores diferentes de tau_q impiden reducirlos exactamente a dos acumuladores comunes por receptor.

La pieza compartida es una partición declarada del operador lineal de sinapsis:

`G(x,t) = W_cont s_cont(x) + W_event s_event(t)`.

Cada consumidor declara sus escalas y canales (incluidos positivo/negativo); los adaptadores que sustituyen filas o pesos deben declararlos también. Un CSR reducido de885.587 aristas puede evaluar la segunda contribución sin leer las otras24,7millones. Esto sólo elimina el recorrido global provocado por **esa contribución**: no resuelve la recurrencia restante ni autoriza omitirla.

Para A se propone una separación exacta, no una región anatómica lenta: una versión barata `F_fast` usa la proyección exacta de eventos y un predictor de corriente continua dentro de las mismas leyes celulares; `F_slow = F_full - F_fast` conserva toda la diferencia no lineal. Las entradas PN/APL se mantienen sólo en las fronteras ya declaradas. El mismo predictor y versión del operador deben usarse en ambos términos. Congelar target/rate y omitir el resto sería otro modelo y queda excluido.

## Tres rivales vigentes

| Ruta | Información y operación | Discriminador/falsador |
|---|---|---|
| A: MRI-GARK con término recurrente corregido | Estado/puertos del mismo bloque; evolución rápida local y combinaciones lentas con coeficientes publicados. | Endpoint normalizado, defecto entre nodos y eventos correctos con≤6 productos globales equivalentes en este bloque, incluida auditoría/setup. La variante global por corte queda descartada por la cota anterior. |
| B: QSS2/CSC con trayectorias por fuente | Estado y derivadas legales, polinomios por fuente, deltas en destinos y envolvente de influencia recurrente; actualiza también sin espigas. | Si la cota fuerza actualización densa o cuesta tanto como el padre, descartar. La tolerancia de producto aislado no se convierte en tolerancia de estado. |
| C: exponencial/Krylov del Jacobiano acoplado | Operador/estado actuales; acción de Jacobiano más corrección no lineal e invalidación ante cambio de versión. | Coste de base/renovación y residual versus referencia. Reservado; no tercer prototipo completo en esta ronda. |

MRI y QSS son métodos conocidos; la partición dispersa es ingeniería, sin afirmación de novedad biológica. Publicación primaria de las tablas MRI: Sandu2019; implementación de referencia versionada [SUNDIALS v7.4.0](https://github.com/LLNL/sundials/blob/v7.4.0/src/arkode/arkode_mri_tables.def). No se atribuye a esas tablas un resultado en este organismo.

**Resultado externo posterior, comprobado localmente:** ChatGPT entregó `mri33_replay.py`, hash `c8c3d75a6ed0d8bd908f41d24804dff4a25789f6da4b8bdeb2e617d6c42a2984`, con split afín exacto y tabla ERK33a corroborada también en [SUNDIALS v7.6.0](https://github.com/LLNL/sundials/blob/v7.6.0/src/arkode/arkode_mri_tables.def). Su prueba CPU de evento tardío pasó aquí (error absoluto2,37e−9,55llamadas), pero el helper de coste aplicado a **los eventos reales** exige al menos49 evaluaciones frente a60: techo1,2245×. Esa instancia A se descarta por coste sin gastar replay GPU. Sus auditorías puntuales tampoco constituyen cota continua; el código sigue como donante matemático y negativo, no candidato admitido. La variante A de partición local anterior y B/C permanecen como hipótesis distintas sin ensayo de trayectoria real.

## Lo necesario para el siguiente replay

La captura es neutral en todos los propietarios serializados del milisegundo. El oráculo vivo reprodujo tres consultas reales, pero sus NPZ no son todavía un runtime recargable. La reparación de propiedad de memoria se comprobó con canarios: la versión antigua los corrompía aun con salidas correctas. Antes de ejecutar una trayectoria nueva deben fijarse fase/versión/intervalo, probarse consultas A→B→A contra evaluación directa y definirse el defecto únicamente sobre coordenadas libres. Los q/s impuestos se verifican con su solución analítica; su rate=0 en el integrador no significa derivada física cero.

No se buscará perfección universal ni igualdad bit a bit entre integradores distintos. La cercanía al padre y el error del método son comprobaciones diferentes. El modo rápido posterior tendrá tolerancias funcionales por observable fijadas antes del ensayo; el ruido estocástico, si se modela, será explícito y no una interpretación del redondeo.

Presupuesto conservado: como máximo dos prototipos/replays de120s,18GiB RAM/12GiB VRAM; gate10× sin cambio. La sonda de operador tiene presupuesto propio menor y ninguna admisión de etapa. Las etapas3/4/5 conservan sus veredictos existentes.
