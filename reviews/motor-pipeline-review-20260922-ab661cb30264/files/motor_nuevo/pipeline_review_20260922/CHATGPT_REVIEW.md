**Conservaría A y desplazaría el trabajo hacia el CNS.** La independencia por célula y la restauración explícita del operador están respaldadas por las fuentes y los resultados publicados. Encontré **dos carencias reproducibles del registro de operador** y una condición de publicación que todavía conviene probar. Ninguna demuestra que las continuaciones exitosas que entregaste sean incorrectas.

**Ejecuté la prueba CPU original de `OperatorState` y dos contraejemplos artificiales. No ejecuté CUDA, el organismo ni la verificación de sus arrays.** Las dos fuentes usadas en CPU coinciden con sus identificadores de blob Git. No descargué ni verifiqué el ZIP completo.

:chatgpt-content-reference{index="18"}

## 1. Registro del operador: dos condiciones pendientes

### El manifiesto interno autentica los valores, pero no sus destinos

`from_state()` comprueba hashes, tipos y formas de los arrays. Sin embargo, acepta el mapa `bindings` del snapshot sin contrastarlo con un esquema esperado ni incluirlo en la identidad comprobada. 

**Contraejemplo CPU ejecutado:** guardé `tau=[1,2]` y `theta=[3,4]`, intercambié únicamente sus rutas en `bindings` y restauré. El resultado fue:

| Campo | Esperado | Restaurado |
|---|---|---|
| `tau` | `[1,2]` | **`[3,4]`** |
| `theta` | `[3,4]` | **`[1,2]`** |

La restauración terminó correctamente y `differences()` devolvió `{}`, porque verifica los valores contra el mismo mapa alterado.

**Corrección:** autenticar conjuntamente esquema, rutas, tipos, formas y valores; además, contrastar las rutas con los bindings admitidos por el adaptador de destino. Un manifiesto externo que autentique el snapshot completo podría detectar esta corrupción antes de llamar a la clase. **No encontré esa alteración en tus datos: el defecto es del contrato interno de restauración portable.**

### Una escritura posterior puede fallar dejando una restauración parcial

La prevalidación comprueba forma y dtype, pero no que todos los destinos sean escribibles. La prueba existente cubre una forma incorrecta en el segundo destino, no un fallo al escribirlo.  

**Contraejemplo CPU ejecutado:** el primer destino se restauró; el segundo era un array NumPy de solo lectura. Se lanzó `ValueError`, pero la primera escritura permaneció.

Comprobaría escribibilidad antes de empezar y establecería una regla explícita para fallos durante las transferencias: **rollback verificable o propietario invalidado**, nunca continuación sobre una restauración parcial. La prueba CPU no reproduce un fallo CUDA.

Estas carencias **no contradicen** la reparación de `tau/theta`: el comprobador de recuperación conserva el registro original, restaura sus valores y reconstruye después los ejecutores; `VERIFIED.json` comunica cero diferencias en ambas continuaciones. Esa evidencia sigue delimitada a los casos ejecutados.  

## 2. A: la barrera funciona por diseño; falta cerrar un error posterior

En `device_cell.cu`, las decisiones de cada bloque son uniformes entre sus lanes; el estado aceptado, detectores y eventos permanecen en almacenamiento privado. `device_cell.py` espera y comprueba todos los estados de salida antes de copiar al propietario. **No identifiqué una lectura del estado evolutivo de otra célula ni una confirmación externa durante un rechazo local.**   

La condición pendiente está **después** de esa barrera: primero se copian membranas y campos axonales al propietario y después se llama a `events.active.add(...)`; los relojes se actualizan al final. Si el receptor de eventos lanza una excepción, queda una publicación parcial dentro del adaptador. Esto se deduce del orden del código; **no lo ejecuté ni afirmo que el rollback exterior falle**. 

Añadiría un único negativo al comprobador existente: hacer que `active.add` falle después de un kernel exitoso. El contrato debe garantizar que la capa exterior restaure **estado, operador, publicador y eventos**, o invalide la sesión. El ensayo actual provoca el fallo antes de la publicación y no cubre ese punto. 

No exigiría otra campaña del organismo para comprobarlo: es un ensayo de la frontera de confirmación.

## 3. B afín: conservar su alcance separado

La implementación aplica saltos al comienzo de cada segmento y respeta las consultas por prefijo; el programa mantiene el orden de los cambios de operador. **No encontré una sustitución por áreas promedio ni una omisión evidente del salto en una frontera interior.** Las pruebas publicadas incluyen precisamente prefijos, evento tardío y orden no conmutativo.   

Mantendría la distinción declarada: la cola de Taylor acota una parte del cálculo afín en aritmética exacta; **no cubre congelación de coeficientes, recurrencia no lineal ni todo el redondeo FP64**. No utilizaría sus resultados como admisión automática para el CNS. 

## 4. El discriminador macro siguiente: ¿cuánto trabajo global obliga realmente cada evento?

El recibo aporta información más específica que el porcentaje de tiempo:

**320 épocas CNS, 3.677 intentos aceptados y cero rechazados.** Con seis evaluaciones de coeficientes por intento, corresponden a **22.062 evaluaciones completas**, incluidas las del predictor. El recibo registra también 1.564 eventos procesados; no deben interpretarse como 1.564 espigas nuevas confirmadas del organismo, porque estos contadores incluyen trabajo predictor.  

Esto **no demuestra que todos esos intentos los causen los eventos**: también intervienen paso máximo, cola de intervalo y controlador. Falta separar esas causas.

### Propongo un solo discriminador: mapa causal de trabajo de una época real

Capturar **el primer intercambio aceptado de 125 µs que contenga eventos**, distinguiéndolo del predictor. Sin cambiar sus entradas ni eventos, registrar:

**Por qué termina cada subpaso:** evento, paso propuesto por precisión, máximo permitido o fin del intercambio.

**Qué dependencias alcanza cada evento:** conservar fuente y marca temporal, no únicamente la lista de tiempos que actualmente recibe el controlador global. Construir el mapa desde el operador efectivo, incluidas rutas reemplazadas, filtros y ciclos; **la matriz `W` por sí sola no representa necesariamente todas las dependencias**. La interfaz publicada entrega al controlador los tiempos, pero no la identidad de las fuentes. 

Mediría el trabajo de coeficientes/aristas que habría que repetir en las componentes afectadas frente al trabajo global repetido. **Eso estima trabajo evitable, no segundos ganados.** La evolución suave de las componentes no afectadas debe continuar; no se las congela por no recibir un evento.

La decisión sería:

**Si predominan cortes por eventos y sus dependencias quedan confinadas en componentes pequeñas**, priorizar partición con relojes y publicación causal por componente.

**Si las dependencias instantáneas y recurrentes abarcan casi todo el operador**, esa partición exacta tiene poca oportunidad: priorizar integración de puertos con una cota del defecto no lineal. Un grafo conservador muy conectado no refuta cualquier multirrate posible; sí debilita la propuesta de bloques independientes sin aproximación de sus fronteras.

**Si la mayoría del trabajo no procede de cortes por eventos**, ninguno de esos dos enfoques ha justificado aún ser la prioridad: revisar el coste de las evaluaciones y el controlador que las solicita.

El falsador común es simple: **la reducción de trabajo declara una componente no afectada o un error de receptor acotado, pero el replay completo contradice esa predicción bajo los mismos eventos y entradas**. Las coincidencias de estado final no rescatan un transitorio omitido.

## Decisión

**No seguiría centrando la ronda en membranas.** En la medición publicada, CNS supone aproximadamente **70,3 %** del paso y membranas **10,0 %**. Hacer gratuitas estas últimas permitiría, aritméticamente y manteniendo el resto igual, solo **1,11× total**; no es una predicción de rendimiento. 

Conservaría A, cerraría los negativos de registro/publicación y realizaría ese **único diagnóstico de causas y dependencias de los cortes** antes de implementar otra reforma macro.

**Lecturas:** completos los archivos solicitados, `ARCHITECTURE.md`, `affine_ports.py`, `check_operator_state.py`, `check_operator_recovery.py`, `check_device.py`, `check_affine.py`, `benchmark_real.py` y el `organism_adapter.py` publicado; consulté parcialmente el índice y la sección final de `device20_01/RESULT.json`. El detalle está en el paquete CPU. **No comprobé independientemente los arrays del organismo, el manifiesto completo ni una trayectoria de un segundo.**
