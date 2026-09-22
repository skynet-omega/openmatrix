Revisión documental externa, turno 2caa7e0b-4b07-40c8-8d5f-435dfbe00b3f. No ejecutó los arrays ni CUDA.

**La evidencia permite retirar el fallo del filtro como explicación prioritaria del transitorio de 75907 y concentrarse en la temporización de las emisiones. Continuaría con la comparación de la guardia contra la referencia refinada de 1 ms que ya programaste.** No cambiaría el filtro ni combinaría todavía esa prueba con la candidata B.

**Revisión documental:** leí los archivos indicados al final. No ejecuté el replay CPU/CUDA, reconstruí los arrays originales ni examiné una implementación publicada de la guardia; sus tiempos y resultados locales son los que comunicas.

## 1. La interpretación causal está sustentada, con un límite preciso

`check_port_replay.py` utiliza una referencia independiente adecuada: evolución mediante `scipy.linalg.expm` de la matriz triangular 2×2 **entre saltos**, aplicando las amplitudes efectivas después del clipping. Consulta también inmediatamente antes, en y después de las marcas, y contrasta los valores publicados al terminar cada frontera. El recibo comunica **166 consultas y un máximo de \(3,47\times10^{-18}\)** entre referencia y puerto CUDA. Esto comprueba el filtro para ambas historias suministradas; no la precisión de las marcas que produjeron esas historias.  

El intercambio de marcas aporta evidencia causal **local**:

| Contraste publicado | Qué demuestra |
|---|---|
| Primer intercambio aceptado: mismos \(q_0,s_0\), conductancias y saltos; cambiar solo marcas elimina la diferencia del filtro | La temporización basta para explicar esa separación del filtro en esa frontera. |
| Época 13: al intercambiar marcas aún quedan \(1,67095\times10^{-4}\) de diferencia máxima entre filtros | Existen diferencias heredadas en otras coordenadas; el contrafactual no reejecuta el organismo ni borra su historia anterior. |

Esa delimitación está correctamente declarada en `analyze_boundaries.py`. **No extendería “elimina toda diferencia” más allá del filtro en la primera frontera.**   

Para el foco **75907**, PORT sitúa el salto posterior en **65,625 frente a 68,750 µs**, con la misma amplitud **0,29942891126207577** y estados iniciales del puerto iguales salvo redondeo. La diferencia de \(s\) llega a **\(1,84402\times10^{-4}\)** al terminar esa época y permanece en **\(1,78588\times10^{-4}\)** al terminar el milisegundo. **El foco del máximo final no es, por ello, la primera fuente divergente del organismo.**  

## 2. Dependencia concreta de la guardia: «estar sobre el umbral» no es solo «cruzarlo»

`EVENT_CANDIDATE.md` declara activar el límite cuando la observación somática o axonal, **en el estado comprometido o en los estados del ensayo**, alcance −40 mV. No debería implementarse únicamente como detección de un cruce desde abajo. 

**Falsador mínimo:** una célula comienza a −35 mV, sigue ascendiendo y después desciende, permaneciendo por encima de −40 mV. Una propuesta mayor de 1562 ns debe quedar protegida por la guardia **aunque no haya ningún nuevo cruce del umbral**. Si solamente se comprueba el cambio de signo alrededor de −40, podría seguir fechándose el máximo con muestras demasiado separadas.

No afirmo que tu implementación cometa ese error. Es la condición concreta que comprobaría al leerla, sin añadir umbrales ni buscar otra configuración favorable. El rechazo debe ocurrir antes de publicar cualquiera de sus medias etapas o eventos.

## 3. El techo temporal no equivale al presupuesto de error del filtro

Una cuenta sobre el puerto publicado muestra por qué hace falta el contraste integrado.

Para un único salto de amplitud \(J\), con historia previa y constantes idénticas, la respuesta filtrada satisface la cota:

\[
|\Delta s(t)|\le \frac{|J|}{\tau_s}\,|\Delta t_{\mathrm{evento}}|.
\]

Con \(J=0,2994289\) y \(\tau_s=0,005\) s, **si la diferencia entre marcas fuese 1,562 µs**, esa contribución podría alcanzar aproximadamente:

\[
9,35\times10^{-5}.
\]

Es una **deducción matemática**, no otro resultado ejecutado. Ya consume casi todo \(10^{-4}\) para un solo salto; faltan diferencias previas, otros eventos y acoplamiento. Además, **limitar el paso a 1562 ns no demuestra que la diferencia entre marcas esté acotada por ese mismo número**. Los parámetros usados en la cuenta proceden de PORT. 

No propongo reducir ahora el techo. Propongo mantenerlo congelado y decidir por el resultado medido, como establece la candidata.

## 4. Qué debe decidir la comparación próxima

La dependencia obligatoria antes de promover es **que la mejora sobreviva en la trayectoria acoplada**, no solamente en dos eventos de un fixture. Mantendría \(10^{-4}\), las reglas discretas declaradas y la separación entre predictor y estado confirmado. Registraría marcas, saltos y conteos por frontera, junto con \(q/s\): la separación de aproximadamente **0,294** y la desigualdad temporal de conteos muestran precisamente por qué el conteo final no basta.  

No realinearía curvas para ocultar la discrepancia ni utilizaría el padre grueso como verdad. Tampoco exigiría una garantía universal sobre picos ocultos: **basta la evidencia acotada del contrato para avanzar**, pero el alcance debe excluir lo que la guardia muestreada no comprueba.

**La candidata B permanece separada:** modificar la propuesta nominal del CNS en esta misma comparación dificultaría atribuir cualquier mejora a la guardia. Si A falla, conservar ese negativo; si satisface el contrato, la decisión posterior puede avanzar sin otra búsqueda hasta PASS.

**Archivos leídos completos:** `README.md`, `PLAN.json`, `PORT_75907.json`, `LOCALIZATION.json`, `PORT_REPLAY.json`, `check_port_replay.py`, `analyze_boundaries.py`, `trace_boundaries.py` y `EVENT_CANDIDATE.md`, del commit `b2ecd5…`. **No encuentro un error documental que obligue a detener la prueba prevista; sí las condiciones anteriores antes de promover su resultado.**
