Turno ea8e7f70-70b7-4d36-96d4-f859a6c31806. Exposición al historial declarada; no ejecutó código.

**Propongo tres alternativas distintas: cambiar cuándo se intercambian trayectorias, cambiar cómo se compila el operador y cambiar el integrador global.** Ninguna cuenta al padre como candidata. Desarrollaría **1 y 2** esta ronda; conservaría 3 como rival especificado. Son propuestas informadas por nuestro historial, no una revisión ciega ni métodos matemáticos inéditos.

Releí `graph_core.py`, `event_ports.py` y las primeras 120 líneas de `gpu_coefficient_layout.py` de `39b86dc…`. **No ejecuté código ni arrays.** El código confirma seis evaluaciones globales por intento y una cadena que calcula resultados posteriormente reemplazados: hay dos costes distintos que atacar.  

## 1. Intercambio de trayectorias con corrección recurrente

**Operación que cambia.** Sustituir evaluaciones globales en cada subpaso local por iteraciones sobre trayectorias dentro del intercambio vigente:

\[
M_i\dot x_i^{(k+1)}
=F_i\!\left(x_i^{(k+1)},u_i^{(k)}(t)\right).
\]

Después de integrar los bloques, recalcular las entradas recurrentes y corregir hasta satisfacer el presupuesto de error. Los puertos afines aportan su historia analítica; **las contribuciones neuronales desconocidas se calculan, no se reproducen desde la referencia**.

**Información y estado.** Trayectorias provisionales, eventos fechados, retardos, operadores efectivos y estado inicial íntegro. La partición incluye dependencias de masa: no se omiten términos entre bloques.

**Coste evitado.** Recorridos globales de conexiones impuestos por eventos locales. CUDA ejecutaría bloques con trabajo desigual y productos dispersos agrupados por nodos temporales.

**Riesgo causal.** Confirmar una iteración que aún utiliza entradas recurrentes inconsistentes. Deben rechazarse conjuntamente estados, emisiones e historias provisionales; convergencia entre iteraciones y error temporal son controles diferentes.

**Falsador barato.** Dos unidades con realimentación instantánea y pulsos de igual integral pero orden opuesto. Deben conservarse el efecto del orden y su retorno; cambiar la partición no debe alterar el resultado fuera del presupuesto.

**Comparación real.** Medir recorridos dispersos, iteraciones y tiempo completo, incluyendo construcción de trayectorias. Si las correcciones consumen el ahorro, se descarta.

Antecedente cercano: [Hahne et al., integración conjunta de dinámica continua y espigas](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00034/full). Fundamenta la separación entre dinámica y comunicación, **no una aceleración garantizada aquí**. :chatgpt-content-reference{index="2"}

## 2. Compilador del operador efectivo: eliminar trabajo sobrescrito

**Operación que cambia.** Convertir la cadena de reemplazos de `coefficients_gpu` en una representación de dependencias con versiones explícitas de cada variable. Generar únicamente los cálculos necesarios para los `target/rate` finales, compartiendo subexpresiones y recorridos compatibles.

No es «bajar Python a CUDA»: **el grafo ya está capturado**. Es retirar operaciones que no contribuyen al resultado. Actualmente existen vistas copiadas, pesos temporalmente modificados/restaurados y destinos recalculados por capas posteriores. 

**Información y estado.** Mapa de lecturas/escrituras, orden de reemplazos, rutas efectivas y parámetros mantenidos por época. Las intervenciones invalidan las especializaciones afectadas. Los pesos temporales pasan a operandos privados, no a mutaciones compartidas.

**Coste evitado.** Ensamblados descartados, tráfico de memoria y recorridos repetidos. Conserva inicialmente integrador, eventos y número de evaluaciones.

**Riesgo causal.** Eliminar un resultado intermedio que otra capa consume, o convertir una entrada dinámica en constante. No reasociar reducciones inadvertidamente ni activar `fastmath`.

**Falsador barato.** Una cadena donde una capa modifica pesos, otra consume el resultado y una tercera sobrescribe solo parte de las salidas. Probar también restauración de pesos e intervención entre épocas. Comparar todos los resultados finales y efectos laterales declarados.

**Comparación real.** Mismos estados de etapa y mismas propuestas; medir bytes/aristas procesados y tiempo del organismo. La fracción eliminable todavía debe medirse: no presupongo que sea mayoritaria.

Antecedente: [compilador optimizador NMODL](https://arxiv.org/abs/1905.02241), con transformaciones simbólicas antes de generar código. No trasladaría sus cifras de rendimiento. :chatgpt-content-reference{index="4"}

## 3. Integración global de orden alto con estimador embebido

**Operación que cambia.** Reemplazar el punto medio exponencial con duplicación por **ERK8/7**, evaluando el RHS completo sobre estados de etapa y proyectando los puertos analíticos en sus tiempos correctos. Conservar inicialmente las fronteras de eventos.

**Información y estado.** Vector libre, estados propietarios de puertos, `target/rate` efectivos y entradas temporales. El primer alcance sería el CNS normalizado; **no convertiría la PN de masa no diagonal en masa identidad**.

**Coste evitado.** Repetición de un paso completo y dos medios pasos como estimador. La apuesta es aceptar menos pasos por precisión, **no reducir evaluaciones por paso**: Fehlberg 8/7 utiliza 13 etapas. :chatgpt-content-reference{index="5"}

**Riesgo causal y numérico.** Evaluar puertos con reloj incorrecto, ignorar discontinuidades o perder estabilidad/positividad que proporcionaba la actualización exponencial.

**Falsador barato.** Cascada con evento tardío más receptor recurrente no lineal, incluyendo estado próximo al límite de dominio. Comparar trayectoria, no solo extremo; ningún `clip`.

**Comparación real.** Misma guardia y operador, incluyendo todas las etapas/rechazos. Si dominan cortes obligatorios o estabilidad, el orden alto puede empeorar el coste.

Antecedente reutilizable: [tablas oficiales ARKODE 7.6](https://sundials.readthedocs.io/en/v7.6.0/arkode/Butcher_link.html). :chatgpt-content-reference{index="6"}

## Tres hipótesis para el fallo de dominio

No elegiría una sin capturar la primera operación infractora.

| Hipótesis | Dato mínimo discriminante |
|---|---|
| **H1. Redondeo/cancelación:** la solución de esa operación permanece en dominio en precisión superior, pero FP64 sale ligeramente. | Operandos exactos de la operación, resultado FP64 y replay de precisión superior; distancia al límite. |
| **H2. Incumplimiento matemático del contrato:** target/tasa/salto inválido, o dominio declarado incorrectamente para esa coordenada. | Identidad y dominio del estado; valor anterior, target, tasa, duración y salto efectivo. Para el puerto, historia completa de esa fuente. |
| **H3. Estado equivocado consumido:** reloj, buffer, propietario o versión incorrectos. | Los mismos datos registrados al producirse y consumirse, generación del buffer y replay sobre copias inmutables. |

Para \(h,r\ge0\) y \(z,a\in[0,1]\), la actualización \(z+(1-e^{-hr})(a-z)\) conserva el intervalo matemáticamente. Ese invariante separa hipótesis; **no autoriza clamping**. El diagnóstico actual captura el endpoint fallido, no todos esos operandos. 

## Decisión y mínimo pendiente

Comparador común: mismo organismo, operador, guardia, relojes sensoriales/motores y \(10^{-4}\); transitorios, eventos y coste completo incluidos. Primero replay acotado con recurrencia libre; después, al sobreviviente, trayectoria continua de segundos dentro del presupuesto.

**No falta otro ZIP ni código para formular las tres propuestas.** Para implementar 1/2 falta materializar el contrato activo de dependencias; para resolver el fallo, los operandos de la primera infracción. Elegiría **1 y 2 separadamente**, sin combinar sus mejoras antes de medirlas. Ninguna promete alcanzar 1 s/60 s.
