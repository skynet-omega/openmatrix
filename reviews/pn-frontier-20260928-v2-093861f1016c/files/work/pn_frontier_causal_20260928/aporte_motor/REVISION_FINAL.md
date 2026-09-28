# Revisión independiente final de la frontera PN

28-09-2026 · Motor C++/CUDA · lectura de resultados existentes.

**La intervención cumple sus criterios instrumentales y produce un efecto causal parcial en el mando de giro. No demuestra navegación ni supera las etapas 4/5. Priorizaría C: capacidad transmisora efectiva hacia DNg100, con los operandos ya adquiridos.**

No encontré un fallo de implementación o de cálculo que invalide el resultado acotado. Reconstruí los estímulos, el consumo y los mandos sin importar el simulador ni los verificadores del autor. Los resultados numéricos y límites de esta revisión están en [VERIFICACION_FINAL.json](VERIFICACION_FINAL.json).

## Resultado reconstruido

Ventana congelada: 51–89 ms, ambos extremos incluidos. `self` mantiene durante cada milisegundo la primera muestra natural PN de ese lado; `common` entrega a ambos lados el promedio de las dos plantillas; `dose` redistribuye ese promedio conservando el total interno `caps*q` propio de cada lado.

| Intervención | Giro L (°/s) | Giro R (°/s) | Semicontraste (L−R)/2 (°/s) |
|---|---:|---:|---:|
| self | 2,828189438 | 2,791422734 | +0,018383352 |
| common | 2,806684376 | 2,813044558 | −0,003180091 |
| dose | 2,806230178 | 2,818310222 | −0,006040022 |

Respecto de `self`, `common` cambia L en −0,021505062 y R en +0,021621824 °/s; `dose`, en −0,021959260 y +0,026887488 °/s. Las reducciones **firmadas** del semicontraste son 0,021563443 y 0,024423374 °/s. Cumplen los signos y mínimos previamente fijados: 0,005 °/s por lado y 0,01 °/s de reducción. Coinciden con la lectura del autor.

El mando de avance es exactamente cero en todos los intervalos guardados de los ocho brazos. Los giros medios siguen siendo positivos en ambos espejos. La inversión del pequeño semicontraste no significa que la mosca haya orientado correctamente su cuerpo.

Comprobaciones pertinentes:

- Los controles vivos de 12 ms coinciden con 57 en todos los campos de trazas y testigos PN retenidos. Esto no equivale a comparar cada variable interna del organismo.
- Los controles `self` de 89 ms cumplen la fidelidad prefijada. Error máximo de giro en la ventana: L 0,000605915 y R 0,001057691 °/s, frente al límite 0,002. Error máximo del diferencial DNb consumido: 7,244×10⁻⁷ y 1,228×10⁻⁶, frente a 1,6×10⁻⁶. Avance idéntico a la referencia.
- Soporte de 686 IDs/filas, dominio [0,1], capacidades iguales, estado inicial de liberación publicado igual, reloj, baseline, identidad de las cuatro DN y desfase del lector de un intervalo: coherentes. Reconstrucción exacta del mando desde la señal realmente consumida.
- Antes/después del escritor: conteos concordantes y positivos; en cada milisegundo, primero/último/mínimo/máximo después de escribir coinciden con el valor prescrito. Se confirma la frontera CSR instrumentada, no todos los consumidores PN del programa.
- El control de cantidad coincide con su plantilla nativa dentro del criterio declarado: error relativo máximo 1,34×10⁻⁸ en L y 3,61×10⁻⁹ en R. Se verificaron contrato, receta, fuente y referencias frente a sus identidades congeladas.

## Tres precisiones que afectan la interpretación

**1. El efecto pertenece a una vía de salida intervenida, bajo otras vías activas.** El escritor modifica el vector temporal FP32 que consume el CSR genérico. Las rutas especializadas siguen funcionando y pueden cambiar por recurrencia. Por tanto, el resultado demuestra que esta manipulación de la frontera genérica modifica el contraste de mando en este contexto. No identifica toda la mediación natural PN, una señal de dirección pura ni una fracción causal universal. `dose` preserva una suma escalar, pero redistribuye el patrón celular; tampoco conserva cada entrada postsináptica ni constituye un estímulo fisiológico equivalente. La reducción firmada supera el semicontraste original porque cambia de signo: no debe publicarse como «más del 100 % de la orientación explicada».

**2. El alcance temporal y experimental sigue siendo local.** Se reutiliza una única preparación 48/sham ya expuesta. Mantener una muestra por milisegundo es una intervención artificial cualificada para estos lectores y esta ventana; no reconstruye exactamente la señal natural de cada evaluación interna. El presupuesto y los criterios respetados permiten cerrar este diagnóstico, pero no generalizar a otras vidas, calibrar biología ni admitir las etapas. No hay razón para extender ahora esta misma vida buscando un resultado favorable.

**3. Los registros DNg agregados no identifican qué aferentes causan su déficit.** Comprobé además las capturas DNg100 de los ocho brazos: el objetivo final es cero en todas las evaluaciones registradas, incluidas las aceptadas de épocas comprometidas. En los brazos de 89 ms, incluso los máximos del margen `net+drive−theta` siguen por debajo de −1.335 y −1.241 unidades internas para los dos IDs. Son envolventes, no medias fisiológicas ni muestras independientes. El estado de estas neuronas no es exactamente cero en todos los registros: objetivo cero y actividad instantánea cero no son sinónimos.

Las capturas contienen sumas, parámetros y estados; no todos los operandos presinápticos individuales de cada evaluación. La tabla PN y el panel celular tampoco completan ese vacío. Para descomponer un grupo contextual hay que reutilizar operandos realmente guardados, como los de 49, o declarar la falta de cobertura. No reconstruir una cinta inexistente a partir de extremos y no emparejar ensayos adaptativos por índice: entre brazos cambian sus conteos y decisiones.

## Siguiente discriminador: C, con salida y límite concretos

Elegiría **un análisis de capacidad y contribución de aferencias clásicas hacia DNg100**, reutilizando la captura de operandos de 49 y las identidades/operadores compatibles que ya existen. La novedad debe ser la atribución por rutas contextualizadas; repetir la verificación de sus seis sumas, ya resuelta, no aporta otra discriminación.

Antes de ver el efecto, fijar un grupo de aferentes por identidad anatómica y evidencia contextual independiente de su actividad en estas corridas. Comprobar transmisión efectiva y correspondencia del operador. Para las evaluaciones que sí contienen sus operandos, calcular su aporte firmado y el del resto al margen DNg; separar explícitamente historias, fases y estados expuestos. No escoger a posteriori las células cuya activación produciría el signo deseado.

Como discriminador matemático barato, obtener también una cota del aporte directo permitido por el dominio del modelo. Si el coeficiente efectivo de una aferencia es `a_i = W_i*cap_i`, con `0 ≤ q_i ≤ 1`, el máximo genérico del grupo G es `Σ max(a_i,0)`, aplicando las mismas exclusiones del consumidor real. Compararlo con el déficit, dejando **sólo para esta cota instantánea** el resto del estado fijo.

- Si ese máximo ni siquiera puede abrir el objetivo, ese grupo no basta por acción directa en ese estado. Esto no descarta efectos recurrentes, cambios del resto de la red ni toda forma de contexto.
- Si tiene capacidad pero no contribución efectiva variable en los estados adquiridos, queda localizada una ruta candidata y su ausencia de reclutamiento. La cota sola no demuestra que un estímulo biológico pueda realizarla.
- Si faltan operandos o correspondencia celular, entregar esa limitación concreta. No suplirla con activación arbitraria, hambre ficticia, pesos nuevos o umbrales reducidos.

El producto de esta ronda sería una tabla breve de aporte real/capacidad/deficiencia de evidencia por ruta y una decisión falsable. Las envolventes actuales de DNg no bastan por sí mismas para ejecutar esa atribución. No propongo otro barrido ni una simulación larga como parte de este siguiente discriminador.

**A queda como confirmación posterior** si otra preparación va a cambiar una decisión de generalización o integración. **B permanece disponible** cuando exista correspondencia entre célula, estímulo y observable fisiológico capaz de distinguir leyes. La ausencia de avance no justifica por sí sola sustituir una ecuación. Elegir C aquí responde al siguiente cuello funcional; no convierte la iniciación en requisito para toda investigación de orientación.

Coincido con ambos ChatGPT en priorizar C y restringir la afirmación de mediación. Matizaría la frase de ASTRA «si los operandos no cambian, DNg100 no media ese efecto»: una comparación de resúmenes o muestras incompletas no establece esa exclusión global. La formulación defendible se limita al canal, estado y señal de salida efectivamente observados. Tampoco una contribución contextual nula en estas historias descarta el contexto en otras preparaciones.

## Coste y cierre

La adquisición original suma **558 ms CNS, ocho procesos y 1.310,5584375 s CPU** según sus recibos; queda dentro del presupuesto congelado. Esta revisión produjo cero CNS y no utilizó GPU, subagentes ni suites generales. Las dos reconstrucciones numéricas más el preflight anterior suman **0,7708122 s CPU instrumentados**; las lecturas breves, la inspección de metadatos y la redacción no están cronometradas individualmente.

Se conserva en el JSON una corrección del cálculo del revisor: una suma vectorizada alternativa cambió el orden de reducción al reconstruir `dose_R`. Usar el orden declarado recuperó identidad exacta. No fue un fallo del ensayo y no se modificaron tolerancias, estímulos ni código compartido.

**Cierre: diagnóstico causal parcial confirmado para la frontera y preparación declaradas; conducta útil y etapas 4/5 pendientes.**
