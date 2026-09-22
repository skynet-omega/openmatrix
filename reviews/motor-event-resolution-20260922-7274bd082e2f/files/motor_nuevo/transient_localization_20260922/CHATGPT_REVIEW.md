Revisión documental recibida e00c2f58-12fa-4314-a5b0-461749b3c6b9. Sin ejecución externa.

**Mantendría la base actual y la decisión B. La siguiente mejora útil no exige otra arquitectura: exige determinar si el transitorio procede de una historia física distinta o de ejecutar/publicar incorrectamente la misma historia.** Después priorizaría la política de pasos del CNS, antes que una partición extensa o un trasplante desde FastFly/flyvis.

**Esta revisión es documental.** Leí las fuentes nuevas indicadas y las dependencias citadas abajo. No ejecuté el motor, pruebas CPU/CUDA ni reconstruí los arrays; tampoco descargué el ZIP ni comprobé su SHA-256.

## 1. Qué queda sustentado por los archivos nuevos

Las reparaciones corresponden a los defectos anteriores: `OperatorState` autentica rutas contra el adaptador esperado, comprueba escribibilidad y contempla rollback/invalidez; `DeviceCell` rechaza reutilización tras un fallo de publicación; `RuntimeSession` selecciona explícitamente el ejecutor de membranas y exige fronteras de eventos. **No he reproducido esas pruebas, pero ya no basaría el diagnóstico en que el runner instala inadvertidamente la variante antigua.**   

El diagnóstico publicado identifica:

\[
234831=177758+57073,
\]

con valores **0,023326186953604493** y **0,023147598720978848**. Es una discrepancia en transmisión filtrada, no directamente en la coordenada de liberación. La identidad del operador y los conteos iguales no explican todavía esa diferencia ni determinan qué método es más preciso. 

**La observación que más acota el problema está en `event_ports.py`:** el puerto calcula directamente \(q,s\) desde su estado inicial, constantes y saltos fechados. `OrganismAdapter` anula después sus derivadas en el integrador general y vuelve a proyectarlos. Por tanto, **si 234831 pertenece efectivamente a `ports.sr` y no es sobrescrito posteriormente**, su valor puede comprobarse sin integrar todo el CNS. Primero verificaría ese mapa de propietario; la etiqueta anatómica `KCg-m` no basta.  

## 2. Tres hipótesis rivales, ordenadas por la primera frontera divergente

Pueden encadenarse. El diagnóstico debe localizar dónde comienza la diferencia, no escoger una por plausibilidad.

| Hipótesis | Firma que la distinguiría | Consecuencia |
|---|---|---|
| **H1. Historia de emisión distinta** | Con estado e inputs iniciales comparables, cambian las marcas o los saltos efectivos de la fuente, aunque el conteo final coincida. El filtro independiente reproduce la diferencia observada. | Investigar el detector y la trayectoria del propietario real de esa fuente. No atribuirlo automáticamente a `DeviceCell`. |
| **H2. Filtrado, reloj o publicación incorrectos** | Con **idénticos** \(q_0,s_0,\tau_q,\tau_s\), marcas, amplitudes e instantes de consulta, la implementación o el valor finalmente publicado discrepan de la respuesta independiente. | Es un defecto de ejecución: índice equivocado, origen temporal, buffer, dato antiguo o sobrescritura. No requiere una referencia fina del organismo para detectarlo. |
| **H3. Historia inicial o acoplamiento recurrente distinto** | La primera divergencia ya está en los estados/entradas entregados al propietario, antes de su avance: predictor restaurado, estado filtrado heredado, conductancias o interfaz de intercambio. | Comparar las fronteras predictor→restauración→avance aceptado. La identidad del operador final no demuestra igualdad de todos esos estados intermedios. |

H1 es compatible con el detector muestreado ya documentado, pero **todavía no está demostrada para la fila 57073**. H3 tampoco implica necesariamente un bug: puede ser error de discretización del acoplamiento. No declararía cuál rama es más precisa sin el contraste correspondiente.

## 3. Un falsador mínimo: replay del puerto 75907, no otra corrida para buscar PASS

Para el puerto declarado:

\[
q'=-q/\tau_q,\qquad s'=(q-s)/\tau_s,
\]

la respuesta es

\[
s(t)=s_0e^{-t/\tau_s}
+q_0\Phi(t)
+\sum_{t_k\le t}J_k\Phi(t-t_k),
\]

\[
\Phi(u)=
\frac{\tau_q}{\tau_q-\tau_s}
\left(e^{-u/\tau_q}-e^{-u/\tau_s}\right).
\]

Si ambas constantes coinciden, el límite es \(\Phi(u)=(u/\tau_s)e^{-u/\tau_s}\). Esta expresión se deriva del contrato de `FilterPorts`; no introduce otra fisiología. **\(J_k\) debe ser el salto realmente aplicado después del clipping, no una amplitud nominal recalculada.** 

Haría **un replay cruzado**, con las historias de ambas ramas, sobre una implementación independiente pequeña —por ejemplo, exponencial de la matriz triangular de dos estados entre saltos— y el puerto CUDA existente:

- Si cada historia reproduce su respectivo resultado y explica la separación, H2 pierde prioridad.
- Si una misma historia da resultados incompatibles, H2 queda localizada.
- Si las historias ya parten de estados distintos, se busca la primera frontera anterior: H3, antes de culpar al detector.

Compararía también el valor **antes y después de la publicación**, para distinguir cálculo correcto de sobrescritura posterior. Para este replay afín pueden comprobarse extremos interiores entre eventos, no únicamente el valor a 1 ms.

### El dato puntual que falta para realizarlo

No necesito todo el checkpoint. Falta una exportación pequeña de **ambas ramas**, por intercambio, para esa fuente. La llamaría `PORT_75907.json` —nombre propuesto, no archivo que suponga existente— e incluiría:

**Mapa de identidad y propietario**, \(q_0,s_0,\tau_q,\tau_s\), origen temporal, saltos efectivos con sus marcas, clasificación predictor/aceptado e instantes/valores realmente proyectados y finalmente publicados.

`CNS_TRACES.json` contiene marcas, fuentes y saltos de la rama instrumentada en las secciones examinadas; eso no aporta por sí solo el contrato completo de ambos lados. Los arrays finales tampoco permiten reconstruir unívocamente toda esa historia. 

**Esto no amplía retrospectivamente las cuatro cargas cerradas.** Primero aprovecharía lo ya registrado; cualquier captura adicional quedaría declarada como el siguiente diagnóstico acotado. La referencia fina de 1 ms se vuelve necesaria para ordenar precisión **si el problema son historias físicas distintas**, no para descubrir un puerto mal ejecutado.

## 4. Qué optimizar después: la propuesta de paso, antes de reorganizar todo el grafo

Los registros distinguen **118 ensayos aceptados en intercambios confirmados y otros 73 en predictores descartados**, sin rechazos. No son 118 evaluaciones del operador: según la contabilización de seis evaluaciones por ensayo del adaptador, representan **1.146 evaluaciones** entre ambos grupos. Eso describe trabajo, no su reparto exacto de tiempo.  

Hay una pista de coste más concreta que la alcanzabilidad CSR: tras aceptar, el controlador calcula la siguiente propuesta a partir del **paso ejecutado y recortado**:

```cpp
*next = min(maxstep, max(minstep, floor(h*1e9*(e < .1 ? 2 : 1))));
```

En el primer intercambio registrado, una propuesta de **31,25 µs** se recorta hasta aproximadamente **4,9902 µs** por un evento y la siguiente queda en **9,980 µs**. El controlador hereda así la reducción impuesta por la frontera y vuelve a crecer desde ella. El patrón concuerda con los **52 de 54** pasos limitados por propuesta que tuvieron error inferior a 0,1.   

**Mi siguiente rival de ejecución sería separar la propuesta nominal del recorte obligatorio por evento**, conservando las mismas fronteras y el test de error en cada nuevo intento. No lo presentaría como reparación inocua ni restauraría ciegamente un paso grande: cambiar la secuencia numérica puede introducir rechazos o aumentar el error.

Antes de conservarlo tendría que reducir trabajo y tiempo completo **sin empeorar el transitorio ni superar \(10^{-4}\)**. Los errores pequeños de los pasos recortados no autorizan a predecir el error de uno mayor.

Tampoco convertiría los 166.311 nodos alcanzables en una obligación de recalcularlos inmediatamente. El mapa publicado se declara parcial y carece de varias dependencias efectivas. **No alcanza para justificar ni descartar la partición.** 

## 5. Prioridad respecto de *awesome-fly*

**No contradigo tu selección de lectura; contradigo cualquier uso de ella como sustitución del solver.**

FastFly aporta ideas para compactar emisores y distribuir eventos, pero su implementación examinada usa LIF, pesos FP16 y acumulación FP32. No resuelve la semántica de nuestro filtro ni la evolución graduada recurrente. Su ordenación de trabajo sería una pieza posterior, no la explicación del transitorio. 

De flyvis conservaría la separación explícita entre nodos, aristas y dinámica. Su implementación predeterminada sigue utilizando `max(time_const, dt)`; no la importaría como integrador que preserve nuestras constantes al variar el paso. La documentación oficial reconsultada confirma esa política. :chatgpt-content-reference{index="15"}

**Para esta ronda no añadiría ninguna dependencia externa:** un replay independiente del puerto y una futura comparación de la política de propuesta cuestan menos y responden directamente a la limitación medida.

### Dictamen y alcance

**Lo que falta para avanzar es cerrar una cadena concreta: propietario → historia emitida → respuesta del filtro → valor consumido.** Después, demostrar que reducir ensayos globales conserva esa cadena y el error requerido. No hacen falta más ganancias fisiológicas ni otra arquitectura antes de esa comprobación.

Leí completos los ocho archivos iniciales solicitados, más `graph_control_trace.cpp`, `reconstruct.py`, `event_ports.py`, `organism_adapter.py` y `synaptic_visual_brain.py`. Consulté las primeras 210 líneas de `CNS_TRACES.json` y una sección del índice; **no reconstruí sus 191 registros desde principio a fin ni la discrepancia desde arrays**.

**Conservaría las reparaciones, mantendría `causal_cuda` sin promoción interpretativa y elegiría ese único replay como próximo falsador.** Si muestra que el puerto es correcto y las historias difieren, la siguiente pregunta será precisión temporal; si falla con historia idéntica, habrá localizado una reparación concreta antes de gastar otra trayectoria completa.
