Message e65ca581-d991-4089-bf10-fe26fdc5749c
Documentary design review only, no new-source execution.

**Usaría el bloque indirecto: es compatible con un grafo reutilizable y evita actualizar decenas de nodos.** Pero «todo en el mismo stream» no basta si algún nodo conserva un puntero de etapa capturado inicialmente. Revisión del contrato y de documentación CUDA 12.0; **no he examinado ni ejecutado vuestra implementación nueva**.

## 1. Fallo concreto: un destino capturado que el bloque no actualiza

Supón que los kernels leen correctamente `parametros->x`, pero la salida termina con una copia capturada equivalente a:

```cpp
cudaMemcpyAsync(destino_primera_llamada, temporal, bytes,
                cudaMemcpyDeviceToDevice, stream);
```

Actualizar `parametros->destino` **no cambia el destino de ese nodo memcpy**. Éste utiliza las direcciones registradas en sus propios parámetros. La segunda llamada podría sobrescribir el vector anterior y dejar intacto el actual, **con todos los valores finitos y el indicador en cero**. Es una deducción de la semántica documentada de los nodos memcpy, no un fallo observado en vuestro código. :chatgpt-content-reference{index="0"}

**Condición de aceptación:** toda operación que toque punteros variables debe resolverlos durante la ejecución desde el bloque GPU. Incluye lecturas iniciales y escritura final. Para esa última copia, sirve un kernel que lea `parametros->destino`; no hace falta actualizar el grafo.

Los kernels deben recibir **la dirección estable del bloque**, no una copia por valor de sus campos al capturar. CUDA copia los valores de los argumentos al construir los nodos; un puntero estable permite acceder después al contenido actualizado. :chatgpt-content-reference{index="1"}

## 2. Propiedad y sincronización que cerraría explícitamente

La secuencia exigida sería:

**Copia H2D del bloque → grafo completo → copia D2H del indicador → espera comprobada → retorno.**

El indicador debe depender de **todos** los productores relevantes y no reiniciarse entre suboperaciones. Comprobaría tanto el indicador como los códigos CUDA de lanzamiento/copia/sincronización: un cero antiguo en memoria no acredita que la ejecución haya terminado correctamente. Las operaciones del mismo stream están ordenadas; la sincronización es la que establece su finalización ante CPU. :chatgpt-content-reference{index="2"}

Hay otra obligación: **el bloque host utilizado por `cudaMemcpyAsync` debe permanecer vivo e inmutable hasta completar la copia**. En particular, memoria host fijada —*pinned*— no significa copia síncrona. El orden del stream no protege contra sobrescrituras realizadas directamente por CPU. :chatgpt-content-reference{index="3"}

**Un único bloque por sesión basta** si el callback no es reentrante y espera antes de permitir su reutilización. Con ese contrato, no añadiría doble buffering. Mantendría también vivos los vectores prestados hasta finalizar; no conservaría sus punteros como estado válido para la siguiente invocación.

## 3. Falsador barato, sin integrar el organismo

Captura una sola vez un kernel de prueba de 64 elementos:

\[
d_i=x_i+\gamma v_i+t.
\]

Realiza estas llamadas sobre asignaciones válidas y distintas:

| Llamada | Entradas | Resultado exacto esperado |
|---|---|---|
| 1, destino A | \(x_i=i,\ v_i=2,\ \gamma=0,5,\ t=0,25\) | \(i+1,25\) |
| 2, destino B | \(x_i=3-i,\ v_i=-4,\ \gamma=0,25,\ t=0,5\) | \(2,5-i\) |
| 3, mismos punteros que 2, contenido nuevo | \(x_i=1,\ v_i=0,\ \gamma=3,\ t=2\) | \(3\) |

Exige igualdad exacta, entradas intactas y que **A no cambie después de la segunda llamada**. Esta prueba detecta punteros capturados, escalares antiguos y reutilización incorrecta por dirección.

Añade una llamada con un NaN que se detecte antes de que una operación posterior escriba una salida finita: debe devolver fallo. La siguiente llamada válida debe funcionar, sin heredar ni borrar prematuramente el error anterior.

**Decisión:** continuar con el bloque indirecto, condicionado a estos contratos. La microprueba valida la interfaz; la mejora de rendimiento sigue decidiéndose en las corridas de segundos, incluyendo la copia del bloque y la comprobación final.
