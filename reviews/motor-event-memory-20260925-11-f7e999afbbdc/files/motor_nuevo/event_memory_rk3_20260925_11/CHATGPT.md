# Revisión de ChatGPT Motor

Conversación autorizada: https://chatgpt.com/c/6ab709cd-5f18-83e9-8276-636af9a5c59a . Se enviaron el diff, controlador completo, protocolo y antecedentes. Revisión estática, no ejecución independiente. Los marcadores de cita de su respuesta pertenecen al mensaje original.

**No encuentro recuperación indebida por fin de época, rechazo ni reloj adelantado en el diff. Mantendría la pareja de 100 ms, sin modificar factores.** Hay una precisión semántica importante y dos defectos heredados que conviene separar del porte.

### 1. El porte: correcto como «no reducir por corte»

Tu implementación calcula:

```cpp
next_ns = clamp(max(propuesta_normal, propuesta_anterior));
```

**No siempre conserva exactamente la propuesta anterior.** Ejemplo: propuesta anterior `1000 ns`, corte interior a `800 ns`, error `0.05` → siguiente propuesta **`1600 ns`**, no `1000 ns`. Con corte a `200 ns` sí recupera `1000 ns`.

Esto implementa **un suelo para la propuesta**, no una restauración estricta. No puedo asegurar identidad con la política antigua sin su implementación; sí señalar esta diferencia respecto de la descripción verbal. No cambiaría ahora el `max` por una asignación sin resolver esa distinción.

En los puntos que preguntas:

| Punto | Resultado de la lectura |
|---|---|
| **Fin de época** | `attempted_stop_s < end_s` lo excluye, incluso cuando hay un evento exactamente en `end_s`. |
| **Rechazo** | No recupera. El rechazo sustituye `next_ns` por `smaller`; el intento posterior lee ese valor reducido, no la propuesta rechazada original. |
| **Reloj** | Correcto: calcula `available_s` antes de modificar `used_s`. No depende del contenido de `clock` que haya utilizado el RK. |
| **Evento exacto** | `h == available_s` compara contra la misma resta usada en `prepare`; no necesita un epsilon. Si evento y propuesta coinciden sin recorte, `h < requested_s` impide recuperar. |
| **Mínimo/máximo** | La propuesta resultante queda acotada. Un paso efectivo menor que `min_ns` por frontera sigue permitido; si se rechaza y su mitad queda bajo el mínimo, termina con `accuracy_limit`, como antes. |

La propuesta recuperada **sigue pasando por recorte y estimación de error en el siguiente intento**; no se acepta anticipadamente.

### 2. Relanzamiento: no veo una carrera nueva

La cadena `prepare → child → decide → commit → relaunch` ordena las operaciones. El autorrelanzamiento por cola espera a que termine el entorno actual; la lectura posterior en el mismo stream espera la cadena completa. Un único hilo lanza una única continuación: coincide con el mecanismo documentado por NVIDIA. **No añadiría sincronizaciones ni fences al porte.** :chatgpt-content-reference{index="0"}

### 3. Dos defectos heredados del controlador

**Fallo en `prepare` no cancela el `child`.** Ante `trial_limit` o `no_progress`, el grafo estático ejecuta igualmente el ensayo, potencialmente con el reloj anterior. `decide` y `commit` impiden comprometer `fine`, pero no protegen otros efectos laterales del modelo. La corrida debe descartarse ante ese fallo, como ya exige tu protocolo.

**La propuesta inicial puede entrar bajo el mínimo.** El contrato admite `0 < *next_ns < min_ns`; `prepare` sólo limita por arriba. Para esta pareja, comprueba que la entrada cumple el mínimo. No mezclaría una corrección del controlador heredado sólo en 11.

### 4. Protocolo: precisar el criterio antes de medir

El JSON fija exactamente:

```text
funcional_pass
&& avance11 <= 0.90 * avance07
&& proceso11 <= proceso07
```

**Eso exige 10 % en avance, no en proceso.** La frase comprimida admite ambas lecturas; mantendría el criterio del JSON ya fijado, sin reinterpretarlo después del resultado.

Además, `same_applied_yaw_command_each_tick` sólo cubre yaw: **si existen otros mandos aplicados, también deben compararse**, conforme a tu requisito de mismos mandos.

El orden fijo y una sola pareja sirven como criba, pero no separan variante de calentamiento/cachés. Registra aceptados, rechazados y activaciones de recuperación junto con los tiempos: una mejora temporal sin activaciones no demostraría beneficio de esta regla.

**Continuaría con los checks locales y la pareja prevista. Esta revisión no verifica el `child` RK, las proyecciones izquierda/derecha ni los resultados del organismo.**

