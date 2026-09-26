# Contraste de asesorías y comprobación local

Se acepta de ChatGPT: reconstruir el lector por brazo antes de restar resultados, respetar DN previa y baseline fijo, comprobar que el primer mando diferencial posible es1002, separar criterio físico y atribución neural, registrar velocidad longitudinal firmada respecto al eje real del cuerpo, y conservar la prueba activa/criterios. El complemento CPU propio verifica lector/latencia/procedencia; no ejecuta la propuesta de cinemática firmada.

Se rechaza la alarma de excepción booleana: en el archivo real `analyze.py:57` hay `np.array_equal(...) and np.all(...)`, no el pseudocódigo abreviado con comparación matricial que se pegó en el encargo. No procede reparar ese archivo. `prefix` sí se exige en `analyze.py:55`. Error de transmisión en el extracto propio, no defecto de la otra sesión; se conserva el encargo original y la respuesta.

Se rechaza la sugerencia de que comprobar finitos cada100ms omite muestras intermedias: `run.py:136` comprueba `np.isfinite(value).all()` sobre arrays que contienen todas las filas del bloque cerrado. La comprobación posthoc independiente sigue siendo útil, pero este punto no es un bug del runner.

ChatGPT declara haber ejecutado sus propios contraejemplos en NumPy2.3.5 y consultado documentación1.26. Eso es evidencia declarada por el asesor, no una ejecución remota verificada por Codex. Nuestros tres contraejemplos y la reconstrucción usan el entorno local NumPy1.26.4. Movimiento lateral o ida/retorno son limitaciones del criterio compuesto congelado si se interpreta como avance neural; no autorizan cambiar ese criterio tras observar45.

Jev prioriza `relay_cpu_diagnosis` y luego `decoder_direction`. Confianzas declaradas0.37/0.56; probabilidades de elección0.58/0.70. Es una clasificación con una petición y cero reintentos, no medición, explicación causal ni aprobación científica. Coincide con un diagnóstico que ya quedó reconstruido localmente. Se mantiene el complemento del analizador por sus pruebas concretas, sin decidir por votación.

El diagnóstico del relé de yaw de12/13 no se traslada a la lectura de avance de45: esta última usa media y clip, sin ese EMA ni relé aplicado. Las rutas A/reclutamiento, B/lector y C/planta quedan diferenciadas. Uniformidad bilateral de45 impide acreditar dirección/distancia. No se afirma que las pérdidas de información del lector hayan causado por sí solas los negativos de navegación.

Clasificación del análisis CPU: **CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA**, únicamente para reconstrucción del lector/relé y contraejemplos del analizador aislado. Motor: **PROMETEDOR_NO_CONFIRMADO**, conserva el FAIL de12. Campaña45 todavía sin pareja completa al revisar; etapas4/5 abiertas, patas posteriores.
