# Etapa 2 del motor: membranas en el núcleo reutilizable

**PASA la pareja real de 100 ms.** Clasificación:
CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA, restringida a esta compatibilidad numérica. Se incorpora el
ejecutor de membranas junto al CNS previamente validado. PN sigue conservado;
no se declara motor universal terminado ni validación de navegación/vuelo.

## Cambio de fondo

El backend celular deja de construir código mediante sustituciones de texto
e importaciones de kernels históricos. Ahora separa: solución de matrices densas
FP64, control adaptativo por bloque, publicación de estado y ecuaciones físicas.
El modelo conserva sus 17 coordenadas, masa completa, canales y eventos.
Una época fallida no publica los bloques que sí terminaron.

El mismo núcleo de álgebra acepta 1–32 coordenadas. Se probó en 1, 3, 17 y 32,
y el mismo controlador integró otro modelo M dy/dt=f-Ky con masa densa no
identidad, sin editar el núcleo: error máximo 7.36920351e-07
frente a su solución analítica. Esto demuestra reutilización en esos modelos,
no suficiencia de este solver para cualquier sistema neuronal.

## Resultado empírico

Una preprueba real de 1 ms comprobó la integración. Se ejecutó la candidata
de 100 ms con olor lateral y se reutilizó la referencia congelada del cierre
anterior, con el mismo checkpoint, parámetros, estado inicial, pesos y PN.
La referencia pendiente no llegó a ejecutarse de nuevo. 166.700 neuronas canónicas, 359.373 estados CNS, 25.582.938 pesos;
101 muestras completas por brazo y todos los eventos registrados. La candidata
terminó con salida 0 y sin errores de limpieza; la referencia conservada también
había terminado así en su ronda original. Umbrales anteriores intactos.

| Comparación máxima | Diferencia | Límite |
|---|---:|---:|
| CNS normalizado | 0 | 1 |
| Voltaje de membrana, mV | 0 | 2e-05 |
| Compuertas | 0 | 2e-07 |
| Tiempo de evento, segundos | 0 | 1e-9 |

Veredicto: **PASS_COUPLED_NUMERICAL**. Coincidencia numérica exacta de todos los campos
comparados: **sí**. Se compararon
7,788 registros de evento, incluidos
predictores, y 44 campos PN finales. PAIR100.json conserva el detalle íntegro.
Esos registros no representan igual cantidad de espigas físicas distintas.

## Coste observado

| Medida | Referencia conservada | Candidata nueva |
|---|---:|---:|
| Avance del organismo, segundos | 637.988 | 563.007 |
| Tiempo celular instrumentado, segundos | 59.166 | 65.910 |
| Intentos celulares aceptados | 10,214,382 | 10,214,382 |
| Intentos celulares rechazados | 47,064 | 47,064 |

La carga CPU/GPU varió y las regiones instrumentadas de los adaptadores difieren.
Estos tiempos, de rondas distintas, no certifican aceleración causal. La referencia
pendiente se canceló tras comprobar la elegibilidad del control conservado, antes
de comparar. REFERENCE_REUSE.json registra la decisión; el plan inicial permanece
intacto. La candidata ya había finalizado: sólo se canceló su supervisor de
despacho para evitar otra corrida GPU durante la campaña36.

La candidata nueva más la preprueba consumieron 622.949 s de proceso,
frente al máximo previsto de 1.920 s. El control conservado consumió
670.036 s en la ronda anterior y no se cuenta como ejecución nueva. El compilador declara 136 bytes de
memoria local por hilo en el kernel celular nuevo frente a
0 de referencia; su efecto no quedó aislado.
La reducción CNS de seis a cinco evaluaciones por intento se conserva.
Una candidata nueva, un control congelado reutilizado, sin ajuste de tolerancias.

## Ayuda externa

La tarea elegida «ChatGPT C++/Cuda», identificada por la app como Codex, desarrolló
la solución de matrices leyendo el código real. No ejecutó CUDA. Se conserva
su original y la adaptación local de inclusión de cabeceras para NVRTC. El primer
intento local de compilación falló por math.h del host; se corrigió esa inclusión
sin cambiar aritmética y se conservó el registro del fallo.

Jev realizó una consulta de prioridades: favoreció separar núcleo/modelo y
comprobar matriz física y rollback. Es asesoría, no código ni certificación.
Localmente pasaron controles de operadores inválidos, residual desde originales,
masa no identidad, dimensiones distintas y recuperación tras fallo parcial.

## Decisión y reproducción

Se conserva esta base para integrar PN y el acoplamiento sobre contratos explícitos, seguido de validación de mayor duración.
Los ZIP incluyen fuentes, contrato y resultados. El paquete de evidencia permite
recalcular la pareja desde las trayectorias; reejecutar el organismo requiere el
modelo/checkpoint históricos. Los snapshots PN compactos no son reinicios completos.

[Código](/mnt/c/Users/gonza/Documents/Codex/2026-09-25/pu/outputs/Motor_CUDA_etapa2_codigo.zip) ·
[Evidencia completa](/mnt/c/Users/gonza/Documents/Codex/2026-09-25/pu/outputs/Motor_CUDA_etapa2_evidencia.zip)
