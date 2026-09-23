# Motor general: eventos preservados y decisión macro

**PROMETEDOR_NO_CONFIRMADO. Etapa 3 abierta.** Reparación del contrato de eventos comprobada hasta20ms del organismo real; no aceleración material ni motor universal. Se conserva la arquitectura declarativa existente en general_v2 como infraestructura que falta enlazar con las ecuaciones del organismo; no se sustituye por un solver PN nuevo.

## Evidencia reconstruida

- Fallo previo localizado: puerto q de coordenada28296 acaba en1.0000000000000002. Antes de proyectar la historia está en dominio; targets válidos. Los operandos FP64 guardados dan1+1.55708e-16 aun usando90decimales. La representación por diferencia aditiva perdió el valor posterior de la saturación física.
- ADD/SET transporta el valor posterior calculado por el productor. No modifica su detector, ecuaciones, saturación original ni tolerancia; no recorta desde el solver. Fuente física y marcas/saltos anteriores idénticos bitabit en el fixture. Otros productores aún usan ADD; no se declara inmunidad universal a futuros fallos de eventos.
- El mismo ejecutor admite dominios por variable: señal0..1, voltaje negativo, variable positiva sin cota superior artificial. Pruebas contra exponencial matricial, eventos simultáneos no conmutativos, consultas retrocediendo en tiempo y rollback al infringir dominio. No prueba de conservación química, DAE ni retardos genéricos.
- Fina:20ms completos, 112.373s de avance, 130.261s totales. Guardia:20ms, 73.028s de avance, 89.819s totales. Antes, la fina fallaba durante el paso15.
- Discrepancia normalizada guardia/fina:1ms 2.18344054e-06;5ms 1.74167221e-05;20ms 3.95945168e-06. Discretos comparados iguales. El screen histórico parcial pasa;64 campos físicos cambiados carecen de criterio completo en ese screen. No son64bugs ni una admisión científica. No se certifican todos los eventos intermedios ni tiempos largos por estas instantáneas.
- Reparación frente al mismo método anterior: máximos2.22e-16 y1.11e-16 a1/5ms. La continuación20ms es nueva evidencia; no afirmar equivalencia a un extremo antiguo inexistente.

## Decisión de rendimiento y arquitectura

Cálculo real:166700neuronas,25582938aristas base,359373estados. Las filas potencialmente sobrescritas cubren6.758% de las aristas base; esto no prueba que todas sean eliminables. Perfil aislado GPU: un ensayo de seis evaluaciones cuesta aproximadamente12.134ms; recorrido base,1.341ms. Seis recorridos representan66.3% del ensayo. Son mediciones aisladas del layout real, no ganancia del organismo.

Guardia20ms: membranas17.832s, PN4.763s, resto50.434s. Aunque el CNS fuera gratis, esos dos componentes suman18.83veces el presupuesto objetivo1.2s/20ms. No basta eliminar una fracción pequeña de CSR ni culpar a captura/MuJoCo sin perfil. La captura total fue0.0212s.

Alternativas vigentes (el padre no cuenta):

1. A: sistema híbrido implícito global desde IR, masa no diagonal y JVP real. Puede perder por iteraciones/callbacks; general_v2 ya contiene un antecedente lento que no debe repetirse como descubrimiento.
2. B: componentes del mismo IR con trayectorias provisionales y corrección recurrente, aceptación transaccional. Riesgo: repetir recorridos hasta agotar el ahorro o publicar una historia inconsistente.
3. C: compilar operador efectivo y programa de primitivas desde dependencias versionadas, con memoria persistente. Sigue válido para generalidad, pero quitar solo sobrescrituras no justifica la ganancia material2× fijada.
4. D adicional: integración de orden alto; no se asume que13etapas ganen a6 cuando mandan eventos y precisión local.

Próximo corte ejecutable: enlazar el IR común con un lazo real CNS↔membrana↔puerto, con retorno, masa y detector conservados; resto del organismo presente en fronteras explícitas. Probar un segundo modelo mediante ecuaciones sin editar scheduler. Comparar A/B sobre ese mismo operador y mantener C rival. Necesita nueva ronda con presupuesto; las cuatro cargas actuales se consumieron. No hay candidato macro implementado ni proceso de organismo pendiente.

## Autocrítica y revisión

El proyecto mantiene dos vías todavía separadas: IR general para ODE suave y adaptadores del organismo con eventos. Presentar el benchmark rápido del primero como velocidad del segundo fue una generalización injustificada. Reparar interfaces es necesario para mediciones fiables, pero no reemplaza resolver esta separación arquitectónica. Los modelos deben declarar unidades, dominios, masa, estados, eventos y dependencias; los backends deben exponer qué familias soportan. «Cualquier biología» no equivale a un único solver que acepte silenciosamente cualquier ecuación.

ChatGPT formuló tres alternativas y revisó la reparación y el problema de integración. Leyó fuentes publicadas pero no ejecutó arrays ni aprobó el motor. Réplicas exactas en CHATGPT_*; propuesta local congelada antes de la nueva respuesta (con exposición previa declarada). Jev: una petición real, HTTP403, sin clasificación; no usar su respuesta anterior como voto nuevo. Sin subagentes Codex.

## Presupuesto, negativos y reproducción

Cuatro cargas, 342.505s agregados frente a1200s y240s por carga; una implementación numérica nueva, no dos integraciones. El diagnóstico original también sufrió un error al intentar guardar un checkpoint durante un reloj incoherente; el payload original sí quedó guardado. Se reparó el manejo del diagnóstico después, sin reejecutar solo para empaquetar. No existe checkpoint completo reanudable del fallo.

El ZIP incluye fuentes, fixtures autónomos, datos de esta ronda, controles necesarios, hashes y diff. Reproducción corta desde extracción limpia:

```bash
PY=/home/daroch/miniconda3/envs/GPU/bin/python
$PY -B motor_nuevo/macro_abc_20260922/reproduce_short.py
```

Reconstruye comparación desde arrays y repite fixtures CPU/CUDA, compilando el controlador local. No vuelve a simular el organismo ni valida reanudación corporal. Para reproducción completa de las cuatro cargas en el entorno local se conserva el árbol histórico estático (solo lectura); el ZIP no incluye toda anatomía/checkpoints/cuerpo ni finge ser una distribución completa de la mosca. Comandos originales en REPRODUCE_FULL.md. No amplían automáticamente el presupuesto de la campaña cerrada.
