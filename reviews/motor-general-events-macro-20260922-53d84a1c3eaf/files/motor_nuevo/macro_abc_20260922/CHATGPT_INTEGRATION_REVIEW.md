# Revisión arquitectónica externa

Conversación 6ab06db7-9908-83e9-a515-58c9e6e18a1a, respuesta 475a149e-0eb1-4b9e-832a-37b0dc64eae8. Documental; no reprodujo los resultados nuevos.

**Recomiendo unificar primero un lazo real bajo IR, memoria, eventos y aceptación comunes; después comparar integración global frente a multirritmo.** La compilación del operador es una tercera ruta, pero no la elegiría por eliminar únicamente el 6,758 % de aristas sobrescritas.

Los conteos y tiempos nuevos son **datos comunicados, no reproducidos aquí**. ADD/SET corrige semántica; no constituye evidencia de aceleración.

## Tres rutas de integración

### 1. Sistema híbrido global compilado

Traducir las ecuaciones al IR y generar \(F\), JVP y aplicación de masa; integrar conjuntamente con SUNDIALS, interrumpiendo en las transiciones declaradas. `general_v2` ya contiene generación de RHS/JVP acoplado, pero su descriptor publicado rechaza capacidades distintas de ODE suave, puertos graduados y ediciones programadas. **Falta conectar eventos, no declarar que ya están soportados.**  

**Primer corte:** CNS recurrente real más un bloque de membrana con masa no diagonal, compuertas y emisión, representado por ecuaciones; el resto del organismo continúa mediante fronteras explícitas, sin eliminarlo.

**Falsador:** la resolución necesita cambiar masa, ignorar un término cruzado o redefinir el detector para funcionar. Riesgo económico: iteraciones globales y precondicionamiento más caros que el esquema actual. ARKODE admite masa no singular; eso no implica DAE general. :chatgpt-content-reference{index="2"}

### 2. Componentes del mismo IR con trayectorias iteradas

Cada componente integra sus ecuaciones utilizando entradas temporales provisionales; se recalcula la recurrencia hasta satisfacer consistencia y error. Un único propietario confirma estado e historia de eventos. **La partición cambia el algoritmo, no las conexiones.**

**Primer corte:** el mismo lazo CNS↔membrana↔puerto, con retorno activo. Comparar dos particiones del lazo, incluyendo los términos cruzados de masa cuando existan.

**Falsador:** converge localmente pero cambia el retorno, orden de eventos o trayectoria global fuera del contrato. Se descarta económicamente si las correcciones repiten tantos recorridos como el método global. La relajación de formas de onda neuronal es un antecedente, no evidencia de velocidad para nuestro organismo. :chatgpt-content-reference{index="3"}

### 3. Programa estático de operadores, conservando inicialmente la discretización

Compilar desde el IR un programa de primitivas matemáticas: reacción local, suma dispersa, masa, filtro y transición. Fusionar operaciones compatibles y mantener almacenamiento persistente; conservar inicialmente etapas e intercambios existentes. **No es el padre:** sustituye la ejecución mediante adaptadores especializados, sin cambiar simultáneamente el método temporal.

**Primer corte:** una evaluación completa del operador CNS real y su avance hasta la frontera siguiente, generados desde descripciones.

**Falsador:** requiere excepciones por nombre anatómico o pierde un lector intermedio al eliminar escrituras sobrescritas. NMODL demuestra la viabilidad del enfoque de compilación, no su rentabilidad aquí. :chatgpt-content-reference{index="4"}

## Secuencia corta, funcional

**Primero:** registrar identidades, layout, unidades, dominios por variable, masa, propietarios e historia pendiente en una sesión. Separar estado confirmado y provisional; los integradores reciben vistas, no mantienen copias científicas independientes. El adaptador publicado todavía transfiere estado y conserva propietarios especializados. 

**Segundo:** ejecutar ese lazo real con ADD/SET ordenados, rechazo y restauración, preservando el detector existente. Incorporar solamente los retardos efectivamente implementados; rechazar otras capacidades. La interfaz común no prueba su soporte.

**Tercero:** comparar rutas 1/2 como máximo dos prototipos completos, reutilizando compilación mínima común. Un modelo adicional debe entrar mediante ecuaciones y declaraciones, sin editar scheduler ni integrador.

## Criterio económico y dato pendiente

Elegir por **tiempo completo a igual error**, memoria, preparación y mantenimiento; no por aristas eliminadas. Exigir una reducción material prefijada y amortización en campañas reales. La optimización exclusiva del CNS no elimina el coste comunicado de membranas/PN.

No necesito otro ZIP para esta decisión. Falta revisar **ADD/SET nuevo y el enlace entre IR y layout/operador efectivo real**; si este último no existe, es la primera pieza ejecutable.

Releí `coupled.py`, las primeras 170 líneas de `model.py` y `organism_adapter.py`. **No ejecuté pruebas ni califico el motor como listo.**

