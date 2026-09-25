# Integración empírica del motor — 25-09-2026

Una candidata: RK3(2) FP64 con control residente CUDA y campo completo del
modelo actual. El núcleo admite un RHS CUDA capturado, no conoce anatomía,
targets conductuales ni nombres de propietarios. La representación por grafo
permite ejecutar las ecuaciones actuales sin reescribirlas antes de medir;
la aritmética del integrador es la de la candidata Neurocore ya validada.
El controlador de grafo genérico del 24 se reutiliza con procedencia conservada.
Las membranas espaciales y PN conservan por ahora sus solvers y masa originales
en el adaptador de comparación. Esto prueba integración CNS acoplada real;
no se llamará reescritura completa de todos los solvers neuronales.

No se normaliza W, no se recorta estado, no se altera el modelo ni se mezclan
preparaciones. Se usa el checkpoint asentado existente y entrada sham/olor
declarada. La otra sesión estable sigue independiente. Cero agentes.

Rivales ya estudiados: explícito residente (elegido para esta integración),
JVP/Krylov y agenda asíncrona. No se prototipan los dos últimos en esta ronda.

Presupuesto total de evaluación: 1800 s de CPU/GPU de procesos medidos;
seis corridas principales como máximo (pares 20 ms, 100 ms y extensión),
18 GiB RSS por proceso, 8 GiB VRAM incremental, 3 GiB de evidencia nueva.
La compilación y reparaciones de implementación se registran por separado.
Una corrección de bug puede repetir sólo el brazo afectado. No cambiar puertas
para rescatar una estrategia numérica. Tiempos bajo GPU compartida se etiquetan.

Comparar en el mismo preparado: máximo |xA-xB|/(1e-7+1e-5*max(|xA|,|xB|))
<=1 para el CNS, sin excluir coordenadas. Conservar estado de membrana,
compuertas, q/s, contadores y eventos para verificar que los propietarios
siguen acoplados. Para igualdad exigida: formas, relojes, RNG y cantidades
discretas exactos. Tiempo de evento <=1e-9 s; voltaje celular <=2e-5 mV;
compuertas <=2e-7; otros flotantes de propietarios <=1e-4 relativo a max(1,abs).
No confundir una puerta local de estado con equivalencia biológica.

Primera pareja: 20 ms sham del organismo completo con entrada sensorial real
del preparado; conservar trayectorias por ms y snapshot de sesión inicial/final.
La referencia mantiene su método anterior; la candidata sustituye únicamente
el avance CNS por RK3(2), con todos los overrides y eventos efectivos.
La segunda pareja de 100 ms procede sólo si pasa la primera. La extensión
hasta 5 s procede sólo con precisión y coste compatibles con el presupuesto.
El objetivo de velocidad 5 s / 600 s es una meta a medir, no puerta por kernel.

Un fallo produce estado FAIL/INCOMPLETE y evidencia conservada. Si la candidata
es correcta pero lenta, se conserva la implementación funcional con el cuello
medido; no abrir otra colección de sondeos en este mismo corte.
