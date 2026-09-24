# Interfaz CuPy→CUDA: fallo conservado y reparación acotada

El primer candidato real (`real_resident_01/RESULT.json`) se detuvo antes de avanzar el CNS: `resident_create` recibió un `cudaGraph_t` nulo y devolvió `invalid resident graph contract`. El control pareado (`real_reference_01/RESULT.json`) completó 1 ms desde el mismo checkpoint. No hay dato de paridad ni rendimiento del candidato 01.

La causa concreta está en [CuPy 13.6.0, `graph.pyx` líneas 11–19](https://github.com/cupy/cupy/blob/v13.6.0/cupy/cuda/graph.pyx): `Graph._init` destruye el grafo fuente después de crear el ejecutable y pone `self.graph = 0`. El puente previo pasaba `self.graph.graph`, por lo que la API nativa rechazó correctamente el valor nulo. Esto es un bug de vida útil de la interfaz, no un rechazo matemático del plan multirritmo ni evidencia de que CUDA no soporte la prueba real.

El segundo intento captura una fuente nueva mediante las API `streamBeginCapture/streamEndCapture`, pasa el handle vivo al controlador nativo y lo destruye sólo después de que éste haya construido su grafo hijo. No cambia ecuaciones, estados, eventos, tolerancias ni el controlador C++/CUDA. Se congela un presupuesto nuevo para repetir sólo el brazo afectado; el primer resultado fallido permanece intacto.
