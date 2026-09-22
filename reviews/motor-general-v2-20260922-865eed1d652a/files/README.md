# Motor general v2

[Resultado, límites, A/B/C y reproducción](RESULTADOS.md). [Contrato de campaña](PLAN.json). [Veredicto reconstruido](VERIFIED.json). [Extracción limpia](CLEAN_VALIDATION.json).

`model.py` declara las ecuaciones y conexiones; `runtime.py` conserva el explícito; `coupled.py` genera el RHS y JVP global; `bridge.cpp` e `implicit.py` enlazan SUNDIALS CUDA; `session.py` administra intervenciones y reinicios físicos por épocas.

Dos perfiles de precisión no equivalen a dos biologías. Admisión actual: ODE determinista suave y masa constante dentro de cada época; no eventos/ruido/retardos/DAE/cuerpo. Todos los resultados de esta ronda son ingeniería expuesta al autor. Etapa3 abierta.
