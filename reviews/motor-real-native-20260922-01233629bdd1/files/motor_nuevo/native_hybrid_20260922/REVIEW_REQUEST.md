# Revisión externa de la ronda nativa

Leer RESULTADOS.md, VERIFIED.json y los archivos reales. No inferir un PASS de los nombres `certified_*`.

1. A: `native_cell.py`, `cell_control.cpp`, `physical_events.cu`, `basis_compile.py`. ¿La cota dependiente de factores cubre el residuo contra el operador almacenado original? Preserva cada potencial de inversión por separado. La mejora medida es sólo 1,51× para el método temporal heredado.
2. Evento tardío: `graph_control_v2.cpp`, `check_event_boundary.py`, `EVENT_BOUNDARY_CHECK.json`; `graph_core.py`, `organism_adapter.py` en la carpeta de transporte. El evento tardío se reproduce y se corrige en CUDA, pero detener globalmente en cada evento eleva coste. `aligned_01` pasa la criba de 5 ms; no hay segundo completo.
3. B: `PLAN_B.json`, `ros_tableau.py`, `ros_step.cu`, `ROS_OPERATOR_CHECK.json`, `ROS_VS_BASE.json`. Se usó W aproximado diagonal por bloques, no tu Schur acoplado completo. Fue más lento y falló 1e-4 de estado global; se descarta esa implementación sin cambiar umbrales.
4. Rollback: `rollback_guard.py`, `check_organism_recovery.py`, `recovery_04/DIAGNOSTIC.json` y arrays control/retry. Estado neural serializado restaurado exactamente; la reanudación tras reconstruir runtime difiere por encima del criterio de ejecución. Se bloquea reutilización automática. Revisar propietarios/matrices que el serializador no capture, no darlo por arreglado. El grafo aislado sí revierte y continúa exactamente en su prueba.
5. Verificador: cinco contraejemplos reparados; conserva segmentación variable y exige forma/tipo/identidades/relojes. Newton=1 evita el grafo de dos correcciones y se probó con una entrada PN real.

Para la siguiente ronda, valorar matemáticamente integrar momentos de puertos analíticos (conservando causalidad y todos los eventos) dentro de operadores de flujo/conductancia, en vez de parar todo el cerebro en cada espiga. Un promedio sin cota no basta. Contrastar con un Jacobiano realmente acoplado y una sesión única que mantenga estado/operadores residentes. No proponer otra sucesión de microajustes del mismo kernel como solución global.

Separar fuente leída, ejemplo ejecutado y organismo no reproducido. El ZIP permite verificar arrays y operadores; faltan recursos estáticos para una ejecución del organismo independiente del árbol histórico.
