# Revisión externa posterior al prototipo

Resumen, no transcripción. Conversación autorizada6ab06db7-9908-83e9-a515-58c9e6e18a1a, turno9485b087-166b-498b-b490-bc6451950a85, respuesta035a3671-dd05-4a90-bb6c-f72ea3dd02e9. Consultado después de recibir el código CPU; todavía no había visto los nuevos arrays50ms.

ChatGPT distingue el ahorro por eliminar barreras globales de la relajación de trayectorias: son resultados diferentes. Con los tiempos20ms comunicados calcula que hacer gratis todo excepto membranas yPN sólo permite1.59× adicional. Sugiere C como infraestructura mínima compartida y A/B como métodos numéricos rivales dentro de ella.

- A: generar residuo, JVP completo y precondicionamiento de M−gammaJ desde ecuaciones; medir ensamblado, factorización, sustitución, rechazos y detector. Pierde si precisión de eventos limita pasos o Jacobiano cuesta más que lo ahorrado.
- B: pasos locales y trayectorias con retorno corregido antes de confirmar. Comparar particiones y número total de integraciones/recorridos. Pierde con retorno fuerte, interpolación cara o dependencia de partición. El negativo CPU conserva valor; no atribuirle aceleración.
- C: compilar ecuaciones, constantes, lecturas/escrituras temporales y transiciones, conservando etapas para aislar implementación. Pierde si no elimina trabajo o si fusión empeora ocupación/registros. NMODL es antecedente, no fuente de una aceleración extrapolable.

Propone un lazo real CNS↔membrana↔puerto con retorno y otro bloque de ecuaciones distintas bajo la misma semántica. Rechazar capacidades todavía ausentes. No ejecutar tres motores completos ni convertir un replay de futuras entradas de referencia en acoplamiento causal.

Declara investigación documental, sin código nuevo, lectura de arrays nuevos ni aprobación del motor/etapa3. El puenteIR CPU local posterior es evidencia propia y no se atribuye a esta revisión.
