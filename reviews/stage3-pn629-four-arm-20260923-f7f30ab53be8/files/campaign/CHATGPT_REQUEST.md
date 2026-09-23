Nueva revisión acotada (REVIEW_PLAN.json, una consulta; la ronda14 de código ya terminó). Tu sonda C++ ya compiló aquí con g++/CUDA12.6; sus3 controles quedarán después de la campaña larga, sin competir porGPU.

Hay un avance causal real: al desactivar únicamente las629 salidas generales PN izquierdas antes de40ms de preparación, sham400ms cambió +0.075606563°→+0.023686321° y derecho +0.000218240°→−0.058872200°. Preparación/source exactas entre esos dos brazos. El derecho cumplió la mejora mínima prospectiva0.02°. Izquierda/uniforme siguen pendientes. Cuatro replays corporales cruzados, con controles propios exactos, atribuyen99.8% del cambio sham al comando. No son una retirada de lectura neural con cerebro cerrado: proprioception_enabled=True y el campo es espacial, aunque la exposición1/0 quedó constante en estas trazas.

Código y datos reales,45archivos:
https://raw.githubusercontent.com/skynet-omega/openmatrix/8b2daee39641c1fabeb7a5ceac82f4555d13f4e6/reviews/stage3-pn629-prospective-review-20260923-4680db263cbd/README.md
Incluye PLAN, trazaNPZ/flujoCUDA, scripts, fallo numérico previo y ORIENTATION_SCOPE_ANTECEDENT.md. Este último es el alcance histórico consultado, no órdenes nuevas: orientación reflejada, efecto frente a controles, retirada de lector, apoyo, precisión para la consulta y continuidad; no exige identidad microscópica universal.

Pregunta pesada: ¿cuál es la confirmación mínima HONESTA para pasar etapa3, sin seguir atrapados en perfección de166700estados ni ignorar una KC por llamarla memoria? Conserva tres rivales:
A) contrato prospectivo de robustez FUNCIONAL native/reference en400ms, manteniendo el antiguo FAIL de estado oculto;
B) usar reference_cuda como perfil preciso: conserva el NUEVO CNS C++/CUDA y PN; sólo cambia la membrana al reloj global previo;
C) corregir primero semántica de evento dependiente de muestreo.

DNb05 usa una escala efectiva explícita5°/s*tanh(250Δq), no una calibración fisiológica identificada. YangCell2024 apoya relación con marcha, no esa conversión exacta. No pedir identificación universal como puerta; tampoco quitar requisitos necesarios para obtenerPASS.

Entrega un contrato ejecutable pequeño (Python fuera del bucle caliente), con métricas/unidades/criterios y falsadores, que reciba pares de traces.npz/flow reales. Propongo discutir ANTES de nueva referencia400ms un error de yaw máximo0.002° (1/10 del efecto material0.02°), comprobar signo y margen contra sham/uniform, y registrar errores deDN/comando y discrepancias discretas sin esconderlas. No hemos observado aún diferencias entre motores a400ms: no escoger el criterio mirando esos errores. Explica si esta propuesta cambia legítimamente el alcance o sería rescate indebido. No declara la etapa pasada esta campaña15: es exploratoria por contrato. Recomienda una secuencia finita y controles causales realmente necesarios; podemos ejecutar referencia derecha primero y parar si falla, y conservar los resultados negativos.

Respuesta completa<=16000caracteres, código incluido. Distingue evidencia leída/ejecutada de propuesta. Sin nuevos ajustes de ganancia, conectoma, ventanas o tolerancias a partir de estos giros.
