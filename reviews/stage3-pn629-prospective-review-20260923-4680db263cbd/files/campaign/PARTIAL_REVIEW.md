# Evidencia parcial para una decisión prospectiva

Sham y olor derecho terminaron400ms con166.700 neuronas y cuerpo. Preparación40ms
sin olor y fuentes idénticas. La bandera de629 destinos generales PN izquierdos
se desactivó antes de preparar;466 destinos dinámicos siguen activos. No hubo
ajuste de ganancias ni anatomía. Izquierda/uniforme aún están en ejecución.

| Condición | Padre400ms | PN629 desactivado400ms |
|---|---:|---:|
| Sham | +0,075606563° | +0,023686321° |
| Derecho | +0,000218240° | −0,058872200° |

El criterio prospectivo de mejora del derecho≥0,02° se cumplió. No es admisión de
etapa3. Cuatro replays corporales cruzados atribuyen99,800% del cambio sham al
comando,~0,200% al estado preparado y una interacción~2,84e−12°. Los controles
propios reprodujeron exactamente qpos/qvel/yaw. Ver datos y código adjuntos.

El motor nuevo tiene el mismoCNS C++/CUDA y PN en ambos perfiles; `reference_cuda`
cambia la membrana al reloj global previo, `causal_cuda` usa adaptación local.
Comparar perfiles a20ms falló en historia KC y mostró diferencias continuas.
La captura posterior encuentra inputs iguales y pasos distintos; aritmética con
igual paso coincide a1e−12. No se ha medido la discrepancia motora a400ms.

La pregunta no es reetiquetar ese FAIL. Es elegir, antes de nuevas corridas, qué
prueba numérica y causal basta para orientación de este modelo con escala motora
efectiva declarada. Una certificación universal de toda química/memoria no es lo
mismo que robustez de esta consulta. Tampoco basta ignorar KC por llamarla memoria.

El lector DNb05 usa5°/s×tanh(250Δq). Su papel en marcha tiene respaldo primario;
la conversión cuantitativa de `q` es provisional. Los filtros de los autores
relacionan fluorescencia/velocidad y no identifican directamente esa conversión.
La etapa4 es aproximación a una fuente con realimentación; todavía no se admite.
