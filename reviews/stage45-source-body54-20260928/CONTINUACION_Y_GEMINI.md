# Continuación de etapas4/5 y contraste con Gemini — 28-09-2026

**Gemini recoge un efecto real, pero lo interpreta demasiado lejos.**52 ya publicó activación y avance basal con G/I. Eso justifica una candidata funcional para estudiar; no demuestra navegación, no identifica una ley fisiológica correcta y no autoriza declarar admitidas las etapas con los umbrales propuestos.

## Lo comprobado directamente

El contrato y el código51/52 definen **G=conductance** e **I=current**. No significan Gaussiana e Instantánea. I conserva un término de derivación basal; es un control de corriente de ingeniería. G/I emplean un objetivo normalizado y una compensación elegida para conservar el punto basal. Esa compensación puede quedar fuera del intervalo de potenciales de reversión; no debe llamarse fuga pasiva fisiológica. Tampoco se registraron aquí disparos de DNg100: las cifras citadas son máximos del objetivo normalizado entre evaluaciones del integrador, incluidas etapas intermedias.

Recomputación de52 en `RECOMPUTACION_52.json`:

|Condición|Máximo objetivoDNg100|Avance medio51–90ms(mm/s)|
|---|---:|---:|
|G sin olor|0,0964745293|0,0276292011|
|G con olor|0,0964871355|0,0276325825|
|I sin olor|0,3547830820|0,0502098608|
|I con olor|0,3548234149|0,0502147457|

La diferencia específica por olor es muy pequeña frente al cambio basal de ley. El hallazgo no estaba omitido: consta en los resultados y decisiones51/52. El experimento52 estimulaba las694ORN con un perfil uniforme entre lados; colocar una fuente a la izquierda exige integrar una frontera espacial real, además de activar el giro.

El trabajo de [Destexhe y colaboradores(1994)](https://papers.cnl.salk.edu/PDFs/An%20Efficient%20Method%20for%20Computing%20Synaptic%20Conductances%20Based%20on%20a%20Kinetic%20Model%20of%20Receptor%20Binding%201994-2988.pdf) describe un modelo cinético de unión a receptores y cálculo eficiente de conductancias. Sustenta estudiar cinéticas de receptores; no valida automáticamente estas fórmulas normalizadas ni la reinterpretación de I como conductancia instantánea.

El otro adjunto propone hambre como disminución global de inhibición y aumento uniforme del reposo. [Root y colaboradores(2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073827/) muestran modulación presináptica específica relacionada con insulina/sNPF en ORN, incluida la vía Or42b/DM1. No proporcionan licencia para esos porcentajes globales ni para sumar50unidades a todas las neuronas. El estado interno sigue siendo una alternativa, pero requiere locus, unidades y observables independientes identificados.

## Pruebas realizadas antes del nuevo CNS

Se restauró el cuerpo real desde48 y su caché de contactos. Cuatro replays físicos de90ms, con giro desactivado, reprodujeron **exactamente todas las poses y velocidades** de52. En otros cuatro se aplicó el mando angular registrado.0msCNS nuevos;11,369sCPU medidos,286MB de máximoRSS.

I produjo aproximadamente0,211° de rotación con o sin olor. Una fuente hipotética a la izquierda mejora y la fuente espejo empeora: un criterio que examine sólo la primera puede seleccionar el sesgo basal. Los sensores de esos replays no alimentaban un CNS vivo; la prueba confirma capacidad de actuación, no feedback ni etapa4. Fuentes, ocho trazas y resultado están en `physical_01`.

Después se comprobó la frontera espacial con el cuerpo y mapa anatómico reales. Hay323ORN de lado izquierdo y371 derecho entre las694 seleccionadas; las78 de lado desconocido permanecen fuera de esta intervención, no se borran del cerebro. Se conserva exactamente la actividad basal y se escala sólo el incremento por la concentración de su antena. Concentración1 recupera el perfil uniforme; concentración0 conserva la basal. Una rotación de±2° cambia concentraciones y entradas sin recurrir al error angular como mando. Las sumas anatómicas difieren entre lados y se declaran, no se corrigen retrospectivamente.

La campaña53 de Motor terminó en paralelo. Emparejar la sumaJO trasFP32 no elimina el contraste neuronal del aire; no apareció avance ni se aplicó giro. Esa pista no es una prueba nueva de orientación. Sus cuatro brazos y los seis de54 pertenecen a contratos y fuentes distintos.

## Mi plan y los planes de los asesores

Las propuestas propias se escribieron en `PLAN_INICIAL.md` antes de las respuestas nuevas. Todas conocen resultados previos; no son una evaluación ciega. No se crearon subagentesCodex. PRO no fue verificado ni se atribuye ejecución de arrays a ChatGPT.

|Autor|Tres alternativas sustanciales|Cómo influyeron en la acción|
|---|---|---|
|Codex/Matrix Astra|A: conservar ley padre y distinguir entrada/contexto; B: I funcional con controles basales y espejo; C: cualificar sensor, lector, cuerpo y feedback|Se ejecutó la criba física, se comprobó el sensor y se integraron padre/I como máximo dos prototipos.|
|ChatGPT ASTRA_V2|Corte causal de la víaJO→AMMC/WED; feedback corporal con replay discriminante; cinética local de receptores después de localizar el corte|Exigió fuente fija en el mundo, antenas actuales y beneficio en ambas fuentes. No impuso un corte neural como puerta universal.|
|ChatGPT_Motor_V2|Trasplante recíprocoPN vivo; sonda finita dependiente del estado; prueba de lector→cuerpo|Apoyó medir primero actuación barata. Corrigió, tras nuestra objeción, su replay potencialmente degenerado.|
|Motor C++/CUDA|Cuerpo y feedback; interfaz espacial real; transmisión/estado|Comprobó el mapa anatómico694 y revisa las unidades/temporización de54 sin duplicar el ejecutorGPU.|

Las respuestas originales y correcciones quedan guardadas en los cuatro archivosCHATGPT*.json y `aporte_motor`. Son propuestas o revisión de código/evidencia visible, no votos que demuestren corrección.

## Piloto54 ejecutado como siguiente paso

Seis brazos: ley padre/I × sin olor/fuente izquierda/fuente derecha, todos desde el mismo estado completo48. Campo gaussiano fijo en coordenadas del mundo; las posiciones reales de las antenas se vuelven a muestrear cada1ms. Avance y giro provienen del lector neural histórico, sin ajustar pesos, ganancias ni umbrales para obtener éxito. La relación concentración→incrementoORN es un proxy de ingeniería, no una calibración de1-hexanol.

La primera ejecución detectó una confusión de unidades en **mi adaptador nuevo** antes de comprometer el primer milisegundo científico. Se preservó el fallo; la reparación sólo convirtió la señal al formato esperado por el controlador. Se comprobó sobre360mandos y se repitieron las cualificaciones. Se redujo cada brazo de90a89ms para que las dos ejecuciones juntas no superen los544ms intentados originalmente presupuestados. No se modificaron los mínimos de efecto.

El resultado definitivo y todas sus cifras se generan desde arrays en [campaña54](../RESULTADOS.md); este informe no sustituye su verificador. El contrato original y la reparación permanecen separados. No extrapolar una criba de89ms a imposibilidad de respuesta tardía. Tampoco escalar automáticamente a una vida de1s si el único efecto es giro basal.

## Qué se necesita después

Una comparación online/replay de la propia cinta desde el mismo estado determinista puede reproducir exactamente la misma trayectoria aunque se corte el enlace acción→sensor. No discrimina la utilidad del feedback por sí sola. Usaremos una cinta nominal obtenida **antes** de una perturbación reservada, restaurando el mismo estado y aplicando esa perturbación tanto a online como a replay. Sólo online recibirá información actualizada de su consecuencia. La ventaja angular positiva es error_replay−error_online, no la resta contraria.

El viento sensorial y el impulso mecánico deben distinguirse. Recuperar el error después de una fuerza requiere controles físicos para separar acción neural de relajación pasiva. Los umbrales0,5° y2° de Gemini son propuestas nuevas; no reemplazan retroactivamente contratos ni bastan para declarar4/5. Una candidata puede merecer un ensayo más largo sin tener equivalencia biológica.

La decisión posterior concreta, dependiente de todos los brazos54, queda en `campaña54/DECISION.md`: continuar sólo la explicación que sobreviva al discriminador, conservando transmisión/estado y contexto/lectura como rivales. Mantener orientación con cuerpo actual, CNS→VNC acotado y patas completas después. No hay necesidad demostrada de ajustar manualmente cada conexión ni de instalar otro escáner universal para interpretar esta ronda.
