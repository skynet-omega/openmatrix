# Fuentes que acotan la decisión de 56

Consultadas el 28-09-2026, después de los índices locales. Los artículos son restricciones y contrastes; no parámetros elegidos para obtener movimiento. Se conservan los archivos históricos sin modificarlos. La revisión de ambos ChatGPT y Motor precedió a los resultados factoriales completos.

## Transferencia ORN→PN

[Nagel et al., 2015](https://doi.org/10.1038/nn.3895) distingue los ajustes a EPSC evocados de su modelo de tasa continua. Los parámetros FAST/SLOW implementados proceden del segundo: el componente lento se ajusta a PN desinhibidas con un estímulo del 55 % de densidad. No corresponde tratar el benchmark de tasa de 56 como reproducción de un tren eléctrico de EPSC. Se reutilizó el texto local `work/sensory_dm1_20260909/Nagel2015_bioc.txt`, con procedencia en el inventario de Motor. El modelo publicado incluye inhibición presináptica con efectos sobre la liberación y la depresión. Que una simplificación tenga compresión sublineal no prueba que su ajuste sea excesivo para DM1.

**Decisión:** el banco B comprueba ecuaciones, historia y un control de recursos congelados; no recibe validación fisiológica. El control conserva basal y estado al inicio, pero no iguala el pico posterior de 20 ms. Para calibrar harán falta entrada temporal y observable biológico emparejados; no basta contrastar `q` con calcio o con EPSC normalizadas. [Kazama y Wilson, 2008](https://doi.org/10.1016/j.neuron.2008.02.030) y [Bhandawat et al., 2007](https://doi.org/10.1038/nn1976) son antecedentes pertinentes, no nuevas réplicas ejecutadas en 56.

## Identidad de DNg100 y significado del lector

[Sapkal et al., 2024](https://www.nature.com/articles/s41586-024-07854-7) aporta activación, silenciamiento e imagen de BDN2. Su activación inicia marcha hacia delante incluso en animales decapitados; el silenciamiento reduce la velocidad de avance. La actividad se correlaciona con velocidad hacia delante, no con velocidad angular. El estudio también diferencia detención mediante inhibición de vías descendentes y detención mediante circuitos del cordón nervioso.

La correspondencia publicada por [Virtual Fly Brain, DNg100](https://www.virtualflybrain.org/term/dng100-fbbt_20007473/) incluye BDN2 como sinónimo y los identificadores MaleCNS **10045 izquierda / 10056 derecha**, que coinciden con los observados en el proyecto. Es una correspondencia de identidad, no una calibración de `q`, umbral o ganancia.

**Decisión:** conservar el lector actual durante 56. La bibliografía da fundamento a la selección de DNg100 para estudiar avance; no prueba que dos tasas y una transformación lineal reproduzcan toda la locomoción. No sustituirlo por la neurona que dé un resultado favorable. La alternativa C debe contrastar mecanismos de excitabilidad o contexto con evidencia independiente, conservando el balance y la ley consumidos como observables.

## VNC y patas

El código local de [Pugliese y colaboradores](https://github.com/smpuglie/Pugliese_2026) declara simulaciones de VNC y datos externos separados. Tener esos fuentes no equivale a tener una integración CNS→VNC completa. El [preprint de control central y periférico de la marcha, versión 2](https://www.biorxiv.org/content/10.64898/2026.04.29.721658v2.full) describe activación de DNg100/BDN2, DNg97/oDN1 y MDN en preparados decapitados y con retroalimentación periférica reducida. Eso apoya separar iniciación descendente y coordinación de patas; no demuestra navegación hacia olor en nuestro modelo.

**Decisión:** mantener VNC como interfaz acotada y las seis patas como línea posterior. No introducirlas para ocultar el bloqueo anterior al mando.

## Límites de acceso y alcance

La apertura directa de algunas páginas PMC/Nature falló o mostró protección de acceso; la consulta de fragmentos indexados de la fuente primaria sí recuperó el pasaje pertinente de Sapkal. No se descargaron nuevos paquetes masivos ni se adquirió una serie biológica numérica emparejada para B. Los resultados de 48 y 55 son datos locales de simulación y se identifican separadamente de los datos biológicos. Ninguna respuesta externa se presenta como auditoría de archivos que el asesor no examinó.

## Análisis propio de identificabilidad

La causa inmediata de target cero en los registros anteriores está localizada: margen negativo antes de la rectificación. Eso no identifica por sí solo la causa biológica. En una célula genérica con `target=max(0,tanh(gain·(Wq+drive−theta)))`, escalar conjuntamente `W`, `drive` y `theta` por un factor positivo y dividir `gain` por ese factor conserva la misma función, si no intervienen otros términos o límites. Es una ambigüedad matemática local, no una afirmación de equivalencia para todas las rutas especializadas del cerebro. Mientras el target siga exactamente cero, muchas combinaciones distintas son además indistinguibles por ese observable.

El escáner y el predictor pueden localizar términos y efectos de intervención del modelo, pero esos registros no eligen automáticamente pesos fisiológicos únicos. Un diccionario de patrones tampoco elimina esa ambigüedad. Hará falta un contraste que atraviese regímenes informativos con un observable biológico correspondiente; no ajustar cada conexión hasta que el cuerpo avance.

Antes de las nuevas respuestas externas mantengo tres explicaciones para C, expuesto a las discusiones anteriores: (1) entrada descendente contextual ausente, con predicción de cambios selectivos de una vía documentada; (2) balance efectivo de excitación/inhibición mal restringido, con predicción distinta ante intervención específica de esa vía y controles emparejados; (3) función de transferencia o correspondencia de unidades de la célula genérica inadecuada, con predicción de otra relación entrada–respuesta aun cuando se fije la entrada. Identidad anatómica y número de sinapsis no separan por sí solos estas opciones. La siguiente elección debe preservar tres alternativas y como máximo dos instrumentos, con restricciones obtenidas independientemente del movimiento deseado. No se implementa un tercer prototipo en 56.
