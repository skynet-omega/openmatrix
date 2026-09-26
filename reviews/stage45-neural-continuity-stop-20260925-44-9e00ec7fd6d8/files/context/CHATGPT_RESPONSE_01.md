**Selecciono A, sin cambiar amplitud, ventana, motor ni umbrales.** Estoy expuesto a40–43 y al control que propuse. Intenté acceder al paquete mediante GitHub y RAW, pero **no conseguí leerlo**: reviso únicamente tu resumen, no certifico los hashes ni audito archivos. Tampoco tengo verificación del selector PRO.

Veo **tres posibles confusores que invalidarían la pareja**:

### 1. Cinta correcta, pero fila, canal o instante de consumo incorrectos

El hash acredita los bytes fijados, no que el CNS consuma los valores previstos. Exigiría un testigo de frontera con:

`world.time_ns → instante de muestreo → paso de consumo → fila de cinta → L/R realmente entregados`.

Con la convención \(k\leftarrow k-1\), comprobar especialmente que **el primer cambio se consume en1021, no en1020 ni1022**;1001–1020 deben ser originales y deben entrar las100 filas previstas. No asumir que el reloj absoluto empieza en cero. Varias consultas al wrapper con el mismo tiempo deben devolver la misma fila, sin avanzar un cursor por llamada.

Esto puede comprobarse antes del CNS con el mismo wrapper. Después debe atestiguarse en la frontera real, tras conversiones de tipo o recortes: canales correctos, media emparejada y ninguna ruta olfativa adicional leyendo el campo original.

**Importante:** la media se empareja en la frontera de concentración definida. No hay que imponer igualdad de la media downstream de ORN: una transformación no lineal puede convertir la redistribución bilateral en un cambio común neural; eso sería parte del efecto, no contaminación.

### 2. El sham no recorre la misma implementación, o el mando “en sombra” modifica algo corporal

**El sham debe utilizar el mismo wrapper y recorrido de ejecución, alimentado con cinta original.** Un sham que omite el wrapper no valida su temporización ni sus efectos secundarios.

Comprobar restauración de la muestra sensorial pendiente, colas de eventos sinápticos, estados del lector/filtros y estado físico/controlador pertinente. En particular, conservar en las tres ramas la misma política histórica de restauración: no recalcular auxiliares en una sola. MuJoCo distingue el estado de integración de las cantidades derivadas, y señala la relevancia del *warmstart* para reproducción exacta; comparar sólo `qpos/qvel` es insuficiente como comprobación de restauración. :chatgpt-content-reference{index="0"}

Durante la ejecución, contrastar **órdenes y fuerzas efectivamente aplicadas en cada subpaso**, además del cuerpo. El nuevo mando no debe escribir en el controlador corporal, actualizar dos veces un filtro compartido ni modificar una propiocepción calculada a partir de la orden.

El testigo propioceptivo decisivo es **el vector externo realmente entregado al CNS**, con identidades de canales y tiempo de consumo, después del procesamiento sensorial pertinente. Guardar también sus muestras retenidas, retardos y filtros. Igualdad corporal no sustituye esa comprobación; tampoco bastan cuatro contactos y un mínimo de upright.

### 3. RNG iguales al inicio, pero ruido distinto consumido por cada neurona

Exigiría dos testimonios: **estado completo del generador y del proceso de ruido al restaurar**, y **realización exógena efectivamente consumida**, identificada por paso y neurona/canal. Para ruido temporalmente correlacionado, incluir su variable interna anterior; una semilla no la representa.

Si se utiliza cuRAND, registrar los estados y posiciones pertinentes: sus funciones de distribución pueden consumir cantidades distintas de números básicos, de modo que una semilla compartida no garantiza el emparejamiento por destinatario. :chatgpt-content-reference{index="1"}

El ruido testigo debe copiar lo ya generado, **no obtenerse haciendo otra extracción RNG**. Si la dinámica transforma ruido mediante un estado neural que diverge, emparejar las innovaciones exógenas; no forzar artificialmente igualdad del resultado endógeno. Si el ruido está desactivado, atestiguar la configuración efectiva y la entrada nula.

### Qué identifica la pareja y cuándo parar

Si pasan esas comprobaciones, la diferencia ORN/PN/DN/mando identifica **el efecto causal de sustituir una cinta por otra con igual concentración media y distinto reparto bilateral**, condicionado a esa preparación, trayectoria prescrita y realización de ruido. No identifica una derivada respecto del error angular, una ruta sináptica exclusiva, corrección corporal ni aprendizaje. La salida motora es **mando calculado**, no conducta realizada.

Mantener explícita la señal de mando preregistrada y las tres ventanas. Hay una ambigüedad que resolver **antes de ejecutar**: «sin signo obligatorio» requiere que el criterio final sea \(\left|\operatorname{media}(\delta u)\right|\ge0,02°/s\), no una comparación sólo positiva ni \(\operatorname{media}|\delta u|\). Comprobar qué dice exactamente el plan hashado; no reinterpretarlo después del resultado.

**Parada:** fallo del test de frontera o del sham ⇒ no lanzar los brazos. Divergencia no olfativa entre brazos ⇒ pareja inválida para esta atribución, aunque el mando supere el umbral material. No completar un fallo ampliando presupuesto. Un resultado válido sobre el piso numérico pero bajo el umbral material será detectable, no material según el criterio fijado; un nulo sólo acotará esta intervención de100ms.
