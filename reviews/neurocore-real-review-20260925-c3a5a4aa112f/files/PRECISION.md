# Única comprobación de convergencia, después de las cuatro corridas iniciales

La pareja real de 100 ms falla CNS (3,5103/1), membrana, compuertas y reloj de
evento (10,323 ns/1 ns); conserva cantidades de eventos y PN. No se promueve
esa configuración. Una comparación con otra aproximación numérica no determina
por sí sola cuál se aleja más de la solución convergida.

Se ejecuta una única pareja adicional, mismos 100 ms/olor/preparado, reduciendo
rtol y atol del integrador CNS diez veces en ambos brazos. No cambian física,
ecuaciones, conectividad ni los límites de aceptación del comparador. No hay
una tercera precisión ni búsqueda de parámetros. Los núcleos y adaptadores
quedan idénticos; el runner añade esa opción y guardado de PN compacto.

Las cuatro corridas previas consumieron 1029,986 s de proceso. Quedan 770 s del
presupuesto original, con máximo seis corridas en total. La candidata tiene
límite de 355 s; antes de lanzar la referencia se asigna el remanente real del
presupuesto agregado, reservando 25 s para el guardado final. Se transfiere sólo
tiempo no consumido entre esos dos brazos. Si no alcanza, se conserva el prefijo y la limitación, sin
alargar esta campaña. Las trayectorias mantienen todos los estados y muestras;
el guardado compacto omite la copia repetida de anatomía y checkpoint completo.
No sirve para reanudar todo el organismo.
